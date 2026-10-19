"""
Face recognition off the frame loop (2026-10-19).

``DriverRecognizer.identify()`` takes ~1 s per call on the Pi (973 ms in the
2026-09-29 profile), and dlib / face_recognition holds the GIL for the whole
call: with recognition on a *thread* the main loop still stalled up to
384 ms (measured 2026-09-29). So a thread cannot take it off the frame loop;
:class:`RecognitionWorker` runs it in a separate **process** instead. The
frame loop hands it a frame (:meth:`~RecognitionWorker.submit`, a ~1 MB
pipe write) and picks the result up on a later frame
(:meth:`~RecognitionWorker.poll`); it never waits for it.

The child is started with the ``spawn`` method - a fresh interpreter, never
a ``fork`` of a process holding the camera, GPIO and GUI threads - and is
given the known encodings (:class:`KnownFaces`), not an API client: it does
no network I/O and touches no hardware. It ignores SIGINT (the launcher's
STOP goes to ``main.py``, which closes it).

Never an orphan, never two (2026-10-19): only one worker may be live at a
time (:meth:`RecognitionWorker.start` refuses a second). ``main.py`` closes
it on END SESSION and on every exit path (:meth:`RecognitionWorker.close`
escalates join -> terminate -> kill). If ``main.py`` dies without closing it,
the child goes too: on Linux the kernel sends it SIGTERM when its parent dies
(``PR_SET_PDEATHSIG``), and on any platform it exits when its pipe breaks or
its parent pid changes. The launcher additionally kills ``main.py``'s whole
process group once ``main.py`` has exited.

:class:`InlineRecognition` has the same interface but calls ``identify()``
in place, blocking as before. It is the fallback when the worker could not
start or failed, and what the tests use with fake recognisers.
"""

import logging
import multiprocessing
import os
import signal
import sys
import time
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# (tag, match or None, faces found or None, seconds the call took)
Result = Tuple[Any, Optional[Dict[str, Any]], Optional[int], float]


class KnownFaces:
    """
    Picklable factory for the worker: rebuilds a ``DriverRecognizer`` with
    the parent's encodings in the child process.
    """

    def __init__(self, encodings: Dict[int, List[float]], names: Dict[int, str],
                 tolerance: float) -> None:
        self.encodings, self.names, self.tolerance = encodings, names, tolerance

    @classmethod
    def of(cls, recognizer: Any) -> "KnownFaces":
        return cls({int(i): [float(x) for x in e] for i, e in recognizer.encodings.items()},
                   dict(recognizer.names), float(recognizer.tolerance))

    def __call__(self) -> Any:
        import numpy as np

        from modules.face_recognition_module import DriverRecognizer
        recognizer = DriverRecognizer(api_client=None, tolerance=self.tolerance)
        recognizer.set_encodings({i: np.asarray(e, dtype=np.float64)
                                  for i, e in self.encodings.items()}, self.names)
        return recognizer


def _die_with_parent(parent_pid: int) -> None:
    """Linux: have the kernel SIGTERM this process when its parent dies."""
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            libc.prctl(1, signal.SIGTERM)          # PR_SET_PDEATHSIG
        except (OSError, AttributeError):
            pass
    if os.getppid() != parent_pid:                 # parent died before the prctl
        os._exit(0)


def _serve(conn: Any, factory: Callable[[], Any], parent_pid: Optional[int] = None) -> None:
    """Child process: build the recogniser, then answer one frame at a time."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if parent_pid is not None:
        _die_with_parent(parent_pid)
    try:
        try:
            recognizer = factory()
        except BaseException as exc:        # report why, then exit
            conn.send(("error", repr(exc)))
            return
        conn.send(("ready", None))
        while True:
            msg = conn.recv()
            if msg is None:
                return
            tag, frame = msg
            t0 = time.monotonic()
            try:
                match, err = recognizer.identify(frame), None
            except Exception as exc:        # one bad frame must not kill the worker
                match, err = None, repr(exc)
            conn.send(("result", (tag, match, getattr(recognizer, "last_face_count", None),
                                  time.monotonic() - t0, err)))
    except (EOFError, OSError):             # pipe broken: main.py is gone - so are we
        return


class RecognitionWorker:
    """
    ``identify()`` in a child process: one request in flight at a time.

    Args:
        factory: Picklable callable returning a recogniser (``identify(frame)``,
            ``last_face_count``) - :class:`KnownFaces` in production.
        start_timeout_s: The child must report ready within this.
        result_timeout_s: A request unanswered for this long fails the worker
            (it is terminated; callers fall back to :class:`InlineRecognition`).
    """

    blocking = False
    # The one live worker (start() refuses a second while it runs).
    _live: ClassVar[Optional["RecognitionWorker"]] = None

    def __init__(self, factory: Callable[[], Any], start_timeout_s: float = 60.0,
                 result_timeout_s: float = 15.0) -> None:
        self.factory = factory
        self.start_timeout_s = start_timeout_s
        self.result_timeout_s = result_timeout_s
        self._ctx = multiprocessing.get_context("spawn")
        self._conn: Any = None
        self._proc: Any = None
        self._started_mono: Optional[float] = None
        self._pending: Optional[Tuple[Any, float]] = None     # (tag, sent at)
        self.ready = False
        self.failed = False
        self.calls = 0
        self.max_call_s = 0.0

    def start(self) -> None:
        """
        Spawn the child (returns at once; it reports ready on its own).

        Raises:
            RuntimeError: if another worker is still live - there is only
                ever one.
        """
        live = RecognitionWorker._live
        if live is not None and live is not self and live.alive:
            raise RuntimeError(f"a recognition worker is already running (pid {live.pid})")
        parent, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=_serve, args=(child, self.factory, os.getpid()),
                                       name="recognition", daemon=True)
        self._proc.start()
        RecognitionWorker._live = self
        child.close()
        self._conn, self._started_mono = parent, time.monotonic()
        logger.info("Recognition worker process started (pid %s) - re-identification runs "
                    "off the frame loop", self._proc.pid)

    @property
    def busy(self) -> bool:
        return self._pending is not None

    @property
    def alive(self) -> bool:
        """Whether the child process is running."""
        return self._proc is not None and self._proc.is_alive()

    @property
    def pid(self) -> Optional[int]:
        return None if self._proc is None else self._proc.pid

    def _fail(self, why: str) -> None:
        if self.failed:
            return
        self.failed, self._pending = True, None
        logger.error("Recognition worker FAILED (%s) - falling back to identify() on the "
                     "frame loop (blocks ~1 s per call on the Pi)", why)
        self.close()

    def _check_start(self) -> None:
        """Pick up the child's ready / error message without blocking."""
        if self.ready or self.failed or self._conn is None:
            return
        try:
            if self._conn.poll():
                kind, payload = self._conn.recv()
                if kind == "ready":
                    self.ready = True
                    logger.info("Recognition worker ready after %.1fs",
                                time.monotonic() - self._started_mono)
                else:
                    self._fail(f"could not start: {payload}")
                return
        except (EOFError, OSError) as exc:
            self._fail(f"pipe closed while starting: {exc!r}")
            return
        if not self._proc.is_alive():
            self._fail(f"exited while starting (code {self._proc.exitcode})")
        elif time.monotonic() - self._started_mono > self.start_timeout_s:
            self._fail(f"not ready after {self.start_timeout_s:.0f}s")

    def submit(self, frame: Any, tag: Any) -> bool:
        """
        Hand ``frame`` over for identification. ``False`` (nothing sent) while
        the worker is starting, busy or failed.
        """
        self._check_start()
        if self.failed or not self.ready or self.busy:
            return False
        try:
            self._conn.send((tag, frame))
        except (OSError, EOFError, ValueError) as exc:
            self._fail(f"send failed: {exc!r}")
            return False
        self._pending = (tag, time.monotonic())
        return True

    def poll(self) -> Optional[Result]:
        """The finished request's result, else ``None``. Never blocks."""
        self._check_start()
        if self._pending is None or self.failed:
            return None
        try:
            if not self._conn.poll():
                waited = time.monotonic() - self._pending[1]
                if waited > self.result_timeout_s:
                    self._fail(f"no result after {waited:.0f}s")
                elif not self._proc.is_alive():
                    self._fail(f"exited (code {self._proc.exitcode})")
                return None
            kind, payload = self._conn.recv()
        except (EOFError, OSError) as exc:
            self._fail(f"pipe closed: {exc!r}")
            return None
        if kind != "result":
            self._fail(f"unexpected message {kind!r}")
            return None
        tag, match, faces, seconds, err = payload
        self._pending = None
        self.calls += 1
        self.max_call_s = max(self.max_call_s, seconds)
        if err is not None:
            logger.warning("Recognition worker: identify() raised %s - treated as no match", err)
        return tag, match, faces, seconds

    def close(self) -> None:
        """
        Stop the child and wait until it has gone (idempotent): ask it to
        exit, then terminate, then kill. Never leaves it running.
        """
        conn, proc, self._conn = self._conn, self._proc, None
        self._pending = None
        if conn is not None:
            try:
                conn.send(None)
            except (OSError, EOFError, ValueError):
                pass
            conn.close()
        if proc is not None:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2.0)
            if proc.is_alive():
                logger.error("Recognition worker pid %s did not exit", proc.pid)
            else:
                logger.info("Recognition worker stopped (pid %s, exit code %s)", proc.pid,
                            proc.exitcode)
        if RecognitionWorker._live is self:
            RecognitionWorker._live = None


class InlineRecognition:
    """Same interface, ``identify()`` called in place (blocks the caller)."""

    blocking = True
    failed = False
    ready = True

    def __init__(self, recognizer: Any) -> None:
        self.recognizer = recognizer
        self._result: Optional[Result] = None

    def start(self) -> None:
        pass

    @property
    def busy(self) -> bool:
        return self._result is not None

    def submit(self, frame: Any, tag: Any) -> bool:
        t0 = time.monotonic()
        match = self.recognizer.identify(frame)
        self._result = (tag, match, getattr(self.recognizer, "last_face_count", None),
                        time.monotonic() - t0)
        return True

    def poll(self) -> Optional[Result]:
        result, self._result = self._result, None
        return result

    def close(self) -> None:
        pass
