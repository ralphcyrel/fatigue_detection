"""
Driver Fatigue Detection System - entry point.

Three phases, selected by the vehicle's ignition state (Module 11)::

    --enroll        Enrollment (operator-supervised, at hiring): face encoding +
                    60 s alert-state calibration -> Laravel backend.
                    ``--driver-id N`` skips the terminal prompt (used by
                    ``launcher.py``, which has no keyboard).

    ignition OFF    Pre-drive assessment: recognise the driver, fetch their
                    thresholds, run a 30 s assessment. PASS -> starter relay
                    released and monitoring begins straight away (no key
                    press). FAIL / not recognised / no baseline -> starter
                    stays inhibited and an operator override request is raised;
                    an approval releases it and likewise continues into
                    monitoring. This is the ONLY phase in which the relay engages.

    ignition ON     Continuous monitoring: same metrics, LEDs / buzzer /
                    backend notification only. The relay is never touched -
                    AlertManager refuses to lock in this phase (Module 8).
                    Entered on key-ON, or directly from a pre-drive release
                    (then it ends at the next ignition ON -> OFF, not while
                    the key has yet to be turned).

    --force-phase P tests one phase alone (a pre-drive release does not
                    continue); --sequence runs pre-drive -> monitoring with no
                    ignition input (the touchscreen launcher's default).

Per-frame metrics are computed by one shared pipeline (Module 12) in both
phases::

    camera -> LandmarkExtractor -> MetricsPipeline (EAR / MAR / blink / PERCLOS
              / yawn / head pose / FRS) -> phase-specific decision -> AlertManager

In both phases the loop also posts a heartbeat every
``config.HEARTBEAT_INTERVAL_SECONDS`` (``POST /devices/{id}/heartbeat``) carrying
the relay state as actually driven (``normal`` / ``interrupted``), so the
operator portal can show the unit as online and spot one that has not applied
a command.

Runs on the Raspberry Pi (Picamera2 + GPIO) and, with automatic fallbacks,
on a development machine (cv2.VideoCapture + mocked GPIO / ignition).
"""

import argparse
import logging
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import config
from modules.alert import AlertManager
from modules.api import (CALIBRATION_SOURCE_DEVICE, REPAIR_EAR_PAIR_SELF_SEEDED,
                         REPAIR_EAR_THRESHOLD_RECOMPUTED, REPAIR_PERCLOS_BASELINE_FLOORED,
                         SELF_SEEDED_PROVENANCE, APIClient, Calibration, repair_defaulted)
from modules.assessment import PredriveAssessment
from modules.blink import BlinkDetector, MicrosleepDetector
from modules.calibration import (CLOSED_CAPTURE_ATTEMPTS, EAR_THRESHOLD_RATIO,
                                 MIN_PERCLOS_BASELINE, SELF_SEED_S, CalibrationManager,
                                 ClosedEyeCapture, ear_calibration_problem, finite_or_none)
from modules.ear import EARCalculator
from modules.face_recognition_module import DriverRecognizer
from modules.head_pose import HeadPoseEstimator
from modules.ignition import ForcedIgnition, IgnitionSensor
from modules.landmark import LandmarkExtractor
from modules.mar import MARCalculator
from modules.perclos import PERCLOSCalculator
from modules.phase import LockReason, Phase
from modules.pipeline import HEAD_POSE_RELEASE_HOLD_S, FrameMetrics, HeadPoseDebounce, MetricsPipeline

logger = logging.getLogger("fatigue")

# ---------------------------------------------------------------------------
# Tunables local to the main loop
# ---------------------------------------------------------------------------

# Monitoring, driver not (yet) recognised: recognition is retried on the
# first face frame at least this long after the previous attempt ENDED.
# Each attempt blocks the loop for ~1 s on the Pi (dlib holds the GIL, so it
# cannot be threaded); retrying every frame held an unrecognised driver at
# ~1 fps, too blind for blink / microsleep detection. At 5 s the loop sees
# ~5 s of every ~6 s. Nothing is accepted between attempts - the driver
# simply stays unrecognised (no calibration, self-seeded baseline).
UNRECOGNISED_RETRY_S: float = 5.0

# While the driver stays in DANGER (monitoring), log at most one event per
# this many seconds - otherwise a 30 fps loop would POST 30 events per second.
DANGER_EVENT_INTERVAL: float = 5.0

# An open monitoring fault is re-POSTed (status "open", updated gap_s) this
# often, so the portal can tell a fault that is still live from one whose
# device went silent.
MONITORING_FAULT_REFRESH_S: float = 5.0

# NoFaceVerdict.event -> the fault_type it opens on POST /monitoring-faults.
FAULT_TYPE_FOR_EVENT: Dict[str, str] = {
    "fault_entered": "no_face",
    "latch_lost": "danger_latched",
}
# Opened when the EAR self-seed gives up (FrameMetrics.seed_failed).
FAULT_TYPE_NO_EAR_BASELINE: str = "no_ear_baseline"


@dataclass
class OpenFault:
    """A monitoring fault that has been opened on the backend and not resolved."""

    fault_uuid: str
    fault_type: str
    driver_id: Optional[int]
    entry_level: str
    started_at: datetime                  # UTC, when the face was lost
    last_known: Optional[Dict[str, Any]]  # FrameMetrics.last_known() before the gap
    gap_s: float                          # latest no-face duration seen
    last_push_mono: float                 # time.monotonic() of the last POST

    def push(self, api_client: APIClient, status: str, now_mono: float,
             resolved_at: Optional[datetime] = None,
             resolution: Optional[str] = None) -> None:
        """POST this fault's current state (fire-and-forget)."""
        api_client.push_monitoring_fault(
            self.fault_uuid, status, self.fault_type, self.driver_id, self.entry_level,
            self.started_at, self.gap_s, self.last_known, resolved_at, resolution,
        )
        self.last_push_mono = now_mono

    def refresh_elapsed(self, api_client: APIClient, now_mono: float) -> None:
        """
        Keep a fault that is not a no-face gap (``no_ear_baseline``) live:
        ``gap_s`` becomes the seconds since it opened, re-POSTed every
        ``MONITORING_FAULT_REFRESH_S``.
        """
        self.gap_s = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        if now_mono - self.last_push_mono >= MONITORING_FAULT_REFRESH_S:
            self.push(api_client, "open", now_mono)

# Pre-drive -> monitoring hand-off: after a pass or an approved override the
# relay is released and a green confirmation is shown on live frames for this
# long, so the driver sees the verdict, before monitoring starts. Kept under 2 s.
RELEASE_INDICATOR_S: float = 1.5

# Frames of face captured for the enrollment encoding.
ENROLL_FRAMES: int = 30

# Enrollment: wall-clock seconds of the driver's own EAR observed before
# calibration starts, to derive a personal closure threshold (see
# run_enrollment). Time-scoped so a slow loop settles for the same period
# as a fast one; SEED_MIN_SAMPLES guards a face that appears late.
SEED_WINDOW_S: float = 1.0
SEED_MIN_SAMPLES: int = 5

# Enrollment closed-eye capture (modules.calibration.ClosedEyeCapture): an
# on-screen countdown this long before each attempt, and this long with the
# eyes open again before the 60 s window starts, so the reopening (and any
# eye-rubbing) stays out of the alert-state baselines.
CLOSED_PROMPT_LEAD_S: float = 3.0
REOPEN_SETTLE_S: float = 2.0

# run_enrollment exit codes beyond the original 2 / 3 / 5.
EXIT_NO_EYELID_CONTRAST: int = 6    # closed-eye check failed; nothing POSTed
EXIT_CLOSED_EAR_DROPPED: int = 7    # POSTed, but the backend did not keep ear_closed_baseline

# Log the measured loop rate every N frames.
FPS_LOG_INTERVAL: int = 30

# Population values for the non-EAR baselines. Used in MONITORING when there
# is no usable calibration (driver unrecognised, no record, backend offline),
# and to fill fields missing from a partial backend record. Pre-drive never
# runs on them: a missing baseline there is a lock condition.
#
# There is deliberately no EAR baseline or threshold here. The old population
# pair (0.30 / 0.225) put the closure threshold at 88 % of a 0.255 EAR -
# inside landmark noise - and on 2026-09-23 drove 130 level changes in 71 s
# for an unrecognised driver. Without a calibration the EAR pair is seeded
# from the driver's own EAR (modules.calibration.EarSelfSeed), and until then
# only the overrides run.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "perclos_baseline": 5.0,
    "blink_duration_baseline": 150.0,
    "blink_frequency_baseline": 5.0,
    # Outer-lip MAR of a resting closed mouth is ~0.4-0.6; the yawn threshold
    # is 2x that (modules/mar.py). Drivers enrolled before yawn detection
    # existed have neither key in the backend and get these generic values.
    "mar_baseline": 0.45,
    "yawn_threshold": 0.90,
}

# check_threshold_sanity: how far ear_threshold / ear_baseline may stray from
# EAR_THRESHOLD_RATIO before the pair counts as inconsistent.
EAR_RATIO_TOLERANCE: float = 0.10

# Threshold keys that only exist for drivers enrolled with yawn detection.
MAR_THRESHOLD_KEYS = ("mar_baseline", "yawn_threshold")

# Text colours (BGR) per alert level for the overlay.
LEVEL_BGR: Dict[str, tuple] = {
    "ALERT": (0, 200, 0),
    "WARNING": (0, 220, 255),
    "DANGER": (0, 0, 255),
    "FAULT": (255, 0, 255),   # magenta: not a fatigue level
}
WHITE = (255, 255, 255)
GREY = (160, 160, 160)

WINDOW_NAME = "Driver Fatigue Detection"

# Preview-window keys.
KEY_QUIT = ord("q")
KEY_IGNITION = ord("i")   # mock mode only: toggle ignition
KEY_RETRY = ord("r")      # pre-drive: re-run the assessment

# Human-readable lock reasons for the overlay.
LOCK_REASON_TEXT: Dict[LockReason, str] = {
    LockReason.FATIGUE_DETECTED: "FATIGUE DETECTED",
    LockReason.MICROSLEEP_DETECTED: "MICROSLEEP DETECTED",
    LockReason.DRIVER_NOT_RECOGNIZED: "DRIVER NOT RECOGNIZED",
    LockReason.NO_BASELINE: "NO BASELINE ON FILE",
    LockReason.FOREIGN_DEVICE_BASELINE: "BASELINE FROM ANOTHER UNIT - RE-ENROL",
}

# ---------------------------------------------------------------------------
# Module-level handles so cleanup() can reach them from any exit path
# ---------------------------------------------------------------------------
_camera: Optional["Camera"] = None
_alert_manager: Optional[AlertManager] = None
_head_pose: Optional[HeadPoseEstimator] = None
_ignition: Any = None
_heartbeat: Optional["Heartbeat"] = None
# Preview window on. False under --no-preview, or once cv2.imshow has failed
# (headless); the draw_* / display_text helpers then skip their work too.
_display_available: bool = True
# LoopProfiler under --profile-loop (monitoring only), else a no-op.
_profiler: Any = None


class QuitRequested(Exception):
    """Raised when the user presses ``q`` in the preview window."""


class PhaseChanged(Exception):
    """Raised inside a phase loop when the ignition state no longer matches it."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Log INFO+ to the console and to a rotating ``logs/fatigue.log``."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # 5 MB x 3 backups keeps the SD card from filling up on a long-running Pi.
    file_handler = RotatingFileHandler(
        config.LOGS_DIR / "fatigue.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Camera abstraction
# ---------------------------------------------------------------------------

class Camera:
    """
    Uniform frame source over Picamera2 (Pi) or cv2.VideoCapture (elsewhere).

    Both back-ends deliver BGR ``uint8`` frames at ``config.CAMERA_WIDTH`` x
    ``config.CAMERA_HEIGHT`` so the rest of the pipeline is back-end agnostic.
    """

    def __init__(self) -> None:
        """Open Picamera2 if importable, otherwise fall back to OpenCV."""
        self.backend: str = "cv2"
        self._picam: Any = None
        self._cap: Optional[cv2.VideoCapture] = None

        try:
            from picamera2 import Picamera2  # type: ignore

            self._picam = Picamera2(config.CAMERA_INDEX)
            cfg = self._picam.create_video_configuration(
                main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"},
                controls={"FrameRate": config.CAMERA_FPS},
            )
            self._picam.configure(cfg)
            self._picam.start()
            self.backend = "picamera2"
            logger.info("Camera: Picamera2 %dx%d @ %d fps",
                        config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS)
            return
        except ImportError:
            logger.info("Picamera2 not available - falling back to cv2.VideoCapture")
        except Exception as exc:  # Picamera2 present but camera failed to open
            logger.warning("Picamera2 failed (%s) - falling back to cv2.VideoCapture", exc)
            self._picam = None

        self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {config.CAMERA_INDEX}")
        logger.info("Camera: cv2.VideoCapture(%d)", config.CAMERA_INDEX)

    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or ``None`` if capture failed."""
        if self._picam is not None:
            # Counter-intuitively, Picamera2's "RGB888" format is laid out in
            # memory as B,G,R - i.e. exactly what OpenCV expects - so no
            # channel swap is needed (see the Picamera2 manual, ch. 4.2).
            return self._picam.capture_array("main")
        ok, frame = self._cap.read() if self._cap is not None else (False, None)
        return frame if ok else None

    def release(self) -> None:
        """Stop the camera and free the device."""
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception as exc:
                logger.warning("Picamera2 release failed: %s", exc)
            self._picam = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def capture_frame() -> Optional[np.ndarray]:
    """Grab one frame from the global camera (Picamera2 or cv2)."""
    if _camera is None:
        return None
    return _camera.read()


def next_frame() -> np.ndarray:
    """Block until a frame is captured (retrying on transient failures)."""
    while True:
        frame = capture_frame()
        if frame is not None:
            return frame
        logger.warning("Frame capture failed - retrying")
        time.sleep(0.05)


class LoopRate:
    """Measured loop rate (frames / wall-clock seconds), logged every N frames."""

    def __init__(self, interval: int = FPS_LOG_INTERVAL) -> None:
        self.interval = interval
        self.fps: Optional[float] = None
        self._count = 0
        self._window_start = time.monotonic()

    def tick(self) -> Optional[float]:
        """Count one loop iteration; returns the last measured fps (or ``None``)."""
        self._count += 1
        if self._count % self.interval == 0:
            now = time.monotonic()
            self.fps = self.interval / max(now - self._window_start, 1e-6)
            self._window_start = now
            logger.info("Loop rate: %.1f fps (%d frames in %.2fs)",
                        self.fps, self.interval, self.interval / self.fps)
        return self.fps


class _NullSection:
    """Context manager that does nothing (profiling off / outside a frame)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return False


_NULL_SECTION = _NullSection()


class LoopProfiler:
    """
    ``--profile-loop``: where the monitoring loop's wall time goes, per frame.

    ``frame()`` is called once at the top of every loop iteration and closes
    the previous frame, so a frame's total is the full iteration time
    including every ``continue`` path, and the frame totals of a window add
    up to its wall time. Work inside a frame is attributed with
    ``with profiler.section(name):``. Sections may nest (``present()`` times
    the heartbeat inside the display section); a parent is charged only its
    own time, never its children's. Whatever no section covers is reported
    as ``other``.

    Every ``window_s`` of wall time a summary is logged and the window's
    frames are appended to a CSV (``logs/loop_profile_<UTC>.csv``, one row
    per frame, milliseconds).

    Only main-thread time is measured. The heartbeat / event / fault POSTs
    and the buzzer run on their own threads; their cost can only appear as
    slower main-thread sections (GIL contention), never under ``api``.
    """

    CATEGORIES = ("capture", "landmarks", "recognition", "pipeline", "gpio", "api", "display")

    def __init__(self, window_s: float = 60.0, label: str = "monitoring") -> None:
        self.window_s = window_s
        self.label = label
        self._csv_path = config.LOGS_DIR / (
            f"loop_profile_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv")
        self._csv_header_written = False
        self._rows: list = []                 # finished frames of the current window
        self._frame: Optional[Dict[str, float]] = None
        self._frame_start = 0.0
        self._window_start: Optional[float] = None
        self._stack: list = []                # [name, start, child_time] of open sections

    # ---- per-frame hooks ---------------------------------------------------

    def frame(self) -> None:
        """Close the previous frame (if any) and open a new one."""
        now = time.perf_counter()
        self._close_frame(now)
        if self._window_start is None:
            self._window_start = now
        elif now - self._window_start >= self.window_s:
            self.report(now)
            self._window_start = now
        self._frame = dict.fromkeys(self.CATEGORIES, 0.0)
        self._frame_start = now

    def section(self, name: str) -> Any:
        """Time a block under ``name`` (no-op outside a frame)."""
        if self._frame is None:
            return _NULL_SECTION
        return _Section(self, name)

    def stop(self) -> None:
        """Close the last frame and report the partial window (loop exit)."""
        now = time.perf_counter()
        self._close_frame(now)
        if self._rows:
            self.report(now)
        self._window_start = None

    # ---- internals ---------------------------------------------------------

    def _enter(self, name: str) -> None:
        self._stack.append([name, time.perf_counter(), 0.0])

    def _exit(self) -> None:
        name, start, child = self._stack.pop()
        elapsed = time.perf_counter() - start
        if self._frame is not None:
            self._frame[name] = self._frame.get(name, 0.0) + elapsed - child
        if self._stack:
            self._stack[-1][2] += elapsed

    def _close_frame(self, now: float) -> None:
        if self._frame is None:
            return
        row = self._frame
        row["total"] = now - self._frame_start
        row["other"] = max(0.0, row["total"] - sum(row[c] for c in self.CATEGORIES))
        self._rows.append(row)
        self._frame = None

    def report(self, now: Optional[float] = None) -> None:
        """Log the summary of the finished frames and flush them to the CSV."""
        rows, self._rows = self._rows, []
        if not rows:
            return
        wall = sum(r["total"] for r in rows)
        totals = np.array([r["total"] for r in rows]) * 1000.0
        lines = [
            f"LOOP PROFILE ({self.label}) - {wall:.1f} s wall, {len(rows)} frames, "
            f"{len(rows) / wall:.1f} fps",
            f"  frame time ms: mean {totals.mean():.1f}  p50 {np.percentile(totals, 50):.1f}  "
            f"p95 {np.percentile(totals, 95):.1f}  max {totals.max():.1f}  |  "
            f"frames > 100 ms: {int((totals > 100).sum())}",
            f"  {'category':<12} {'total s':>8} {'share':>7} {'frames':>7} "
            f"{'ms/frame':>9} {'ms/call':>8} {'max ms':>8}",
        ]
        for cat in self.CATEGORIES + ("other",):
            vals = np.array([r[cat] for r in rows]) * 1000.0
            ran = vals[vals > 0]
            lines.append(
                f"  {cat:<12} {vals.sum() / 1000:8.2f} {vals.sum() / 10 / wall:6.1f}% "
                f"{len(ran):7d} {vals.mean():9.2f} "
                f"{(ran.mean() if len(ran) else 0.0):8.2f} {vals.max():8.1f}")
        summary = "\n".join(lines)
        logger.info("\n%s", summary)
        try:
            config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            cols = ("total",) + self.CATEGORIES + ("other",)
            with open(self._csv_path, "a", encoding="utf-8") as fh:
                if not self._csv_header_written:
                    fh.write(",".join(f"{c}_ms" for c in cols) + "\n")
                    self._csv_header_written = True
                for r in rows:
                    fh.write(",".join(f"{r[c] * 1000:.3f}" for c in cols) + "\n")
        except OSError as exc:
            logger.warning("Could not write loop profile CSV %s: %s", self._csv_path, exc)


class _Section:
    """One timed block of a :class:`LoopProfiler` frame."""

    __slots__ = ("profiler", "name")

    def __init__(self, profiler: LoopProfiler, name: str) -> None:
        self.profiler = profiler
        self.name = name

    def __enter__(self) -> None:
        self.profiler._enter(self.name)

    def __exit__(self, *exc: Any) -> bool:
        self.profiler._exit()
        return False


class _NoProfiler:
    """Stand-in when ``--profile-loop`` is off: every hook is a no-op."""

    def frame(self) -> None:
        pass

    def section(self, name: str) -> _NullSection:
        return _NULL_SECTION

    def stop(self) -> None:
        pass


_NO_PROFILER = _NoProfiler()


class Heartbeat:
    """
    Periodic "still online" post to the portal, ticked from the frame loop.

    Fires every ``config.HEARTBEAT_INTERVAL_SECONDS`` of wall-clock time,
    the first one on the first tick so the portal sees the unit as soon as
    the supervisor starts. It is driven from the loop rather than a
    background timer on purpose: a stalled detection loop then stops
    heartbeating and the portal shows the unit offline. The POST itself
    runs on a daemon thread (``APIClient.post_heartbeat``), so a slow or
    dead backend never delays a frame.

    ``confirmed_state`` is read from ``AlertManager`` at send time - the
    relay as actually driven, not as commanded (``UNLOCKED`` -> ``normal``,
    ``LOCKED`` -> ``interrupted``). Without an ``AlertManager`` the state is
    reported as ``unknown``; in practice ``main()`` creates the manager
    (which drives the relay to its fail-secure state) before this ticker,
    so a real run never sends ``unknown``.
    """

    def __init__(
        self,
        api_client: APIClient,
        alert_manager: Optional[AlertManager],
        interval: float = config.HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self.api_client = api_client
        self.alert_manager = alert_manager
        self.interval = interval
        self._last_sent: Optional[float] = None

    def tick(self, now: Optional[float] = None) -> bool:
        """
        Queue a heartbeat if the interval has elapsed.

        Args:
            now: ``time.monotonic()`` for this frame (taken here if omitted).

        Returns:
            ``True`` if a heartbeat was queued on this tick.
        """
        now = time.monotonic() if now is None else now
        if self._last_sent is not None and now - self._last_sent < self.interval:
            return False
        self._last_sent = now
        relay = self.alert_manager.get_relay_state() if self.alert_manager else None
        self.api_client.post_heartbeat(relay)
        return True


# ---------------------------------------------------------------------------
# Display helpers (all tolerate a headless Pi with no monitor)
# ---------------------------------------------------------------------------

def show_frame(frame: np.ndarray) -> int:
    """
    Show ``frame`` in the preview window if a display is available.

    On a headless Pi (or with ``opencv-python-headless`` installed)
    ``cv2.imshow`` raises; the first failure disables the preview for the
    rest of the run so the loop keeps going, and says so loudly - once - in
    the log and on stdout.

    Returns:
        The key code pressed (``cv2.waitKey`` & 0xFF), or ``-1`` if none /
        headless.
    """
    global _display_available
    if not _display_available:
        return -1
    try:
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1)
        return key & 0xFF if key >= 0 else -1
    except cv2.error as exc:
        _display_available = False
        detail = exc.err if getattr(exc, "err", None) else str(exc).strip().splitlines()[-1]
        logger.warning(
            "PREVIEW WINDOW DISABLED for the rest of this run - cv2.imshow failed: %s. "
            "Detection continues headless. If you expected a window: 'not implemented' "
            "means the installed OpenCV has no GUI backend (opencv-python-headless) - "
            "run `pip uninstall -y opencv-python-headless && pip install opencv-python`; "
            "otherwise check DISPLAY / that you are on the Pi's desktop, not SSH.",
            detail,
        )
        print(
            "\n" + "=" * 72 + "\n"
            "  !! NO PREVIEW WINDOW - running headless for the rest of this run !!\n"
            f"  cv2.imshow failed: {detail}\n"
            "  Fix (GUI-less OpenCV):  pip uninstall -y opencv-python-headless && pip install opencv-python\n"
            "  Fix (no display):       run from the Pi desktop terminal, or set DISPLAY=:0\n"
            + "=" * 72 + "\n",
            flush=True,
        )
        return -1


def present(frame: np.ndarray) -> int:
    """
    Show ``frame`` and act on the global keys.

    ``q`` raises :class:`QuitRequested`; ``i`` toggles the mock ignition.
    Other keys are returned to the caller (e.g. ``r`` to retry).

    Every loop in both phases passes through here once per frame, so this
    is also where the :class:`Heartbeat` is ticked (a no-op until
    ``config.HEARTBEAT_INTERVAL_SECONDS`` have elapsed; never blocks).
    """
    prof = _profiler if _profiler is not None else _NO_PROFILER
    if _heartbeat is not None:
        with prof.section("api"):
            _heartbeat.tick()
    with prof.section("display"):
        key = show_frame(frame)
    if key == KEY_QUIT:
        raise QuitRequested()
    if key == KEY_IGNITION and _ignition is not None:
        if getattr(_ignition, "mock", False):
            new_state = not _ignition.read()
            logger.info("Preview key 'i': mock ignition -> %s", "ON" if new_state else "OFF")
            _ignition.set_mock_state(new_state)
        else:
            logger.info("Preview key 'i' ignored: ignition is read from hardware")
    return key


def draw_banner(frame: np.ndarray, text: str, color: tuple = WHITE) -> None:
    """Bottom strip with the phase / status text, drawn in place."""
    if not _display_available:
        return  # no preview: nobody sees the annotation
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    driver: Optional[Dict[str, Any]],
    m: FrameMetrics,
    fps: Optional[float] = None,
) -> None:
    """
    Annotate ``frame`` in place with driver name, FRS, level, EAR, PERCLOS,
    MAR / yawn state, relay state, measured loop rate and head pose, plus a
    MICROSLEEP banner while that override is active.

    Args:
        frame: BGR frame to draw on.
        driver: Dict from ``DriverRecognizer.identify()`` (may be ``None``).
        m: Pipeline output for this frame.
        fps: Last measured loop rate, or ``None`` before the first sample.
    """
    if not _display_available:
        return  # no preview: nobody sees the annotation
    level = m.level
    color = LEVEL_BGR.get(level, WHITE)
    name = driver["name"] if driver else "Unknown"
    font = cv2.FONT_HERSHEY_SIMPLEX
    w = frame.shape[1]

    # Dark strip behind the text keeps it legible over a bright face.
    cv2.rectangle(frame, (0, 0), (w, 135), (0, 0, 0), -1)

    cv2.putText(frame, f"Driver: {name}", (10, 25), font, 0.6, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}" if fps is not None else "FPS: --",
                (w - 150, 25), font, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FRS: {m.frs:.3f}" if m.scored
                else "FRS: -- (NO EAR BASELINE)" if m.seed_failed
                else "FRS: -- (learning EAR baseline)",
                (10, 50), font, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(frame, level, (w - 150, 50), font, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"EAR: {m.ear:.3f}", (10, 75), font, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"PERCLOS: {m.perclos:.1f}%", (10, 100), font, 0.55, WHITE, 1, cv2.LINE_AA)

    # Mouth column next to the eye metrics: raw MAR, then the yawn state
    # machine so a talking mouth ("open 0.4s", resets) can be told apart
    # from a confirmed yawn ("YAWNING", red) at a glance.
    cv2.putText(frame, f"MAR: {m.mar:.3f}", (200, 75), font, 0.55, WHITE, 1, cv2.LINE_AA)
    yawn = m.yawn_status
    state = str(yawn.get("state", "closed"))
    if state == "yawning":
        yawn_text, yawn_color = f"Yawn: YAWNING {yawn['open_s']:.1f}s", (0, 0, 255)
    elif state == "open":
        yawn_text, yawn_color = f"Yawn: open {yawn['open_s']:.1f}s", (0, 220, 255)
    else:
        yawn_text, yawn_color = "Yawn: --", (0, 200, 0)
    cv2.putText(frame, f"{yawn_text}  (n={yawn.get('count', 0)})", (200, 100), font, 0.55,
                yawn_color, 1, cv2.LINE_AA)

    relay = _alert_manager.get_relay_state() if _alert_manager else "n/a"
    cv2.putText(frame, f"Relay: {relay}", (w - 150, 100), font, 0.5,
                (0, 0, 255) if relay == "LOCKED" else (0, 200, 0), 1, cv2.LINE_AA)

    # Head pose row: angles on the left, status on the right. "ALERT" from
    # get_status_text() means a normal head, which reads better as "OK" here.
    pose = m.pose
    if pose is not None:
        angles = f"Pitch: {pose['pitch']:+.1f}  Yaw: {pose['yaw']:+.1f}  Roll: {pose['roll']:+.1f}"
        status = _head_pose.get_status_text(pose) if _head_pose else "n/a"
        if status == "ALERT":
            status = "OK"
        status_color = (0, 0, 255) if pose["alert"] else (0, 200, 0)
    else:
        angles, status, status_color = "Pitch: --  Yaw: --  Roll: --", "N/A", GREY
    cv2.putText(frame, angles, (10, 125), font, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Head: {status}", (w - 200, 125), font, 0.55,
                status_color, 2, cv2.LINE_AA)

    # Microsleep banner across the middle of the frame while the override is
    # up. Deliberately loud and deliberately not in the metrics strip: this is
    # the one event the whole system exists to catch.
    if m.microsleep_override:
        closed_s = float(m.microsleep_status.get("closed_s", 0.0))
        held = f" {closed_s:.1f}s" if closed_s > 0 else " (recovering)"
        banner_y = frame.shape[0] // 2
        cv2.rectangle(frame, (0, banner_y - 34), (w, banner_y + 12), (0, 0, 0), -1)
        cv2.putText(frame, f"MICROSLEEP{held}  (n={m.microsleep_count})",
                    (12, banner_y), font, 0.95, (0, 0, 255), 2, cv2.LINE_AA)


def draw_pose_debug(
    frame: np.ndarray,
    debounce: HeadPoseDebounce,
    now: float,
    fps: Optional[float] = None,
) -> None:
    """
    ``--debug-pose`` strip under the main overlay: how long the pose has been
    alerting vs. the DANGER threshold, whether the override is active, and
    (while active) how long the pose has been normal vs. the release hold.

    Args:
        frame: BGR frame to draw on.
        debounce: The pipeline's :class:`HeadPoseDebounce`.
        now: ``time.monotonic()`` for this frame.
        fps: Last measured loop rate (shown so the rate is visible on the
            NO FACE / UNKNOWN DRIVER frames, which have no main overlay).
    """
    if not _display_available:
        return  # no preview: nobody sees the annotation
    font = cv2.FONT_HERSHEY_SIMPLEX
    y0 = 135
    cv2.rectangle(frame, (0, y0), (frame.shape[1], y0 + 48), (30, 30, 30), -1)

    alert_s = debounce.alert_elapsed(now)
    normal_s = debounce.normal_elapsed(now)
    if debounce.active:
        state, color = "OVERRIDE: DANGER", (0, 0, 255)
    elif alert_s > 0:
        state, color = "OVERRIDE: arming", (0, 220, 255)
    else:
        state, color = "OVERRIDE: off", (0, 200, 0)

    # Progress bar for whichever timer is currently counting.
    if debounce.active:
        frac = min(normal_s / debounce.release_hold, 1.0) if debounce.release_hold > 0 else 1.0
        label = f"normal {normal_s:.2f}s / release at {debounce.release_hold:.2f}s"
        bar_color = (0, 200, 0)
    else:
        frac = min(alert_s / debounce.danger_hold, 1.0)
        label = f"alerting {alert_s:.2f}s / DANGER at {debounce.danger_hold:.2f}s"
        bar_color = (0, 0, 255)

    cv2.putText(frame, f"POSE DEBUG  {state}", (10, y0 + 18), font, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}" if fps is not None else "FPS: --",
                (frame.shape[1] - 150, y0 + 18), font, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (10, y0 + 40), font, 0.45, WHITE, 1, cv2.LINE_AA)
    bx0, bx1, by = frame.shape[1] - 150, frame.shape[1] - 10, y0 + 30
    cv2.rectangle(frame, (bx0, by), (bx1, by + 12), (90, 90, 90), 1)
    if frac > 0:
        cv2.rectangle(frame, (bx0, by), (bx0 + int((bx1 - bx0) * frac), by + 12), bar_color, -1)


def display_text(frame: np.ndarray, text: str, color: tuple = (0, 0, 255), dy: int = 0) -> None:
    """
    Draw ``text`` centred on ``frame`` in place (e.g. ``"UNKNOWN DRIVER"``).

    Args:
        frame: BGR frame to draw on.
        text: Message to show.
        color: BGR colour.
        dy: Vertical offset from centre (for a second line).
    """
    if not _display_available:
        return  # no preview: nobody sees the annotation
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 1.0, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max((frame.shape[1] - tw) // 2, 0)
    y = (frame.shape[0] + th) // 2 + dy
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def cleanup() -> None:
    """Release the camera, reset GPIO and close any OpenCV windows."""
    global _camera, _profiler
    logger.info("Shutting down...")
    if _profiler is not None:
        # Quit ('q' / Ctrl-C / launcher STOP) mid-window: report what was timed.
        _profiler.stop()
        _profiler = None
    if _camera is not None:
        _camera.release()
        _camera = None
    if _alert_manager is not None:
        _alert_manager.cleanup()
    if _ignition is not None:
        _ignition.cleanup()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # headless
    logger.info("Cleanup complete")


# ---------------------------------------------------------------------------
# Enrollment mode
# ---------------------------------------------------------------------------

def run_enrollment(
    api_client: APIClient, extractor: LandmarkExtractor, driver_id: Optional[int] = None
) -> int:
    """
    Enrol a new driver: face encoding + 60 s alert-state calibration.

    Steps:
    1. Ask for the driver's DB id on the terminal (skipped when ``driver_id``
       is supplied, e.g. via ``--driver-id`` from the touchscreen launcher).
    2. Capture ``ENROLL_FRAMES`` frames containing a face and average their
       128-d encodings (averaging is more robust than a single frame).
    3. Settle for ``SEED_WINDOW_S`` to derive a personal closure threshold
       from the driver's own median EAR.
    4. Closed-eye capture (:func:`run_closed_eye_capture`): the driver shuts
       their eyes on cue, up to ``CLOSED_CAPTURE_ATTEMPTS`` tries, and the
       drop must show the landmarks follow the eyelid. Done *before* the
       alert window so the closure cannot contaminate it.
    5. Run ``CalibrationManager`` for ``config.CALIBRATION_DURATION`` s,
       feeding it EAR / blink / PERCLOS / MAR from the live pipeline, then
       re-check the contrast against the final ``ear_baseline``.
    6. Write the per-frame series to ``config.CALIBRATIONS_DIR``, POST
       encoding + baselines to the backend, and read the record back to
       confirm ``ear_closed_baseline`` was stored.

    Args:
        api_client: Connected API client.
        extractor: Landmark extractor for the calibration phase.
        driver_id: Driver's DB id. ``None`` prompts on the terminal.

    Returns:
        Process exit code: 0 success; 2 bad driver id; 3 too few face
        frames; 5 backend rejected the enrollment;
        ``EXIT_NO_EYELID_CONTRAST`` closed-eye check failed (nothing
        POSTed); ``EXIT_CLOSED_EAR_DROPPED`` saved, but the backend did not
        keep ``ear_closed_baseline``.
    """
    import face_recognition  # heavy import; only needed here

    if driver_id is None:
        raw = input("Enter driver_id to enrol: ").strip()
        try:
            driver_id = int(raw)
        except ValueError:
            logger.error("Invalid driver_id %r - must be an integer", raw)
            return 2
    logger.info("Enrolling driver %s", driver_id)

    # ---- 1. Face encoding ------------------------------------------------
    logger.info("Look at the camera. Capturing %d frames for the face encoding...", ENROLL_FRAMES)
    encodings = []
    attempts = 0
    while len(encodings) < ENROLL_FRAMES and attempts < ENROLL_FRAMES * 10:
        attempts += 1
        frame = capture_frame()
        if frame is None:
            continue
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        locations = face_recognition.face_locations(rgb, model="hog")
        if not locations:
            display_text(frame, "NO FACE DETECTED")
            present(frame)
            continue
        # Use only the largest face in case someone is visible in the background.
        largest = max(locations, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
        enc = face_recognition.face_encodings(rgb, [largest])
        if enc:
            encodings.append(enc[0])
        display_text(frame, f"ENCODING {len(encodings)}/{ENROLL_FRAMES}")
        present(frame)

    if len(encodings) < ENROLL_FRAMES // 2:
        logger.error("Only %d usable face frames captured - aborting enrollment", len(encodings))
        return 3
    face_encoding = np.mean(np.stack(encodings), axis=0)
    logger.info("Face encoding computed from %d frames", len(encodings))

    # ---- 2. Calibration ---------------------------------------------------
    ear_calc = EARCalculator()
    mar_calc = MARCalculator()
    blink_detector = BlinkDetector(frequency_window=60)
    perclos_calc = PERCLOSCalculator()
    microsleep_detector = MicrosleepDetector()
    calib = CalibrationManager()

    # No personal threshold exists yet. Rather than seeding one from
    # DEFAULT_THRESHOLDS (0.225 is 75 % of a 0.30 EAR but 88 % of a 0.255 one -
    # inside landmark noise, so it manufactures false closures), observe the
    # driver for SEED_WINDOW_S of wall-clock time *before* calibration starts
    # and derive the threshold from their own median EAR. Nothing is fed to
    # the blink detector, PERCLOS or the calibration until then, so the
    # calibration window is pure detection time under a personal threshold.
    ear_history: list = []
    seed_start = time.time()
    logger.info("Settling for %.0f s to derive a personal closure threshold...", SEED_WINDOW_S)
    while time.time() - seed_start < SEED_WINDOW_S or len(ear_history) < SEED_MIN_SAMPLES:
        frame = capture_frame()
        if frame is None:
            continue
        landmarks, _ = extractor.extract(frame)
        if landmarks is None:
            display_text(frame, "FACE LOST - please look at the camera")
            present(frame)
            continue
        ear_history.append(ear_calc.compute_average_ear(landmarks))
        display_text(frame, "SETTLING - look at the road")
        present(frame)
    open_reference = float(np.median(ear_history))
    threshold = open_reference * EAR_THRESHOLD_RATIO
    logger.info("Seed threshold %.4f from %d frames (median EAR %.4f)",
                threshold, len(ear_history), open_reference)

    # Closed-eye capture, before the alert window (see run_closed_eye_capture).
    closed_capture, closed_attempts = run_closed_eye_capture(extractor, ear_calc, open_reference)
    if closed_capture is None:
        logger.error("ENROLLMENT ABORTED for driver %s: the closed-eye check failed %d times - "
                     "nothing was saved. If the driver did close their eyes, the landmarks are "
                     "not tracking their eyelids: check lighting and camera position, and run "
                     "tools/eye_check.py to see where the eye landmarks sit.",
                     driver_id, len(closed_attempts))
        return EXIT_NO_EYELID_CONTRAST
    settle_start = time.time()
    while time.time() - settle_start < REOPEN_SETTLE_S:
        frame = capture_frame()
        if frame is None:
            continue
        display_text(frame, "OPEN YOUR EYES - look at the road")
        present(frame)

    logger.info("Starting %d s calibration - stay alert, look at the road and keep "
                "your mouth relaxed (talking inflates the MAR baseline).",
                config.CALIBRATION_DURATION)
    calib.start()
    while calib.is_calibrating:
        frame = capture_frame()
        if frame is None:
            continue
        landmarks, _ = extractor.extract(frame)
        if landmarks is None:
            display_text(frame, "FACE LOST - please look at the camera")
            present(frame)
            continue

        ear = ear_calc.compute_average_ear(landmarks)
        mar = mar_calc.compute_mar(landmarks)
        # Keep tightening to the running median as more of the driver is seen.
        ear_history.append(ear)
        threshold = float(np.median(ear_history)) * EAR_THRESHOLD_RATIO

        now = time.time()
        eye_closed = ear < threshold  # the test both detectors below apply
        event = blink_detector.update(ear, threshold, now)
        # Same EAR and threshold: closures >= 1 s are microsleeps and are
        # rejected by the blink detector, so they must be collected here or
        # they would vanish from the calibration record entirely.
        ms_event = microsleep_detector.update(ear, threshold, now)
        # No warm-up prior: the baseline is what is being measured.
        perclos = perclos_calc.update(ear, threshold, now)
        status = calib.update(
            ear,
            event["duration_ms"] if event else None,
            blink_detector.get_blink_frequency(now),
            perclos,
            timestamp=now,
            mar=mar,
            eye_closed=eye_closed,
            threshold=threshold,
            microsleep_ms=ms_event["duration_ms"] if ms_event else None,
        )

        display_text(frame, f"CALIBRATING {status['progress'] * 100:.0f}%  "
                            f"({status['seconds_remaining']}s left)")
        present(frame)

    baselines = calib.compute_baselines()
    baselines["ear_closed_baseline"] = float(closed_capture.closed_ear)
    logger.info("Calibration baselines: %s", baselines)
    # Pre-floor measurements and counts, so a baseline that landed on a
    # MIN_* floor (modules/calibration.py) can be traced to its cause.
    logger.info("Calibration raw: %s", calib.raw_summary())
    # The pre-check used the 1 s settle median; the stored contrast is
    # against the 60 s ear_baseline, and that is what load_calibration judges.
    problem = ear_calibration_problem(baselines["ear_baseline"],
                                      baselines["ear_closed_baseline"])
    # Per-frame series + summary on disk, written before the backend call so
    # a rejected enrollment still leaves the evidence behind.
    try:
        calib.write_files(config.DEVICE_ID, driver_id, baselines, closed_capture={
            "attempts": closed_attempts,
            "accepted": closed_capture.summary(baselines["ear_baseline"]),
            "problem": problem,
        })
    except OSError as exc:
        logger.error("Could not write calibration data: %s", exc)
    if problem is not None:
        logger.error("ENROLLMENT REFUSED for driver %s: %s. Nothing was sent to the backend; "
                     "re-run the enrollment.", driver_id, problem)
        return EXIT_NO_EYELID_CONTRAST
    logger.info("Closed-eye contrast %.2f (closed %.4f / ear_baseline %.4f)",
                baselines["ear_closed_baseline"] / baselines["ear_baseline"],
                baselines["ear_closed_baseline"], baselines["ear_baseline"])

    # ---- 3. Persist ---------------------------------------------------------
    ok = api_client.save_driver_enrollment(driver_id, face_encoding.tolist(), baselines)
    if not ok:
        logger.error("Backend rejected enrollment for driver %s", driver_id)
        return 5
    # A Laravel FormRequest drops fields it does not validate, silently: the
    # record would come back without ear_closed_baseline and be treated as a
    # pre-2026-09-30 record (no contrast check). Read it back to be sure.
    stored = api_client.get_driver_calibration(driver_id)
    if stored is None:
        logger.warning("Enrollment saved, but the calibration could not be read back to confirm "
                       "ear_closed_baseline was stored")
    elif "ear_closed_baseline" not in stored.thresholds:
        logger.error("Enrollment saved, but the backend DROPPED ear_closed_baseline: this "
                     "driver's record has no contrast check and will be treated as a legacy "
                     "record. Deploy the backend change (deploy/BACKEND_CHANGES_2026-09-28.md, "
                     "item 6) and re-enrol.")
        return EXIT_CLOSED_EAR_DROPPED

    msg = f"Enrollment complete for driver {driver_id}"
    logger.info(msg)
    print(msg)
    return 0


def run_closed_eye_capture(
    extractor: LandmarkExtractor, ear_calc: EARCalculator, open_reference: float
) -> Tuple[Optional[ClosedEyeCapture], List[Dict[str, Any]]]:
    """
    Enrollment step: measure the driver's EAR with the eyes deliberately shut.

    Each attempt shows a ``CLOSED_PROMPT_LEAD_S`` countdown, beeps once
    (close now), captures for ``CLOSED_CAPTURE_S`` and beeps long (open) -
    the driver cannot read the screen with their eyes shut, so the operator
    should also say "open". An attempt passes when closed / open is at most
    ``EAR_CONTRAST_MAX`` against ``open_reference`` (the settle median); a
    failure is logged with its reason and retried, up to
    ``CLOSED_CAPTURE_ATTEMPTS`` in all. A driver who did not close their eyes
    and landmarks that do not follow the eyelid look the same in the
    numbers, so the reason names both and the operator decides.

    Returns:
        ``(capture, attempts)``: the passing :class:`ClosedEyeCapture` (or
        ``None`` if every attempt failed) and one summary dict per attempt.
    """
    attempts: List[Dict[str, Any]] = []

    def beep(pattern: str) -> None:
        # trigger_buzzer blocks for the beep; keep the capture loop running.
        if _alert_manager is not None:
            threading.Thread(target=_alert_manager.trigger_buzzer, args=(pattern,),
                             daemon=True).start()

    for attempt in range(1, CLOSED_CAPTURE_ATTEMPTS + 1):
        capture = ClosedEyeCapture()
        logger.info("Closed-eye capture %d/%d: tell the driver to close their eyes on the beep "
                    "and keep them shut until the long beep (%.0f s)",
                    attempt, CLOSED_CAPTURE_ATTEMPTS, capture.duration_s)
        lead_start = time.time()
        while time.time() - lead_start < CLOSED_PROMPT_LEAD_S:
            frame = capture_frame()
            if frame is None:
                continue
            remaining = CLOSED_PROMPT_LEAD_S - (time.time() - lead_start)
            display_text(frame, f"CLOSE YOUR EYES in {remaining:.0f}s - keep them shut "
                                f"until the long beep")
            present(frame)

        capture.start(time.time())
        beep("short")
        while not capture.done(time.time()):
            frame = capture_frame()
            if frame is None:
                continue
            landmarks, _ = extractor.extract(frame)
            if landmarks is not None:
                capture.update(ear_calc.compute_average_ear(landmarks), time.time())
                display_text(frame, "EYES CLOSED - keep them shut")
            else:
                display_text(frame, "FACE LOST - keep facing the camera, eyes shut")
            present(frame)
        beep("long")

        ok, ratio, reason = capture.evaluate(open_reference)
        record = dict(capture.summary(open_reference), attempt=attempt, ok=ok,
                      reason=reason or None)
        attempts.append(record)
        if ok:
            logger.info("Closed-eye capture %d passed: closed EAR %.4f / open %.4f = %.2f "
                        "(%d frames measured)", attempt, capture.closed_ear, open_reference,
                        ratio, len(capture.measured))
            return capture, attempts
        logger.warning("Closed-eye capture %d/%d FAILED: %s (closed EAR %s, open %.4f)",
                       attempt, CLOSED_CAPTURE_ATTEMPTS, reason,
                       "n/a" if capture.closed_ear is None else f"{capture.closed_ear:.4f}",
                       open_reference)
    return None, attempts


# ---------------------------------------------------------------------------
# Shared helpers for the two operating phases
# ---------------------------------------------------------------------------

def check_phase(ignition: Any, expected: Phase) -> None:
    """Raise :class:`PhaseChanged` if the ignition no longer implies ``expected``."""
    if ignition.phase() is not expected:
        raise PhaseChanged()


def load_calibration(
    api_client: APIClient, driver_id: int, phase: Phase
) -> Tuple[Optional[Calibration], Optional[Dict[str, Any]]]:
    """
    Fetch the recognised driver's calibration, and decide whether it is usable.

    Keyed by driver and device (``APIClient.get_driver_calibration``): the
    driver's calibration captured on this unit, else - flagged
    ``foreign_device`` - one captured on another unit. Never another
    driver's. (Pre-drive locks on ``foreign_device`` itself; see
    :func:`run_predrive_assessment`.)

    **Pre-drive and monitoring treat a broken record differently, on
    purpose.** Pre-drive can release the starter, so it is fail-secure: any
    defect - an EAR key or core baseline missing, a threshold inconsistent
    with its baseline, a zero PERCLOS baseline - returns ``None`` and the unit
    locks with ``NO_BASELINE``. Monitoring cannot release anything and a
    driver's real alert-state baseline is worth keeping, so there a broken
    record is *repaired* - but only where its EAR baseline is independently
    credible (``ear_calibration_problem``):

    * threshold missing / inconsistent -> recomputed as
      ``ear_baseline * EAR_THRESHOLD_RATIO``;
    * EAR baseline missing -> the EAR pair is self-seeded, the record's other
      baselines are kept;
    * PERCLOS baseline <= 0 -> floored at calibration's MIN_PERCLOS_BASELINE;
    * another core baseline missing -> filled from DEFAULT_THRESHOLDS.

    An EAR baseline *present but not credible* is never repaired in either
    phase: it is the suspect value, and a threshold recomputed from it would
    launder the corruption. The record is discarded (monitoring then
    self-seeds everything). Credible means (2026-09-30): with an
    ``ear_closed_baseline``, closed / open at most ``EAR_CONTRAST_MAX`` and
    the baseline inside the degenerate-value backstop; without one (records
    enrolled before the closed-eye capture), grandfathered if the baseline
    is inside ``EAR_BASELINE_UNCONTRASTED_MIN`` - ``EAR_BASELINE_MAX``. Every repair is logged at WARNING with original
    and new values and listed in ``Calibration.repairs``, which travels as
    ``calibration_repairs`` on every fatigue event. Frequent repairs point at
    a problem in the enrollment path.

    Args:
        api_client: Backend client.
        driver_id: Recognised driver.
        phase: ``Phase.PREDRIVE`` (no repairs) or ``Phase.MONITORING``.

    Returns:
        ``(calibration, provenance)``. ``calibration`` is usable - every
        pipeline key present except, when ``self_seed_ear``, the EAR pair -
        or ``None``. ``provenance`` describes the record the backend
        returned *whether or not it was used* (``None`` only if there was
        none), so a lock on a rejected record still shows the operator which
        calibration was refused and where it was captured.
    """
    fetched = api_client.get_driver_calibration(driver_id)
    if fetched is None:
        return None, None
    thresholds = dict(fetched.thresholds)
    source = f"backend, driver {driver_id} ({fetched.source}, calibration {fetched.calibration_id})"
    monitoring = Phase(phase) is Phase.MONITORING

    baseline = thresholds.get("ear_baseline")
    closed = thresholds.get("ear_closed_baseline")
    if baseline is not None:
        problem = ear_calibration_problem(baseline, closed)
        if problem is not None:
            logger.error("CALIBRATION DISCARDED (%s): %s - the baseline itself is suspect, so "
                         "the record is not repaired%s", source, problem,
                         "; self-seeding instead" if monitoring else "; locking")
            return None, fetched.provenance()
        if closed is None:
            logger.info("Calibration (%s) predates the closed-eye capture: no contrast check "
                        "possible, grandfathered on its ear_baseline %.4f", source, baseline)
        else:
            logger.info("Calibration (%s): closed/open EAR %.4f/%.4f = %.2f", source, closed,
                        baseline, closed / baseline)

    # Drivers enrolled before yawn detection have no MAR baseline; say so
    # explicitly because the generic default silently makes yawn detection
    # non-personal. Allowed in both phases (a known enrollment vintage, not
    # a broken record).
    missing_mar = [k for k in MAR_THRESHOLD_KEYS if k not in thresholds]
    if missing_mar:
        logger.warning("Driver %s has no MAR baseline (%s missing) - yawn detection will "
                       "use generic defaults; re-enrol with `main.py --enroll` to calibrate it",
                       driver_id, ", ".join(missing_mar))
        for key in missing_mar:
            thresholds[key] = DEFAULT_THRESHOLDS[key]
    missing_core = [k for k in DEFAULT_THRESHOLDS if k not in thresholds]

    if not monitoring:
        # Pre-drive: fail-secure, no repairs.
        missing = [k for k in ("ear_baseline", "ear_threshold") if k not in thresholds]
        if missing + missing_core:
            logger.error("Calibration unusable for pre-drive (%s): missing %s",
                         source, ", ".join(missing + missing_core))
            return None, fetched.provenance()
        if not check_threshold_sanity(thresholds, source):
            return None, fetched.provenance()
        usable = Calibration(thresholds, fetched.driver_id, fetched.source,
                             fetched.calibration_id, fetched.captured_on_device_id)
        return usable, usable.provenance()

    # Monitoring-only repairs (see docstring).
    repairs = []
    for key in missing_core:
        thresholds[key] = DEFAULT_THRESHOLDS[key]
        repairs.append(repair_defaulted(key))
        logger.warning("CALIBRATION REPAIRED (%s): %s missing -> default %.3f",
                       source, key, thresholds[key])
    if baseline is None:
        thresholds.pop("ear_threshold", None)
        repairs.append(REPAIR_EAR_PAIR_SELF_SEEDED)
        logger.warning("CALIBRATION REPAIRED (%s): no ear_baseline -> EAR pair self-seeded, "
                       "other baselines kept", source)
    else:
        original = thresholds.get("ear_threshold")
        if original is None or abs(original / baseline - EAR_THRESHOLD_RATIO) > EAR_RATIO_TOLERANCE:
            thresholds["ear_threshold"] = baseline * EAR_THRESHOLD_RATIO
            repairs.append(REPAIR_EAR_THRESHOLD_RECOMPUTED)
            logger.warning("CALIBRATION REPAIRED (%s): ear_threshold %s (ratio %s) -> %.4f "
                           "(%.2f x ear_baseline %.4f)", source,
                           "missing" if original is None else f"{original:.4f}",
                           "-" if original is None else f"{original / baseline:.3f}",
                           thresholds["ear_threshold"], EAR_THRESHOLD_RATIO, baseline)
    if thresholds["perclos_baseline"] <= 0.0:
        original = thresholds["perclos_baseline"]
        thresholds["perclos_baseline"] = MIN_PERCLOS_BASELINE
        repairs.append(REPAIR_PERCLOS_BASELINE_FLOORED)
        logger.warning("CALIBRATION REPAIRED (%s): perclos_baseline %.3f -> %.3f (calibration "
                       "floor)", source, original, MIN_PERCLOS_BASELINE)
    if baseline is not None:
        check_threshold_sanity(thresholds, source)   # logs the (now consistent) pair
    usable = Calibration(thresholds, fetched.driver_id, fetched.source,
                         fetched.calibration_id, fetched.captured_on_device_id, tuple(repairs))
    return usable, usable.provenance()


def check_threshold_sanity(thresholds: Dict[str, float], source: str) -> bool:
    """
    Check that ``ear_threshold`` is consistent with ``ear_baseline``.

    Everything that decides "are the eyes shut" -
    PERCLOS, the blink detector and the microsleep detector - tests
    ``ear < thresholds["ear_threshold"]``, so a threshold that does not match
    the baseline it was derived from silently disables all three at once:
    PERCLOS stays at 0 %, no blink ever completes, and no closure is ever
    long enough to confirm a microsleep. The FRS then rides on the EAR term
    alone, which still responds to closure because it divides by the
    baseline rather than comparing against the threshold.

    Calibration sets ``ear_threshold = ear_baseline * EAR_THRESHOLD_RATIO``
    (0.75), so the ratio below should read ~0.75. Anything much lower means
    the two values came from different places - a stale threshold against a
    re-enrolled baseline, or a partial backend record.

    Until 2026-09-28 this only logged, and monitoring then ran on the record
    anyway; now a failing record is not used (pre-drive locks, monitoring
    self-seeds).

    Args:
        thresholds: The dict about to be handed to the pipeline.
        source: Where it came from, for the log line.

    Returns:
        ``True`` if the record is usable.
    """
    # finite_or_none: a key that is present but None / non-numeric reads as
    # 0 here (unusable) instead of raising.
    def num(key: str) -> float:
        return finite_or_none(thresholds.get(key)) or 0.0

    baseline = num("ear_baseline")
    threshold = num("ear_threshold")
    if baseline <= 0.0:
        logger.error("THRESHOLD CHECK (%s): ear_baseline is %.4f - EAR normalisation "
                     "and the closure test are both meaningless", source, baseline)
        return False
    ratio = threshold / baseline
    usable = True
    logger.info(
        "THRESHOLD CHECK (%s): ear_baseline=%.4f ear_threshold=%.4f ratio=%.3f "
        "(expected %.2f) | perclos_baseline=%.3f blink_duration_baseline=%.1fms "
        "blink_frequency_baseline=%.2f",
        source, baseline, threshold, ratio, EAR_THRESHOLD_RATIO,
        num("perclos_baseline"), num("blink_duration_baseline"),
        num("blink_frequency_baseline"),
    )
    if ratio < EAR_THRESHOLD_RATIO - EAR_RATIO_TOLERANCE:
        logger.error(
            "THRESHOLD TOO LOW (%s): ear_threshold is %.1f%% of ear_baseline, expected "
            "%.0f%%. The eye must drop to %.1f%% of its open value (below %.4f) before "
            "anything counts as closed. Real closures often only reach 40-50%% of "
            "baseline, so PERCLOS, blink detection and microsleep detection may ALL "
            "report nothing while the driver's eyes are visibly shut. Re-enrol this "
            "driver, or check the backend is not serving a stale threshold against a "
            "newer baseline.",
            source, ratio * 100, EAR_THRESHOLD_RATIO * 100, ratio * 100, threshold,
        )
        usable = False
    elif ratio > EAR_THRESHOLD_RATIO + EAR_RATIO_TOLERANCE:
        logger.error(
            "THRESHOLD TOO HIGH (%s): ear_threshold is %.1f%% of ear_baseline, expected "
            "%.0f%%. Partly-open eyes will register as closed, inflating PERCLOS and "
            "manufacturing blinks. Re-enrol this driver.",
            source, ratio * 100, EAR_THRESHOLD_RATIO * 100,
        )
        usable = False
    if num("perclos_baseline") <= 0.0:
        logger.error(
            "THRESHOLD CHECK (%s): perclos_baseline is 0 - PERCLOSCalculator.normalize() "
            "returns 0.0 for a zero baseline, so the PERCLOS term is dead regardless of "
            "how long the eyes are shut", source)
        usable = False
    if not usable:
        logger.error("THRESHOLD CHECK (%s): record NOT USED", source)
    return usable


def identify_driver_bounded(
    recognizer: DriverRecognizer,
    ignition: Any,
    rate: LoopRate,
    timeout_s: float = config.PREDRIVE_RECOGNITION_TIMEOUT_S,
    matches_required: int = config.PREDRIVE_RECOGNITION_MATCHES,
) -> Optional[Dict[str, Any]]:
    """
    Pre-drive recognition: identify on every frame for up to ``timeout_s``
    and accept once the same ``driver_id`` has matched ``matches_required``
    times (rejects a single false match; gives the camera time to settle).

    Returns:
        The accepted match, or ``None`` on timeout.

    Raises:
        PhaseChanged: if the ignition turns ON meanwhile.
    """
    deadline = time.monotonic() + timeout_s
    counts: Dict[int, int] = {}
    best: Optional[Dict[str, Any]] = None
    logger.info("Pre-drive: identifying driver (up to %.0fs, %d consistent matches required)",
                timeout_s, matches_required)
    while time.monotonic() < deadline:
        check_phase(ignition, Phase.PREDRIVE)
        frame = next_frame()
        rate.tick()
        match = recognizer.identify(frame)
        if match is not None:
            did = int(match["driver_id"])
            counts[did] = counts.get(did, 0) + 1
            best = match
            if counts[did] >= matches_required:
                logger.info("Driver identified: %s (id=%s, confidence %.2f, %d matches)",
                            match["name"], did, match["confidence"], counts[did])
                return match
        seen = counts.get(int(best["driver_id"]), 0) if best else 0
        display_text(frame, f"IDENTIFYING {seen}/{matches_required}", WHITE)
        display_text(frame, f"{max(0.0, deadline - time.monotonic()):.0f}s left", GREY, dy=40)
        draw_banner(frame, "PRE-DRIVE  |  starter LOCKED  |  identifying driver", (0, 220, 255))
        present(frame)
    logger.warning("Pre-drive: driver not recognised within %.0fs (matches seen: %s)",
                   timeout_s, counts or "none")
    return None


def wait_for_ignition(ignition: Any, rate: LoopRate, message: str, color: tuple) -> bool:
    """
    Show ``message`` until the ignition turns ON (return ``True``) or the
    user presses ``r`` to re-run the assessment (return ``False``).
    """
    while True:
        if ignition.phase() is Phase.MONITORING:
            return True
        frame = next_frame()
        rate.tick()
        display_text(frame, message, color)
        display_text(frame, "waiting for ignition ON  (r = re-assess)", GREY, dy=40)
        relay = _alert_manager.get_relay_state() if _alert_manager else "n/a"
        draw_banner(frame, f"PRE-DRIVE  |  starter {relay}", color)
        if present(frame) == KEY_RETRY:
            logger.info("Pre-drive: re-assessment requested from the preview window")
            return False


def show_release(rate: LoopRate, message: str) -> None:
    """
    Pre-drive -> monitoring hand-off: show ``message`` in green on live
    frames for :data:`RELEASE_INDICATOR_S`. Called after the relay has been
    released; does not read the ignition (monitoring follows either way).
    """
    logger.info("Pre-drive: %s - starter released, monitoring starts in %.1fs",
                message, RELEASE_INDICATOR_S)
    deadline = time.monotonic() + RELEASE_INDICATOR_S
    while time.monotonic() < deadline:
        frame = next_frame()
        rate.tick()
        display_text(frame, message, (0, 200, 0))
        display_text(frame, "starter released - monitoring starting", GREY, dy=40)
        relay = _alert_manager.get_relay_state() if _alert_manager else "n/a"
        draw_banner(frame, f"PRE-DRIVE  |  starter {relay}", (0, 200, 0))
        present(frame)


def await_override(
    api_client: APIClient,
    alert_manager: AlertManager,
    ignition: Any,
    rate: LoopRate,
    driver_id: Optional[int],
    reason: LockReason,
    assessment_id: Optional[int],
    provenance: Optional[Dict[str, Any]] = None,
    continue_to_monitoring: bool = True,
) -> bool:
    """
    Starter stays inhibited: raise one override request and poll the
    operator's decision until approved (-> release starter), the user
    presses ``r`` (-> re-assess) or the phase changes.

    Every :class:`LockReason` resolves through this one path. ``provenance``
    (``Calibration.provenance()``) goes on the request so the operator can
    see which calibration was involved - for ``FOREIGN_DEVICE_BASELINE``,
    which unit it was captured on.

    Returns:
        ``True`` if the override was approved and ``continue_to_monitoring``
        is set: the starter is released and the caller goes straight into
        monitoring. ``False`` otherwise (``r``, or - with
        ``continue_to_monitoring`` off, i.e. ``--force-phase predrive`` - an
        approval followed by ``r`` on the "start vehicle" screen). A phase
        change raises :class:`PhaseChanged` as before.
    """
    reason_text = LOCK_REASON_TEXT[reason]
    logger.warning("Pre-drive: starter stays LOCKED - %s (driver=%s)", reason.value, driver_id)
    alert_manager.lock_relay()  # already locked; explicit for clarity + log

    request_id = api_client.request_override(config.DEVICE_ID, driver_id, reason, assessment_id,
                                             provenance)
    status = "pending"
    last_poll = time.monotonic()
    offline_logged = request_id is None
    if offline_logged:
        logger.error("Override request could not be raised (backend offline?) - will retry "
                     "every %.0fs; starter stays locked", config.OVERRIDE_POLL_SECONDS)

    while True:
        check_phase(ignition, Phase.PREDRIVE)
        now = time.monotonic()
        if now - last_poll >= config.OVERRIDE_POLL_SECONDS:
            last_poll = now
            if request_id is None:
                request_id = api_client.request_override(
                    config.DEVICE_ID, driver_id, reason, assessment_id, provenance)
            elif status == "pending":
                polled = api_client.check_override_request(request_id)
                if polled is not None and polled != status:
                    status = polled
                    logger.warning("Override request %s -> %s", request_id, status)

        if status == "approved":
            logger.warning("Operator override APPROVED (request %s) - releasing starter", request_id)
            alert_manager.unlock_relay()
            if continue_to_monitoring:
                show_release(rate, "OVERRIDE APPROVED")
                return True
            wait_for_ignition(ignition, rate, "OVERRIDE APPROVED - start vehicle", (0, 200, 0))
            return False

        frame = next_frame()
        rate.tick()
        display_text(frame, f"STARTER LOCKED: {reason_text}")
        if request_id is None:
            second, color = "backend unreachable - retrying override request", (0, 220, 255)
        elif status == "denied":
            second, color = f"override DENIED (request {request_id})  -  r = re-assess", (0, 0, 255)
        else:
            second, color = f"awaiting operator override (request {request_id})", (0, 220, 255)
        display_text(frame, second, color, dy=40)
        draw_banner(frame, "PRE-DRIVE  |  starter LOCKED", (0, 0, 255))
        if present(frame) == KEY_RETRY:
            logger.info("Pre-drive: re-assessment requested from the preview window")
            return False


# ---------------------------------------------------------------------------
# Phase 1: pre-drive assessment (ignition OFF)
# ---------------------------------------------------------------------------

def run_predrive_assessment(
    api_client: APIClient,
    extractor: LandmarkExtractor,
    recognizer: DriverRecognizer,
    alert_manager: AlertManager,
    head_pose: HeadPoseEstimator,
    ignition: Any,
    rate: LoopRate,
    debug_pose: bool = False,
    pose_release_hold: float = HEAD_POSE_RELEASE_HOLD_S,
    diag_ear: bool = False,
    continue_to_monitoring: bool = True,
) -> bool:
    """
    One pre-drive cycle.

    1. Relay inhibited (``set_phase(PREDRIVE)``).
    2. Recognise the driver (bounded)          -> else DRIVER_NOT_RECOGNIZED.
    3. This driver's calibration from this unit  -> none at all: NO_BASELINE;
       only from another unit: FOREIGN_DEVICE_BASELINE (never assessed on it).
    4. 30 s assessment through the shared pipeline; every frame recorded.
    5. Persist CSV/JSON + POST /assessments.
    6. PASS -> release starter, then hand off to monitoring.
       FAIL -> FATIGUE_DETECTED.
    7. Any lock reason -> :func:`await_override`; the starter stays
       inhibited until an operator approves (then as for a pass).

    With ``continue_to_monitoring`` off (``--force-phase predrive``, the
    phase tested alone) a release instead waits on the "start vehicle"
    screen, where ``r`` re-runs the assessment - the pre-2026-09-29 flow.

    Returns:
        ``True`` if the starter was released (pass, or approved override)
        and the caller should go straight into monitoring. ``False`` if the
        user asked to re-run (``r``), or the ignition turned ON before a
        release (the caller re-reads the phase; the starter stays inhibited).
    """
    alert_manager.set_phase(Phase.PREDRIVE)
    # Every pre-drive cycle starts inhibited, including a re-run requested
    # with 'r' after a pass (set_phase is a no-op when already in PREDRIVE).
    alert_manager.lock_relay()
    alert_manager.set_alert_level("ALERT")
    logger.info("=== PRE-DRIVE phase (ignition OFF) - starter %s ===",
                alert_manager.get_relay_state())

    driver: Optional[Dict[str, Any]] = None
    driver_id: Optional[int] = None
    calibration: Optional[Calibration] = None
    # Provenance of the record the backend returned, used or not.
    calibration_record: Optional[Dict[str, Any]] = None
    lock_reason: Optional[LockReason] = None
    assessment_id: Optional[int] = None

    try:
        # 2. Recognise
        driver = identify_driver_bounded(recognizer, ignition, rate)
        if driver is None:
            lock_reason = LockReason.DRIVER_NOT_RECOGNIZED
        else:
            driver_id = int(driver["driver_id"])
            # 3. Calibration - this driver's own, captured on this unit, or
            # lock. One captured on another unit is never allowed to release
            # the starter (fail-secure: a baseline that reads low passes a
            # fatigued driver), so it locks without an assessment and the
            # override request carries its provenance for the operator.
            calibration, calibration_record = load_calibration(api_client, driver_id,
                                                               Phase.PREDRIVE)
            if calibration is None:
                # No record, or a broken one. A broken record from another
                # unit also lands here, deliberately: no_baseline is the
                # accurate category and the operator's action is the same,
                # whereas foreign_device_baseline would imply that approving
                # leaves a usable baseline. Its provenance still goes on the
                # override request as context.
                lock_reason = LockReason.NO_BASELINE
            elif calibration.source != CALIBRATION_SOURCE_DEVICE:
                logger.warning("Pre-drive: driver %s's only calibration (%s) was captured on "
                               "%s, not %s - locking for operator override",
                               driver_id, calibration.calibration_id,
                               calibration.captured_on_device_id or "an unrecorded device",
                               config.DEVICE_ID)
                lock_reason = LockReason.FOREIGN_DEVICE_BASELINE
            else:
                thresholds = calibration.thresholds

        if lock_reason is None:
            # 4. Assessment
            pipeline = MetricsPipeline(
                head_pose,
                blink_window_s=config.PREDRIVE_ASSESSMENT_SECONDS,
                pose_release_hold=pose_release_hold,
                diag_ear=diag_ear,
            )
            assessment = PredriveAssessment()
            assessment.calibration = calibration.provenance()
            assessment.start(time.time())
            logger.info("Pre-drive: %d s assessment started for driver %s (%s), "
                        "calibration %s (%s)", config.PREDRIVE_ASSESSMENT_SECONDS,
                        driver_id, driver["name"], calibration.calibration_id,
                        calibration.source)

            while not assessment.is_complete(time.time()):
                check_phase(ignition, Phase.PREDRIVE)
                frame = next_frame()
                fps = rate.tick()
                now, now_mono = time.time(), time.monotonic()
                remaining = assessment.seconds_remaining(now)
                banner = (f"PRE-DRIVE  |  assessing {remaining:4.1f}s left  |  "
                          f"starter {alert_manager.get_relay_state()}")

                landmarks, _rect = extractor.extract(frame)
                if landmarks is None:
                    # Policy disabled here: verdict.level is ALERT. A face
                    # missing for too long voids the assessment instead.
                    verdict = pipeline.note_no_face(now, now_mono)
                    assessment.add_no_face(now)
                    alert_manager.set_alert_level(verdict.level)
                    display_text(frame, "NO FACE")
                    draw_banner(frame, banner, (0, 220, 255))
                    if debug_pose:
                        draw_pose_debug(frame, pipeline.pose_debounce, now_mono, fps)
                    present(frame)
                    continue

                m = pipeline.process(landmarks, thresholds, now, now_mono)
                assessment.add(m, now)
                # LEDs / buzzer give the driver feedback; the relay is decided
                # once, below, on the aggregate - never per frame.
                alert_manager.set_alert_level(m.level)
                draw_overlay(frame, driver, m, fps)
                draw_banner(frame, banner, (0, 220, 255))
                if debug_pose:
                    draw_pose_debug(frame, pipeline.pose_debounce, now_mono, fps)
                present(frame)

            # 5. Persist
            result = assessment.result()
            logger.info("Pre-drive result: passed=%s void=%s worst_%.0fs_mean=%.3f mean=%.3f "
                        "median=%.3f max=%.3f n=%d no_face=%d pose_override_frames=%d "
                        "yawns=%d microsleeps=%d",
                        result["passed"], result["void"], result["worst_window_s"],
                        result["worst_window_mean"], result["mean_frs"], result["median_frs"],
                        result["max_frs"], result["n_samples"], result["n_no_face"],
                        result["pose_override_frames"], result["yawns"],
                        result["microsleeps"])
            try:
                assessment.write_files(config.DEVICE_ID, driver_id)
            except OSError as exc:
                logger.error("Could not write assessment files: %s", exc)

            if result["void"]:
                # Not enough face to judge - treat like an unrecognised driver.
                logger.warning("Pre-drive: assessment VOID (face missing %.0f%% of frames)",
                               result["no_face_fraction"] * 100)
                lock_reason = LockReason.DRIVER_NOT_RECOGNIZED
            elif result["failed_on_microsleep"]:
                # Checked before the aggregate so the lock reason names the
                # real finding: a microsleep fails outright, and the worst
                # window can easily still be under the threshold when it does.
                logger.warning(
                    "Pre-drive: FAILED ON MICROSLEEP - %d confirmed during the %.0fs "
                    "assessment (worst %.0fs mean %.3f vs threshold %.2f; the aggregate "
                    "alone would have %s). Sleep intrusion in a stationary vehicle is "
                    "disqualifying on its own.",
                    result["microsleeps"], result["duration_s"], result["worst_window_s"],
                    result["worst_window_mean"], result["pass_threshold"],
                    "FAILED" if result["worst_window_mean"] >= result["pass_threshold"]
                    else "PASSED",
                )
                lock_reason = LockReason.MICROSLEEP_DETECTED
            elif not result["passed"]:
                logger.warning(
                    "Pre-drive: FAILED on aggregate - worst %.0fs mean %.3f >= %.2f",
                    result["worst_window_s"], result["worst_window_mean"],
                    result["pass_threshold"],
                )
                lock_reason = LockReason.FATIGUE_DETECTED

            assessment_id = api_client.post_assessment(
                config.DEVICE_ID, driver_id, result, lock_reason)

            # 6. Verdict
            if lock_reason is None:
                logger.info("Pre-drive PASSED (worst %.0fs mean %.3f < %.2f) - releasing starter",
                            result["worst_window_s"], result["worst_window_mean"],
                            result["pass_threshold"])
                alert_manager.unlock_relay()
                alert_manager.set_alert_level("ALERT")
                if continue_to_monitoring:
                    show_release(rate, "ASSESSMENT PASSED")
                    return True
                wait_for_ignition(ignition, rate, "ASSESSMENT PASSED - start vehicle", (0, 200, 0))
                return False

        # 7. Lock path (every lock reason)
        alert_manager.set_alert_level("ALERT")
        return await_override(api_client, alert_manager, ignition, rate, driver_id,
                              lock_reason, assessment_id, calibration_record,
                              continue_to_monitoring=continue_to_monitoring)

    except PhaseChanged:
        logger.info("Pre-drive: ignition turned ON - leaving pre-drive (starter %s)",
                    alert_manager.get_relay_state())
        return False


# ---------------------------------------------------------------------------
# Phase 2: continuous monitoring (ignition ON)
# ---------------------------------------------------------------------------

def run_monitoring(
    api_client: APIClient,
    extractor: LandmarkExtractor,
    recognizer: DriverRecognizer,
    alert_manager: AlertManager,
    head_pose: HeadPoseEstimator,
    ignition: Any,
    rate: LoopRate,
    debug_pose: bool = False,
    pose_release_hold: float = HEAD_POSE_RELEASE_HOLD_S,
    diag_ear: bool = False,
    from_release: bool = False,
    profile_s: Optional[float] = None,
) -> None:
    """
    Monitor continuously while the ignition is ON. Returns when it turns OFF.

    ``from_release``: entered straight from a pre-drive release (pass or
    approved override), when the key may not have been turned yet. The
    ignition reading OFF then means "not started yet", not "trip over", so
    this returns only once the ignition has been seen ON and has gone OFF
    again. Under ``--sequence`` (no ignition input) that never happens and
    monitoring runs until stopped.

    ``profile_s``: ``--profile-loop`` - time every frame by category and log
    a :class:`LoopProfiler` summary every ``profile_s`` seconds.

    Alerts (LEDs, buzzer) and backend DANGER notifications only. The relay is
    never driven here; ``AlertManager`` refuses ``lock_relay()`` in this
    phase, so nothing in this loop can inhibit the starter even by mistake.
    A driver without a usable calibration (unrecognised, no record, record
    failing the threshold check, backend offline) is monitored on an EAR
    baseline self-seeded from their own EAR over the first
    ``SELF_SEED_S`` of face time; until it freezes, only the head-pose and
    microsleep overrides run. The remaining baselines are
    :data:`DEFAULT_THRESHOLDS`.

    Face recognition runs on face frames only until the driver is first
    recognised, then never again this session (2026-09-29). At ~1 s per call
    on the Pi (``face_locations`` + ``face_encodings``) the old 1.5 s
    re-identify cadence held the loop at ~11 fps; it cannot move to a
    thread because dlib holds the GIL for the whole call. Until a match it
    is retried every :data:`UNRECOGNISED_RETRY_S` (was: every face frame,
    ~1 fps). What an unrecognised driver gets is unchanged: no driver, no
    calibration, self-seeded EAR baseline, ``driver_id`` null on events and
    faults.

    Frames with no face follow ``modules.pipeline.NoFacePolicy``: the level
    is held (never lowered), a DANGER is latched, and ALERT / WARNING turn
    into FAULT after ``NO_FACE_FAULT_S``. Two cases become backend faults
    on ``POST /monitoring-faults``: FAULT (``no_face``) and a latched DANGER
    past ``NO_FACE_HOLD_S`` (``danger_latched``, critical). Each is opened
    with the last scored frame's metrics, re-POSTed every
    ``MONITORING_FAULT_REFRESH_S`` while open, and resolved when the face
    returns or the ignition turns off.

    A self-seed that gives up (``SELF_SEED_MAX_ATTEMPTS`` discarded seeds)
    puts the level at FAULT and opens a third fault, ``no_ear_baseline``,
    independent of any no-face gap; it resolves when a recognised driver
    restarts the pipeline or the ignition turns off.
    """
    global _profiler
    alert_manager.set_phase(Phase.MONITORING)
    alert_manager.set_alert_level("ALERT")
    logger.info("=== MONITORING phase (%s) - starter %s, lock refused ===",
                "entered from pre-drive release" if from_release else "ignition ON",
                alert_manager.get_relay_state())

    prof: Any = _NO_PROFILER
    if profile_s:
        prof = _profiler = LoopProfiler(profile_s)
        logger.info("Loop profiling ON: summary every %.0fs, per-frame CSV %s",
                    profile_s, prof._csv_path)

    pipeline = MetricsPipeline(head_pose, pose_release_hold=pose_release_hold,
                               no_face_policy=True, diag_ear=diag_ear)
    logger.info("Monitoring: FRS weights %s, theoretical max %s",
                pipeline.frs_calc.weights, pipeline.frs_calc.theoretical_max())

    driver: Optional[Dict[str, Any]] = None
    current_driver_id: Optional[int] = None
    # None = no usable calibration: the pipeline self-seeds the EAR pair and
    # takes the other baselines from DEFAULT_THRESHOLDS.
    calibration: Optional[Calibration] = None
    thresholds: Dict[str, float] = dict(DEFAULT_THRESHOLDS)
    defaults_logged = False
    # Unrecognised-driver retry schedule (UNRECOGNISED_RETRY_S).
    next_identify = float("-inf")
    identify_attempts = 0
    last_danger_push = 0.0
    # Effective level (FRS band, or DANGER under the head-pose override) as
    # of the previous scored frame, so transitions can be logged once rather
    # than every frame. ``None`` until the first face is processed.
    last_level: Optional[str] = None
    # Last scored frame, for the last_known block of a fault report.
    last_m: Optional[FrameMetrics] = None
    # Open monitoring fault (no_face or danger_latched), if any.
    open_fault: Optional[OpenFault] = None
    # Open no_ear_baseline fault (the self-seed gave up), if any.
    seed_fault: Optional[OpenFault] = None
    # See from_release: whether this session has seen the ignition ON yet.
    ignition_seen_on = False

    while True:
        prof.frame()
        with prof.section("gpio"):
            ignition_on = ignition.phase() is Phase.MONITORING
        if ignition_on:
            ignition_seen_on = True
        elif ignition_seen_on or not from_release:
            break
        with prof.section("capture"):
            frame = next_frame()
        fps = rate.tick()
        now, now_mono = time.time(), time.monotonic()
        if alert_manager.relay_locked:
            # Key turned ON before an assessment passed: the fuse tap is live
            # but the starter is still inhibited. Nothing here can release it.
            banner, banner_color = ("MONITORING  |  starter LOCKED - turn ignition OFF "
                                    "for pre-drive assessment"), (0, 0, 255)
        else:
            banner, banner_color = "MONITORING  |  relay never engages", (0, 200, 0)

        # Landmarks. No face never lowers the level and never raises it to
        # DANGER: the pipeline's NoFacePolicy holds, latches or faults.
        with prof.section("landmarks"):
            landmarks, _rect = extractor.extract(frame)
        if landmarks is None:
            with prof.section("pipeline"):
                verdict = pipeline.note_no_face(now, now_mono)
            if verdict.level != last_level:
                logger.info("Level %s -> %s (no face %.1fs, %s)", last_level or "(none)",
                            verdict.level, verdict.gap_s, verdict.band)
            last_level = verdict.level
            with prof.section("gpio"):
                alert_manager.set_alert_level(verdict.level)
            if verdict.event in FAULT_TYPE_FOR_EVENT:
                open_fault = OpenFault(
                    fault_uuid=str(uuid.uuid4()),
                    fault_type=FAULT_TYPE_FOR_EVENT[verdict.event],
                    driver_id=current_driver_id,
                    entry_level=verdict.entry_level,
                    started_at=datetime.now(timezone.utc) - timedelta(seconds=verdict.gap_s),
                    last_known=last_m.last_known() if last_m is not None else None,
                    gap_s=verdict.gap_s,
                    last_push_mono=now_mono,
                )
                with prof.section("api"):
                    open_fault.push(api_client, "open", now_mono)
            elif open_fault is not None:
                open_fault.gap_s = verdict.gap_s
                if now_mono - open_fault.last_push_mono >= MONITORING_FAULT_REFRESH_S:
                    with prof.section("api"):
                        open_fault.push(api_client, "open", now_mono)
            if seed_fault is not None:
                with prof.section("api"):
                    seed_fault.refresh_elapsed(api_client, now_mono)
            with prof.section("display"):
                display_text(frame, f"NO FACE {verdict.gap_s:.1f}s  [{verdict.band}]",
                             LEVEL_BGR.get(verdict.level, WHITE))
                draw_banner(frame, banner, banner_color)
                if debug_pose:
                    draw_pose_debug(frame, pipeline.pose_debounce, now_mono, fps)
            present(frame)   # times its own heartbeat (api) and imshow (display)
            continue

        # First face after a fault resolves it. Done here rather than from
        # m.gap_ended: a driver change below resets the pipeline first.
        fault_resolved = open_fault is not None
        if open_fault is not None:
            resolved = datetime.now(timezone.utc)
            logger.info("Face re-acquired after %.1fs - %s fault resolved",
                        (resolved - open_fault.started_at).total_seconds(),
                        open_fault.fault_type)
            with prof.section("api"):
                open_fault.push(api_client, "resolved", now_mono, resolved, "face_reacquired")
            open_fault = None

        # Identify - every face frame until the driver is recognised, then
        # never again this session (see docstring: ~1 s per call, GIL-bound).
        if current_driver_id is None and now_mono >= next_identify:
            identify_attempts += 1
            with prof.section("recognition"):
                match = recognizer.identify(frame)
            # Spaced from the END of the ~1 s attempt, so the loop always
            # gets UNRECOGNISED_RETRY_S of real frames between stalls.
            next_identify = time.monotonic() + UNRECOGNISED_RETRY_S
            if match is None:
                if not defaults_logged:
                    logger.warning("Monitoring: driver not recognised - self-seeding the EAR "
                                   "baseline from this driver's own EAR (overrides only for "
                                   "the first %.0fs of face time); retrying recognition "
                                   "every %.0fs", SELF_SEED_S, UNRECOGNISED_RETRY_S)
                    defaults_logged = True
                else:
                    logger.info("Monitoring: recognition attempt %d - no match, driver stays "
                                "UNRECOGNISED; next attempt in %.0fs",
                                identify_attempts, UNRECOGNISED_RETRY_S)
            else:
                driver = match
                current_driver_id = int(match["driver_id"])
                logger.info("Driver identified: %s (id=%s, confidence %.2f, attempt %d) - "
                            "recognition stops for this session", match["name"],
                            current_driver_id, match["confidence"], identify_attempts)
                # A foreign-device calibration IS used here (flagged on every
                # event), unlike pre-drive where it locks. Decided 2026-09-28:
                # a real alert-state baseline measured at the wrong camera
                # angle beats one self-seeded from a driver who may already
                # be drowsy - that failure is silent and defeats detection
                # for exactly the drivers who need it. The angle error is
                # unquantified (a stated limitation). Self-seed only when
                # there is no calibration, or (a flagged repair) when the
                # record has no EAR baseline.
                with prof.section("api"):   # synchronous GET, once, on recognition
                    calibration, _ = load_calibration(api_client, current_driver_id,
                                                      Phase.MONITORING)
                if calibration is None:
                    logger.warning("Monitoring: no usable calibration for driver %s - "
                                   "self-seeding the EAR baseline; re-enrol with "
                                   "`main.py --enroll`", current_driver_id)
                thresholds = (calibration.thresholds if calibration is not None
                              else dict(DEFAULT_THRESHOLDS))
                # Fresh history so metrics accumulated before recognition
                # (on a self-seeded baseline) don't leak into this driver's.
                # That includes a self-seed that gave up: it starts over.
                pipeline.reset()
                if seed_fault is not None:
                    logger.info("Driver identified - no_ear_baseline fault resolved (%s)",
                                "calibration loaded" if calibration is not None
                                else "self-seed restarts")
                    with prof.section("api"):
                        seed_fault.push(api_client, "resolved", now_mono,
                                        datetime.now(timezone.utc), "driver_identified")
                    seed_fault = None
                last_danger_push = 0.0

        with prof.section("pipeline"):
            m = pipeline.process(landmarks, thresholds, now, now_mono,
                                 self_seed=calibration is None or calibration.self_seed_ear)
        last_m = m
        if m.seed_failed and seed_fault is None:
            seed_fault = OpenFault(
                fault_uuid=str(uuid.uuid4()),
                fault_type=FAULT_TYPE_NO_EAR_BASELINE,
                driver_id=current_driver_id,
                entry_level=last_level or "ALERT",
                started_at=datetime.now(timezone.utc),
                last_known=None,     # nothing was ever scored
                gap_s=0.0,
                last_push_mono=now_mono,
            )
            with prof.section("api"):
                seed_fault.push(api_client, "open", now_mono)
        elif seed_fault is not None:
            with prof.section("api"):
                seed_fault.refresh_elapsed(api_client, now_mono)
        # (A gap that opened a fault was already logged above when it was resolved.)
        if m.gap_ended is not None and m.gap_ended.face_lost and not fault_resolved:
            logger.info("Face re-acquired after %.1fs (%s from %s)%s",
                        m.gap_ended.gap_s, m.gap_ended.band, m.gap_ended.entry_level,
                        " - DANGER latched until %.0fs below DANGER"
                        % pipeline.no_face.latch.release_hold if m.no_face_latch else "")

        # Log the full FRS breakdown whenever the effective level changes, so
        # the session log shows which term carried the score into the new
        # band rather than just the total.
        if m.level != last_level:
            logger.info(
                "Level %s -> %s: %s%s%s",
                last_level or "(none)", m.level,
                pipeline.frs_calc.format_breakdown(m.frs_result) if m.scored
                else "not scored (EAR self-seed gave up - no baseline)" if m.seed_failed
                else "not scored (self-seeding EAR baseline)",
                f"  [{m.override_reason} override forcing DANGER]" if m.overridden else "",
                f"  [band {m.band} held; raw {m.frs_result['level']}]"
                if m.scored and not m.overridden and m.band != m.frs_result["level"] else "",
            )
            logger.debug(
                "Level %s inputs: ear=%.3f (norm %.3f) perclos=%.1f%% (norm %.3f) "
                "blink_dur=%.0fms (norm %.3f) blink_freq=%.1f (norm %.3f) "
                "mar=%.3f (norm %.3f) yawn_norm=%.3f",
                m.level, m.ear, m.ear_norm, m.perclos, m.perclos_norm,
                m.blink_duration_ms, m.bd_norm, m.blink_freq, m.bf_norm,
                m.mar, m.mar_norm, m.yawn_norm,
            )
            last_level = m.level

        # Physical alerts - LEDs and buzzer only. m.level is DANGER while the
        # head-pose override is active regardless of the (lower) FRS level.
        with prof.section("gpio"):
            alert_manager.set_alert_level(m.level)

        # Operator notification: DANGER to the backend, rate-limited.
        if m.level == "DANGER" and now - last_danger_push >= DANGER_EVENT_INTERVAL:
            with prof.section("api"):
                api_client.push_fatigue_event(
                    current_driver_id, m.effective_result(), m.ear, m.perclos,
                    relay_triggered=False, phase=Phase.MONITORING,
                    provenance=(calibration.provenance() if calibration is not None
                                else SELF_SEEDED_PROVENANCE),
                )
            last_danger_push = now

        with prof.section("display"):
            draw_overlay(frame, driver, m, fps)
            draw_banner(frame, banner, banner_color)
            if debug_pose:
                draw_pose_debug(frame, pipeline.pose_debounce, now_mono, fps)
        present(frame)   # times its own heartbeat (api) and imshow (display)

    prof.stop()
    _profiler = None
    if open_fault is not None:
        # Ignition OFF with the driver still unseen: close the record so the
        # portal does not show a fault open forever.
        open_fault.push(api_client, "resolved", time.monotonic(),
                        datetime.now(timezone.utc), "ignition_off")
    if seed_fault is not None:
        seed_fault.push(api_client, "resolved", time.monotonic(),
                        datetime.now(timezone.utc), "ignition_off")
    logger.info("Monitoring: ignition turned OFF - returning to pre-drive")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Driver fatigue detection system")
    parser.add_argument(
        "--enroll", action="store_true",
        help="enrol a new driver (face encoding + 60 s calibration) and exit",
    )
    parser.add_argument(
        "--driver-id", type=int, default=None, metavar="N",
        help="with --enroll: the driver's DB id (skips the terminal prompt)",
    )
    parser.add_argument(
        "--mock-gpio", action="store_true",
        help="force AlertManager and IgnitionSensor into mock mode even on a Pi "
             "(press 'i' in the preview window to toggle the mock ignition)",
    )
    parser.add_argument(
        "--force-phase", choices=[p.value for p in Phase], default=None,
        help="ignore the ignition input and stay in this phase (testing one phase in "
             "isolation); in predrive a release does NOT continue into monitoring - "
             "'r' re-runs the assessment",
    )
    parser.add_argument(
        "--sequence", action="store_true",
        help="full session with no ignition input (bench / touchscreen): pre-drive, then - "
             "once a pass or an approved override releases the starter - monitoring "
             "until stopped",
    )
    parser.add_argument(
        "--profile-loop", type=float, nargs="?", const=60.0, default=None, metavar="SECONDS",
        help="monitoring: time every frame by category (capture, landmarks, recognition, "
             "pipeline, gpio, api, display) and log a summary every SECONDS (default 60); "
             "per-frame CSV under logs/",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="no OpenCV preview window and no frame annotation (demos / data collection; "
             "~11%% of monitoring frame time). Keys q / i / r are then unavailable - stop "
             "with Ctrl-C or the launcher's STOP",
    )
    parser.add_argument(
        "--debug-pose", action="store_true",
        help="overlay the head-pose debounce timers / override state on the preview",
    )
    parser.add_argument(
        "--diag-ear", action="store_true",
        help="TEMPORARY DIAGNOSTIC: log raw EAR, the active closure threshold and the "
             "closed/open decision on every transition. Use when PERCLOS, blink and "
             "microsleep all report nothing - they share this one comparison",
    )
    parser.add_argument(
        "--pose-release-hold", type=float, default=HEAD_POSE_RELEASE_HOLD_S, metavar="SECONDS",
        help="seconds the head pose must be normal before a head-pose DANGER "
             f"is released (default {HEAD_POSE_RELEASE_HOLD_S})",
    )
    args = parser.parse_args(argv)
    if args.driver_id is not None and not args.enroll:
        parser.error("--driver-id only makes sense with --enroll")
    if args.sequence and (args.force_phase or args.enroll):
        parser.error("--sequence cannot be combined with --force-phase or --enroll")
    if args.profile_loop is not None and args.profile_loop <= 0:
        parser.error("--profile-loop SECONDS must be positive")
    return args


def main(argv: Optional[list] = None) -> int:
    """
    Start-up sequence, then enrollment or the phase supervisor loop.

    ``cleanup()`` is guaranteed to run via ``try/finally`` so the camera is
    always released and GPIO reset (the starter is left inhibited -
    fail-secure).

    Returns:
        Process exit code.
    """
    global _camera, _alert_manager, _head_pose, _ignition, _heartbeat, _display_available

    args = parse_args(argv)
    setup_logging()
    if args.no_preview:
        _display_available = False
        logger.info("Preview window OFF (--no-preview) - stop with Ctrl-C or the launcher")
    logger.info("=== Driver Fatigue Detection System starting ===")
    exit_code = 0

    try:
        # 1. Config is imported at module load; log the key values.
        logger.info("Config: device %s, camera %dx%d@%dfps, API %s, calibration %ds, "
                    "pre-drive %ds (worst %.0fs window)",
                    config.DEVICE_ID, config.CAMERA_WIDTH, config.CAMERA_HEIGHT,
                    config.CAMERA_FPS, config.API_BASE_URL, config.CALIBRATION_DURATION,
                    config.PREDRIVE_ASSESSMENT_SECONDS, config.PREDRIVE_WORST_WINDOW_SECONDS)

        # 2. Backend
        api_client = APIClient()
        if not api_client.ping():
            logger.warning("Laravel backend is offline - continuing; pre-drive will lock "
                           "(no baseline) and monitoring will use defaults")

        # 3. Landmarks
        if not config.LANDMARK_MODEL.exists():
            logger.error("Landmark model not found at %s (see README.md)", config.LANDMARK_MODEL)
            return 1
        extractor = LandmarkExtractor(str(config.LANDMARK_MODEL), config.SCALE_FACTOR)

        # 4. Driver recognition
        recognizer = DriverRecognizer(api_client)
        loaded = recognizer.load_encodings()
        if loaded == 0 and not args.enroll:
            logger.warning("No driver encodings loaded - every driver will be UNKNOWN. "
                           "Run `python main.py --enroll` first.")

        # 5. Ignition (selects the phase) - forced, mocked, or real GPIO
        if args.force_phase:
            _ignition = ForcedIgnition(Phase(args.force_phase))
        elif args.sequence:
            # Reads as ignition OFF for good: pre-drive first, and monitoring
            # (entered from a release) never sees an ON -> OFF edge, so it
            # runs until stopped.
            _ignition = ForcedIgnition(Phase.PREDRIVE, source="--sequence")
        else:
            _ignition = IgnitionSensor(mock=args.mock_gpio)
        initial_phase = _ignition.phase()

        # 6. Alerts (auto-detects GPIO). Relay starts inhibited in pre-drive.
        _alert_manager = AlertManager(mock=args.mock_gpio, phase=initial_phase)

        # 6b. Head pose (camera intrinsics derived from the frame size)
        _head_pose = HeadPoseEstimator(
            frame_width=config.CAMERA_WIDTH, frame_height=config.CAMERA_HEIGHT
        )

        # 7. Camera
        _camera = Camera()

        if args.enroll:
            exit_code = run_enrollment(api_client, extractor, driver_id=args.driver_id)
        else:
            rate = LoopRate()
            # Heartbeat to the portal, ticked from present() in every phase
            # loop. Reads the relay state from _alert_manager at send time.
            _heartbeat = Heartbeat(api_client, _alert_manager)
            logger.info("Supervisor started in %s - %s to stop "
                        "(heartbeat every %.0fs, firmware %s)",
                        initial_phase.value,
                        "press 'q' in the window or Ctrl-C" if _display_available else "Ctrl-C",
                        config.HEARTBEAT_INTERVAL_SECONDS,
                        config.FIRMWARE_VERSION)
            # 8. Phase supervisor. A pre-drive that releases the starter (pass
            #    or approved override) goes straight into monitoring - no key
            #    press - except under --force-phase predrive, which tests that
            #    phase alone. Monitoring returns when the ignition turns OFF,
            #    and every pre-drive starts by re-inhibiting the starter.
            continue_to_monitoring = args.force_phase is None
            while True:
                if _ignition.phase() is Phase.PREDRIVE:
                    released = run_predrive_assessment(
                        api_client, extractor, recognizer, _alert_manager, _head_pose,
                        _ignition, rate, debug_pose=args.debug_pose, diag_ear=args.diag_ear,
                        pose_release_hold=args.pose_release_hold,
                        continue_to_monitoring=continue_to_monitoring,
                    )
                    if not released:
                        # 'r' (re-run), or ignition ON before a release:
                        # re-read the phase, starter still inhibited.
                        continue
                    run_monitoring(
                        api_client, extractor, recognizer, _alert_manager, _head_pose,
                        _ignition, rate, debug_pose=args.debug_pose, diag_ear=args.diag_ear,
                        pose_release_hold=args.pose_release_hold, from_release=True,
                        profile_s=args.profile_loop,
                    )
                else:
                    run_monitoring(
                        api_client, extractor, recognizer, _alert_manager, _head_pose,
                        _ignition, rate, debug_pose=args.debug_pose, diag_ear=args.diag_ear,
                        pose_release_hold=args.pose_release_hold,
                        profile_s=args.profile_loop,
                    )

    except QuitRequested:
        logger.info("Quit requested from the preview window")
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl-C)")
    except Exception:
        logger.exception("Fatal error in main loop")
        exit_code = 1
    finally:
        cleanup()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
