"""
The recognition worker (2026-10-19): monitoring's identify() off the frame
loop, a microsleep during a call still detected, answers from before a face
loss discarded, only one worker, and none left behind - by END SESSION, a
crash, or main.py being killed outright.
"""

import logging
import multiprocessing
import os
import subprocess
import sys
import textwrap
import time

import pytest

from fakes import (FakeAPI, FakeExtractor, FakeRecognizer, Operator, ScriptedIgnition,
                   SlowRecognizerFactory, TimedFace)
from fakes import monitoring_summaries as summaries
from modules.phase import Phase, SessionState
from modules.recognition_worker import RecognitionWorker

pytestmark = pytest.mark.slow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READY, IDLE = SessionState.READY_TO_START, SessionState.IDLE


def pid_alive(pid: int) -> bool:
    """Whether process ``pid`` is still running (Windows and POSIX)."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259                             # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_gone(pid: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    return not pid_alive(pid)


# ---- monitoring off the frame loop --------------------------------------------

def test_monitoring_retries_never_block_the_frame_loop(m, rig, hp, tmp_path, worker):
    """Unrecognised driver, 1 s identify() in the worker: retried every 5 s, no stall."""
    w = worker(SlowRecognizerFactory(cost=1.0, driver_id=None))
    api = FakeAPI()
    am = rig(api, ScriptedIgnition([(13.0, Phase.MONITORING), (99, Phase.PREDRIVE)]),
             Phase.MONITORING)
    m.run_monitoring(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, m._ignition, m.LoopRate())
    (rec,) = summaries(tmp_path)
    assert rec["recognition"] == "worker" and rec["end_reason"] == "ignition_off"
    assert rec["max_frame_gap_s"] < m.config.MAX_FRAME_GAP_S, rec
    assert rec["frame_gaps_over_limit"] == 0
    assert rec["identify_attempts"] == 3 and w.calls >= 2, "t = 0 / ~6 / ~12 s"
    assert rec["driver_id"] is None
    print(f"\nworst frame gap, monitoring with the worker: {rec['max_frame_gap_s'] * 1000:.0f} ms")


def test_monitoring_identifies_once_through_the_worker(m, rig, hp, tmp_path, worker):
    w = worker(SlowRecognizerFactory(cost=1.0))
    api = FakeAPI()
    am = rig(api, ScriptedIgnition([(4.0, Phase.MONITORING), (99, Phase.PREDRIVE)]),
             Phase.MONITORING)
    m.run_monitoring(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, m._ignition, m.LoopRate())
    (rec,) = summaries(tmp_path)
    assert rec["driver_id"] == 6 and rec["identify_attempts"] == 1 and w.calls == 1
    assert rec["identified_after_s"] >= 1.0, "the answer came from the 1 s call"
    assert rec["max_frame_gap_s"] < m.config.MAX_FRAME_GAP_S
    assert api.calls.count("get_driver_calibration") == 1


def test_monitoring_discards_an_answer_from_before_a_face_loss(m, rig, hp, tmp_path, worker,
                                                                caplog):
    """The face goes for 2.9 s while identify() runs: that answer is void, a new one is asked."""
    worker(SlowRecognizerFactory(cost=1.5))
    face = TimedFace(lambda t: None if 0.3 <= t < 3.2 else "open")
    api = FakeAPI()
    am = rig(api, ScriptedIgnition([(7.0, Phase.MONITORING), (99, Phase.PREDRIVE)]),
             Phase.MONITORING)
    caplog.set_level(logging.INFO, logger="fatigue")
    m.run_monitoring(api, face, FakeRecognizer(0.0), am, hp, m._ignition, m.LoopRate())
    (rec,) = summaries(tmp_path)
    assert "discarded (from before a face loss)" in caplog.text
    assert rec["identify_attempts"] == 2 and rec["driver_id"] == 6
    assert rec["identified_after_s"] > 3.2, "identified on a frame after the face came back"


def test_microsleep_during_monitoring_identify_is_detected(m, rig, hp, worker, caplog):
    """
    Unrecognised driver; the session-start identify() takes 4 s in the worker.
    A 1.1 s closure from 2.4 s (after the 2 s provisional self-seed threshold)
    falls entirely inside that call and is confirmed as a microsleep.
    """
    w = worker(SlowRecognizerFactory(cost=4.0, driver_id=None))
    busy = []

    def state(t):
        if 2.4 <= t < 3.5:
            busy.append(w.busy)
            return "closed"
        return "open"
    api = FakeAPI()
    am = rig(api, ScriptedIgnition([(6.0, Phase.MONITORING), (99, Phase.PREDRIVE)]),
             Phase.MONITORING)
    caplog.set_level(logging.INFO, logger="modules.pipeline")
    m.run_monitoring(api, TimedFace(state), FakeRecognizer(0.0), am, hp, m._ignition,
                     m.LoopRate())
    assert busy and all(busy), "the whole closure happened during the identify() call"
    assert "MICROSLEEP confirmed" in caplog.text


# ---- one worker, never an orphan ----------------------------------------------

def test_only_one_worker_can_run(m, worker):
    worker(SlowRecognizerFactory(cost=0.1))
    second = RecognitionWorker(SlowRecognizerFactory(cost=0.1))
    with pytest.raises(RuntimeError, match="already running"):
        second.start()


@pytest.fixture
def real_worker_main(patched_main, monkeypatch):
    """main() with a real RecognitionWorker (fake 0.2 s recogniser); records every one started."""
    m = patched_main
    monkeypatch.setattr(m.KnownFaces, "of",
                        classmethod(lambda cls, r: SlowRecognizerFactory(cost=0.2)))
    started = []
    real_start = m.start_recognition

    def spy():
        real_start()
        if m._recognition is not None and m._recognition not in started:
            started.append(m._recognition)
    monkeypatch.setattr(m, "start_recognition", spy)
    return m, started


def test_end_session_stops_the_worker_and_the_next_driver_gets_a_new_one(real_worker_main,
                                                                        monkeypatch):
    m, started = real_worker_main
    op = Operator(m, (READY, 0.3, m.KEY_END_SESSION), (IDLE, 0.5, m.KEY_BEGIN),
                  (READY, 0.3, m.KEY_END_SESSION), (IDLE, 0.5, m.KEY_QUIT), timeout_s=90.0)
    at_idle = []

    def show(frame):
        if m._session_state is IDLE:
            at_idle.append((m._recognition, [w.alive for w in started]))
        return op(frame)
    monkeypatch.setattr(m, "show_frame", show)
    assert m.main(["--sequence"]) == 0
    assert op.steps == [], f"stopped early: {op.states()}"
    assert len(started) == 2 and started[0] is not started[1], "a fresh worker per driver"
    assert at_idle and all(current is None and not any(alive) for current, alive in at_idle), \
        "no worker process in IDLE"
    assert not any(pid_alive(w.pid) for w in started)
    assert RecognitionWorker._live is None and multiprocessing.active_children() == []


def test_a_crash_leaves_no_worker(real_worker_main, monkeypatch):
    m, started = real_worker_main
    calls = {"n": 0}

    class Crashing(FakeExtractor):
        def extract(self, frame):
            calls["n"] += 1
            if calls["n"] > 60:
                raise RuntimeError("camera pipeline died")
            return super().extract(frame)
    monkeypatch.setattr(m, "LandmarkExtractor", lambda *a, **k: Crashing())
    monkeypatch.setattr(m, "show_frame", Operator(m, timeout_s=60.0))
    assert m.main(["--sequence"]) == 1
    assert started and not any(w.alive or pid_alive(w.pid) for w in started)
    assert m._recognition is None and RecognitionWorker._live is None
    assert multiprocessing.active_children() == []


KILLED_PARENT = textwrap.dedent("""
    import os, sys, time
    sys.path[:0] = [{root!r}, {tests!r}]
    import numpy as np
    from fakes import SlowRecognizerFactory
    from modules.recognition_worker import RecognitionWorker

    if __name__ == "__main__":
        w = RecognitionWorker(SlowRecognizerFactory(cost=5.0))
        w.start()
        while not w.ready:
            w.poll()
            time.sleep(0.05)
        if {in_flight!r}:
            w.submit(np.zeros((48, 64, 3), np.uint8), 0)     # identify() running
        print(w.pid, flush=True)
        time.sleep(0.3)
        os._exit(1)            # no cleanup, no atexit - as if SIGKILLed
""")


@pytest.mark.parametrize("in_flight", [False, True])
def test_a_killed_main_leaves_no_worker(tmp_path, in_flight):
    """The parent dies without closing the worker: the worker exits on its own."""
    script = tmp_path / "killed_parent.py"
    script.write_text(KILLED_PARENT.format(root=ROOT, tests=os.path.join(ROOT, "tests"),
                                           in_flight=in_flight), encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 1, done.stderr
    pid = int(done.stdout.strip().splitlines()[-1])
    assert wait_gone(pid), f"worker pid {pid} outlived its parent"
