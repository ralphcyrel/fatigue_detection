"""
Shared fixtures. ``m`` is main.py with every module-level handle reset to a
fake for the duration of one test and restored afterwards, so tests can run
in any order. Nothing touches the camera, GPIO, the backend or ``logs/``.
"""

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import main  # noqa: E402
from fakes import (CAL, FakeAPI, FakeCam, FakeExtractor, FakeRecognizer,  # noqa: E402
                   ShortAssessment)
from modules.alert import AlertManager  # noqa: E402
from modules.head_pose import HeadPoseEstimator  # noqa: E402
from modules.phase import Phase  # noqa: E402
from modules.recognition_worker import RecognitionWorker  # noqa: E402
from modules.start_confirm import OperatorRequests, TouchStartConfirm  # noqa: E402


@pytest.fixture
def m(monkeypatch, tmp_path):
    """main.py, isolated: fake camera, no window, logs under tmp_path."""
    monkeypatch.setattr(main.config, "LOGS_DIR", tmp_path)
    # Enrollment stores the neutral head pose; keep it out of data/.
    monkeypatch.setattr(main.config, "POSE_NEUTRAL_FILE", tmp_path / "pose_neutral.json")
    for name, value in {
        "_camera": FakeCam(), "_alert_manager": None, "_heartbeat": None, "_ignition": None,
        "_head_pose": None, "_profiler": None, "_rate": None, "_stream": None,
        # Video mode with no window: every frame goes through show_frame,
        # which the tests replace to inject keys.
        "_display_mode": "video", "_display_available": False, "_fullscreen": False,
        "_window_ready": False, "_reserve_bottom": 0, "_enroll_report": {},
        "_last_screen_draw": float("-inf"),
        "_session_state": None, "_state_note": "", "_state_file": None, "_recognition": None,
        "_recognition_factory": None, "_calibration_cache": {}, "_released_driver": None,
    }.items():
        monkeypatch.setattr(main, name, value)
    # Fresh operator requests: nothing pending from another test.
    requests = OperatorRequests()
    monkeypatch.setattr(main, "_requests", requests)
    monkeypatch.setattr(main, "_start_confirm", TouchStartConfirm(requests))
    monkeypatch.setattr(main, "show_frame", lambda frame: -1)
    monkeypatch.setattr(main, "PredriveAssessment", ShortAssessment)
    yield main
    if main._stream is not None:
        main._stream.close()


@pytest.fixture
def hp():
    return HeadPoseEstimator(640, 480)


@pytest.fixture
def rig(m):
    """``rig(api, ignition, phase)``: wire fakes into main's globals (mock GPIO only)."""
    def setup(api, ignition, phase=Phase.PREDRIVE):
        m._alert_manager = AlertManager(mock=True, phase=phase)
        m._heartbeat = m.Heartbeat(api, m._alert_manager)
        m._ignition = ignition
        return m._alert_manager
    return setup


@pytest.fixture
def patched_main(m, monkeypatch):
    """main() with fakes for every device; GPIO forced to mock even on a Pi."""
    monkeypatch.setattr(m, "setup_logging", lambda: None)
    monkeypatch.setattr(m, "APIClient", lambda: FakeAPI())
    monkeypatch.setattr(m, "LandmarkExtractor", lambda *a, **k: FakeExtractor())
    monkeypatch.setattr(m, "DriverRecognizer", lambda api: FakeRecognizer(0.0))
    # No worker factory -> no worker process: recognition inline on the fake
    # (instant) recogniser. Tests that want the real worker patch KnownFaces.of.
    monkeypatch.setattr(m.KnownFaces, "of", classmethod(lambda cls, r: None))
    monkeypatch.setattr(m, "Camera", FakeCam)
    monkeypatch.setattr(m, "AlertManager",
                        lambda mock=False, phase=None: AlertManager(mock=True, phase=phase))
    monkeypatch.setattr(m, "identify_driver_bounded",
                        lambda *a, **k: {"driver_id": 6, "name": "Test Driver", "confidence": 0.6})
    monkeypatch.setattr(m, "load_calibration", lambda *a, **k: (CAL, CAL.provenance()))
    monkeypatch.setattr(m.config, "LANDMARK_MODEL", m.config.BASE_DIR / "main.py")  # exists()
    return m


@pytest.fixture
def worker(m):
    """``worker(factory)``: a real RecognitionWorker process, started and ready."""
    made = []

    def make(factory):
        w = RecognitionWorker(factory)
        w.start()
        made.append(w)
        deadline = time.monotonic() + 60.0
        while not w.ready and not w.failed and time.monotonic() < deadline:
            w.poll()
            time.sleep(0.05)
        assert w.ready, "worker started"
        m._recognition = w
        return w
    yield make
    for w in made:
        w.close()
