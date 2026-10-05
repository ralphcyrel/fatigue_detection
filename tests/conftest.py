"""
Shared fixtures. ``m`` is main.py with every module-level handle reset to a
fake for the duration of one test and restored afterwards, so tests can run
in any order. Nothing touches the camera, GPIO, the backend or ``logs/``.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import main  # noqa: E402
from fakes import FakeCam, ShortAssessment  # noqa: E402
from modules.alert import AlertManager  # noqa: E402
from modules.head_pose import HeadPoseEstimator  # noqa: E402
from modules.phase import Phase  # noqa: E402


@pytest.fixture
def m(monkeypatch, tmp_path):
    """main.py, isolated: fake camera, no window, logs under tmp_path."""
    monkeypatch.setattr(main.config, "LOGS_DIR", tmp_path)
    for name, value in {
        "_camera": FakeCam(), "_alert_manager": None, "_heartbeat": None, "_ignition": None,
        "_head_pose": None, "_profiler": None, "_rate": None, "_stream": None,
        # Video mode with no window: every frame goes through show_frame,
        # which the tests replace to inject keys.
        "_display_mode": "video", "_display_available": False, "_fullscreen": False,
        "_window_ready": False, "_reserve_bottom": 0, "_enroll_report": {},
        "_last_screen_draw": float("-inf"),
    }.items():
        monkeypatch.setattr(main, name, value)
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
