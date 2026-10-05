"""main() end to end per CLI mode: relay starts inhibited, monitoring never locks it."""

import logging

import pytest

from fakes import CAL, FakeAPI, FakeCam, FakeExtractor, FakeRecognizer, keys_after
from modules.alert import AlertManager
from modules.ignition import IgnitionSensor

pytestmark = pytest.mark.slow


@pytest.fixture
def patched_main(m, monkeypatch):
    """main() with fakes for every device; GPIO forced to mock even on a Pi."""
    monkeypatch.setattr(m, "setup_logging", lambda: None)
    monkeypatch.setattr(m, "APIClient", lambda: FakeAPI())
    monkeypatch.setattr(m, "LandmarkExtractor", lambda *a, **k: FakeExtractor())
    monkeypatch.setattr(m, "DriverRecognizer", lambda api: FakeRecognizer(0.0))
    monkeypatch.setattr(m, "Camera", FakeCam)
    monkeypatch.setattr(m, "AlertManager",
                        lambda mock=False, phase=None: AlertManager(mock=True, phase=phase))
    monkeypatch.setattr(m, "IgnitionSensor", lambda mock=False: IgnitionSensor(mock=True))
    monkeypatch.setattr(m, "identify_driver_bounded",
                        lambda *a, **k: {"driver_id": 6, "name": "Test Driver", "confidence": 0.6})
    monkeypatch.setattr(m, "load_calibration", lambda *a, **k: (CAL, CAL.provenance()))
    monkeypatch.setattr(m.config, "LANDMARK_MODEL", m.config.BASE_DIR / "main.py")  # exists()
    return m


@pytest.mark.parametrize("argv, frames", [
    (["--sequence"], 300),
    (["--force-phase", "predrive"], 300),
    (["--force-phase", "monitoring"], 60),
    (["--mock-gpio"], 300),
])
def test_supervisor_modes(patched_main, monkeypatch, caplog, argv, frames):
    m = patched_main
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(m, "show_frame", keys_after(frames, m.KEY_QUIT))
    m.main(argv)
    lines = [r.getMessage() for r in caplog.records]
    mon = [line for line in lines if "=== MONITORING" in line]
    pre = [line for line in lines if "=== PRE-DRIVE" in line]
    assert any("Relay starts INHIBITED" in line for line in lines)
    assert not any("Relay lock REFUSED" in line for line in lines), \
        "no relay actuation attempted from monitoring"
    if argv[0] in ("--sequence", "--mock-gpio"):
        assert len(mon) == 1 and "entered from pre-drive release) - starter UNLOCKED" in mon[0]
    elif argv[-1] == "predrive":
        assert not mon, "isolated pre-drive never enters monitoring"
    else:
        assert len(mon) == 1 and "starter LOCKED" in mon[0] and not pre
