"""Pre-drive: pass -> hand-off, isolated pass, override approved / denied."""

import time

import pytest

from fakes import CAL, FakeAPI, FakeExtractor, FakeRecognizer, keys_after
from modules.phase import Phase

pytestmark = pytest.mark.slow

MATCH = {"driver_id": 6, "name": "Test Driver", "confidence": 0.6}


@pytest.fixture
def ign(m):
    return m.ForcedIgnition(Phase.PREDRIVE, source="tests")


def test_pass_hands_off_to_monitoring(m, rig, hp, ign, monkeypatch):
    monkeypatch.setattr(m, "identify_driver_bounded", lambda *a, **k: MATCH)
    monkeypatch.setattr(m, "load_calibration", lambda *a, **k: (CAL, CAL.provenance()))
    api = FakeAPI()
    am = rig(api, ign)
    t = time.monotonic()
    released = m.run_predrive_assessment(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, ign,
                                         m.LoopRate(), continue_to_monitoring=True)
    elapsed = time.monotonic() - t
    assert released and not am.relay_locked
    assert elapsed - 4.0 < 2.0, "release indicator under 2 s"


def test_isolated_pass_waits_for_r(m, rig, hp, ign, monkeypatch):
    """--force-phase predrive: no hand-off; 'r' on the start-vehicle screen re-runs."""
    monkeypatch.setattr(m, "identify_driver_bounded", lambda *a, **k: MATCH)
    monkeypatch.setattr(m, "load_calibration", lambda *a, **k: (CAL, CAL.provenance()))
    monkeypatch.setattr(m, "show_frame", keys_after(200, m.KEY_RETRY))
    api = FakeAPI()
    am = rig(api, ign)
    released = m.run_predrive_assessment(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, ign,
                                         m.LoopRate(), continue_to_monitoring=False)
    assert released is False


def test_not_recognised_override_approved(m, rig, hp, ign, monkeypatch):
    monkeypatch.setattr(m, "identify_driver_bounded", lambda *a, **k: None)
    api = FakeAPI("approved")
    am = rig(api, ign)
    released = m.run_predrive_assessment(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, ign,
                                         m.LoopRate(), continue_to_monitoring=True)
    assert released and not am.relay_locked
    assert "request_override" in api.calls


def test_not_recognised_override_denied_stays_locked(m, rig, hp, ign, monkeypatch):
    monkeypatch.setattr(m, "identify_driver_bounded", lambda *a, **k: None)
    monkeypatch.setattr(m, "show_frame", keys_after(250, m.KEY_RETRY))   # > 5 s, then 'r'
    api = FakeAPI("denied")
    am = rig(api, ign)
    released = m.run_predrive_assessment(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, ign,
                                         m.LoopRate(), continue_to_monitoring=True)
    assert released is False and am.relay_locked, "no fall-through on a denial"
