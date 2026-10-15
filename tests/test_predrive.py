"""Pre-drive: pass -> hand-off, isolated pass, override approved / denied."""

import time

import numpy as np
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
    # Settling, then the 4 s ShortAssessment, then the release indicator.
    assert elapsed - (m.config.PREDRIVE_SETTLE_SECONDS + 4.0) < 2.0, "release indicator under 2 s"


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


def test_never_settles_is_invalid_assessment(m, rig, hp, ign, monkeypatch):
    """No valid face after recognition: FAULT, raised as driver_not_recognized."""
    monkeypatch.setattr(m, "identify_driver_bounded", lambda *a, **k: MATCH)
    monkeypatch.setattr(m, "load_calibration", lambda *a, **k: (CAL, CAL.provenance()))
    monkeypatch.setattr(m.config, "PREDRIVE_SETTLE_TIMEOUT_S", 1.0)
    monkeypatch.setattr(m, "show_frame", keys_after(60, m.KEY_RETRY))
    seen = {}
    real = m.await_override

    def spy(*a, **k):
        seen["reason"], seen["display"] = a[5], k.get("display_reason")
        return real(*a, **k)
    monkeypatch.setattr(m, "await_override", spy)
    api = FakeAPI("pending")
    am = rig(api, ign)
    released = m.run_predrive_assessment(api, FakeExtractor(face=lambda: False),
                                         FakeRecognizer(0.0), am, hp, ign, m.LoopRate())
    assert released is False and am.relay_locked
    assert seen == {"reason": m.LockReason.DRIVER_NOT_RECOGNIZED,
                    "display": "INVALID ASSESSMENT"}
    assert "post_assessment" not in api.calls, "no assessment was run"


def test_too_few_valid_frames_is_invalid_assessment(m, rig, hp, ign, monkeypatch):
    """Settles, then loses the face on every other frame: < 80 % valid -> FAULT."""
    monkeypatch.setattr(m, "identify_driver_bounded", lambda *a, **k: MATCH)
    monkeypatch.setattr(m, "load_calibration", lambda *a, **k: (CAL, CAL.provenance()))
    monkeypatch.setattr(m, "show_frame", keys_after(400, m.KEY_RETRY))
    t0, n = time.monotonic(), {"i": 0}

    def face():
        n["i"] += 1
        return time.monotonic() - t0 < m.config.PREDRIVE_SETTLE_SECONDS + 0.3 or n["i"] % 2 == 0
    results = {}
    monkeypatch.setattr(m.PredriveAssessment, "result",
                        lambda self, _r=m.PredriveAssessment.result:
                        results.setdefault("r", _r(self)))
    api = FakeAPI("pending")
    am = rig(api, ign)
    released = m.run_predrive_assessment(api, FakeExtractor(face=face), FakeRecognizer(0.0),
                                         am, hp, ign, m.LoopRate())
    assert released is False and am.relay_locked
    r = results["r"]
    assert r["void"] and r["invalid_cause"] == "INVALID_ASSESSMENT"
    assert r["valid_fraction"] < 0.8


class ScriptedFace:
    """Extractor whose face state follows the time since its first call."""

    def __init__(self, state_at):
        from fakes import close_eyes, synthetic_landmarks, FakeRect
        self.state_at, self.t0, self.n = state_at, None, 0
        self.rng = np.random.default_rng(0)
        self._close, self._lm, self._rect = close_eyes, synthetic_landmarks, FakeRect

    def extract(self, frame):
        time.sleep(0.015)
        self.t0 = self.t0 or time.monotonic()
        self.n += 1
        state = self.state_at(time.monotonic() - self.t0, self.n)
        if state is None:
            return None, None
        lm = self._lm(self.rng)
        return (self._close(lm) if state == "closed" else lm), self._rect(200, 120, 440, 360)


def _spy_reason(m, monkeypatch):
    seen = {}
    real = m.await_override

    def spy(*a, **k):
        seen["reason"], seen["display"] = a[5], k.get("display_reason")
        return real(*a, **k)
    monkeypatch.setattr(m, "await_override", spy)
    return seen


def test_microsleep_during_settling_fails(m, rig, hp, ign, monkeypatch):
    monkeypatch.setattr(m, "identify_driver_bounded", lambda *a, **k: MATCH)
    monkeypatch.setattr(m, "load_calibration", lambda *a, **k: (CAL, CAL.provenance()))
    monkeypatch.setattr(m, "show_frame", keys_after(60, m.KEY_RETRY))
    seen = _spy_reason(m, monkeypatch)
    api = FakeAPI("pending")
    am = rig(api, ign)
    face = ScriptedFace(lambda t, n: "closed" if 1.0 <= t < 2.6 else "open")
    released = m.run_predrive_assessment(api, face, FakeRecognizer(0.0), am, hp, ign,
                                         m.LoopRate())
    assert released is False and am.relay_locked
    assert seen["reason"] is m.LockReason.MICROSLEEP_DETECTED and seen["display"] is None
    assert "post_assessment" not in api.calls, "failed in settling, no 30 s run"


def test_microsleep_outranks_invalid_assessment(m, rig, hp, ign, monkeypatch):
    monkeypatch.setattr(m, "identify_driver_bounded", lambda *a, **k: MATCH)
    monkeypatch.setattr(m, "load_calibration", lambda *a, **k: (CAL, CAL.provenance()))
    monkeypatch.setattr(m, "show_frame", keys_after(400, m.KEY_RETRY))
    seen = _spy_reason(m, monkeypatch)
    results = {}
    monkeypatch.setattr(m.PredriveAssessment, "result",
                        lambda self, _r=m.PredriveAssessment.result:
                        results.setdefault("r", _r(self)))
    settle = m.config.PREDRIVE_SETTLE_SECONDS

    def state(t, n):
        if t < settle + 0.4:
            return "open"
        if t < settle + 1.8:
            return "closed"                     # 1.4 s: a microsleep
        return None if n % 2 else "open"        # then half the frames invalid
    api = FakeAPI("pending")
    am = rig(api, ign)
    m.run_predrive_assessment(api, ScriptedFace(state), FakeRecognizer(0.0), am, hp, ign,
                              m.LoopRate())
    r = results["r"]
    assert r["void"] and r["microsleeps"] >= 1
    assert seen["reason"] is m.LockReason.MICROSLEEP_DETECTED
