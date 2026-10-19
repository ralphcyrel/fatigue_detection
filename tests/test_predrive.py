"""
Pre-drive: rolling, start-gated assessment (2026-10-19) - READY_TO_START is
held without advancing, entered only after the minimum duration, lost and
regained on face loss, lost on a change of driver; the start confirmation
advances to monitoring; END SESSION returns to IDLE. Plus the hard failures:
override approved / denied, never settles, too few valid frames, microsleep.

The tests' ShortAssessment has a 4 s minimum (the real one is 30 s) and a
2 s worst window, so READY_TO_START comes ~7 s in (3 s settling + 4 s).
"""

import time

import numpy as np
import pytest

from fakes import CAL, FakeAPI, FakeExtractor, FakeRecognizer, Operator, ShortAssessment, keys_after
from modules.assessment import PredriveAssessment
from modules.phase import Phase, SessionState

pytestmark = pytest.mark.slow

MATCH = {"driver_id": 6, "name": "Test Driver", "confidence": 0.6}
READY, ASSESSING, IDLE = (SessionState.READY_TO_START, SessionState.ASSESSING,
                          SessionState.IDLE)


@pytest.fixture
def ign(m):
    return m.ForcedIgnition(Phase.PREDRIVE, source="tests")


@pytest.fixture
def known_driver(m, monkeypatch):
    """Recognition and calibration succeed instantly (driver 6, own unit)."""
    monkeypatch.setattr(m, "identify_driver_bounded", lambda *a, **k: MATCH)
    monkeypatch.setattr(m, "load_calibration", lambda *a, **k: (CAL, CAL.provenance()))


@pytest.fixture
def transitions(m, monkeypatch):
    """Every set_session_state call: (state, note, relay_locked, monotonic time)."""
    seen = []
    real = m.set_session_state

    def record(state, note=""):
        real(state, note)
        am = m._alert_manager
        seen.append((state, note, am.relay_locked if am else None, time.monotonic()))
    monkeypatch.setattr(m, "set_session_state", record)
    return seen


@pytest.fixture
def runs(m, monkeypatch):
    """Every PredriveAssessment the code under test creates (ShortAssessment)."""
    made = []

    class Kept(ShortAssessment):
        def __init__(self):
            super().__init__()
            made.append(self)
    monkeypatch.setattr(m, "PredriveAssessment", Kept)
    return made


def states(transitions):
    return [s for s, *_ in transitions]


def test_pass_holds_in_ready_without_advancing(m, rig, hp, ign, known_driver, transitions, runs,
                                               monkeypatch):
    """A pass releases the starter and then just waits: no timeout into monitoring."""
    op = Operator(m, (READY, 6.0, m.KEY_QUIT))      # 6 s in READY, nobody taps BEGIN
    monkeypatch.setattr(m, "show_frame", op)
    api = FakeAPI()
    am = rig(api, ign)
    with pytest.raises(m.QuitRequested):
        m.run_predrive_assessment(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, ign,
                                  m.LoopRate())
    seq = states(transitions)
    assert seq[-1] is READY and SessionState.MONITORING not in seq
    assert seq.count(READY) == 1, "READY held throughout - nothing advanced or dropped it"
    assert not am.relay_locked, "starter released in READY_TO_START"
    assert am.current_level == "ALERT" and am.pin_states[m.config.LED_GREEN] == 1, "green LED"
    assert m._screen.status[0][0] == "Assessment passed - you may start the vehicle"
    ready = transitions[seq.index(READY)]
    assert ready[2] is False, "relay released before READY_TO_START was published"
    events = [e["event"] for e in runs[0].timeline]
    assert events == ["ready_entered", "ended"], "no confirmation was recorded"
    assert runs[0].timeline_summary()["end_reason"] == "stopped"
    assert runs[0].timeline_summary()["predrive_duration_s"] is None
    assert "post_assessment" in api.calls, "the first-window verdict went to the backend"


def test_confirmation_advances_to_monitoring(m, rig, hp, ign, known_driver, runs, monkeypatch):
    """BEGIN MONITORING in READY_TO_START: True (-> monitoring), times logged apart."""
    op = Operator(m, (READY, 1.0, m.KEY_BEGIN))
    monkeypatch.setattr(m, "show_frame", op)
    api = FakeAPI()
    am = rig(api, ign)
    t = time.monotonic()
    released = m.run_predrive_assessment(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, ign,
                                         m.LoopRate(), continue_to_monitoring=True)
    elapsed = time.monotonic() - t
    assert released and not am.relay_locked
    # Settling, the 4 s minimum, 1 s in READY, then the release indicator.
    assert elapsed - (m.config.PREDRIVE_SETTLE_SECONDS + 4.0 + 1.0) < 2.5
    run = runs[0]
    events = {e["event"]: e for e in run.timeline}
    assert list(events) == ["ready_entered", "confirmed", "ended"]
    assert events["confirmed"]["t_abs"] - events["ready_entered"]["t_abs"] >= 1.0
    summary = run.timeline_summary()
    assert summary["end_reason"] == "confirmed"
    assert summary["first_ready_at"] < summary["confirmed_at"]
    assert summary["predrive_duration_s"] == pytest.approx(
        summary["confirmed_at"] - summary["predrive_started_at"], abs=1e-3)


def test_isolated_pass_r_reruns(m, rig, hp, ign, known_driver, monkeypatch):
    """--force-phase predrive: 'r' on the READY_TO_START screen re-runs (False, relocked)."""
    monkeypatch.setattr(m, "show_frame", Operator(m, (READY, 0.5, m.KEY_RETRY)))
    api = FakeAPI()
    am = rig(api, ign)
    released = m.run_predrive_assessment(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, ign,
                                         m.LoopRate(), continue_to_monitoring=False)
    assert released is False and am.relay_locked


def test_not_recognised_override_approved(m, rig, hp, ign, monkeypatch):
    """An approved override releases the starter, then waits for the confirmation too."""
    monkeypatch.setattr(m, "identify_driver_bounded", lambda *a, **k: None)
    op = Operator(m, (READY, 0.5, m.KEY_BEGIN))
    monkeypatch.setattr(m, "show_frame", op)
    api = FakeAPI("approved")
    am = rig(api, ign)
    released = m.run_predrive_assessment(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, ign,
                                         m.LoopRate(), continue_to_monitoring=True)
    assert released and not am.relay_locked
    assert "request_override" in api.calls
    assert op.states()[-2:] == [SessionState.LOCKED, READY]


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


# ---- READY_TO_START is not a latch (2026-10-19) -------------------------------

def test_ready_lost_on_face_loss_and_regained(m, rig, hp, ign, known_driver, transitions,
                                              monkeypatch):
    """No face > NO_FACE_FAULT_S drops READY (starter inhibited); it comes back after."""
    runs = []

    class LowValidBar(ShortAssessment):
        # In a 4 s window a 2.4 s gap is 60 % of the frames, and the 80 %
        # valid-frame rule would drop READY before the face rule; in the real
        # 30 s window the same gap is 8 %. A 30 % bar isolates the face rule.
        def __init__(self):
            PredriveAssessment.__init__(self, duration_s=4.0, worst_window_s=2.0,
                                        min_valid_fraction=0.3)
            runs.append(self)
    monkeypatch.setattr(m, "PredriveAssessment", LowValidBar)
    gap = {}

    def state(t, n):
        if gap.get("from") is None and m._session_state is READY:
            gap["from"] = t + 0.5                    # half a second into READY
        start = gap.get("from")
        if start is not None and start <= t < start + 2.4:
            return None                              # 2.4 s: longer than NO_FACE_FAULT_S
        return "open"

    began = time.monotonic()

    def show(frame):
        readies = [t for s, _note, _locked, t in transitions if s is READY]
        if len(readies) >= 2 and time.monotonic() - readies[-1] > 0.5:
            return m.KEY_QUIT                         # regained and held: done
        if time.monotonic() - began > 40.0:
            return m.KEY_QUIT                         # never regained: the asserts fail
        return -1
    monkeypatch.setattr(m, "show_frame", show)
    rec = FakeRecognizer(0.0)
    api = FakeAPI()
    am = rig(api, ign)
    with pytest.raises(m.QuitRequested):
        m.run_predrive_assessment(api, ScriptedFace(state), rec, am, hp, ign, m.LoopRate())
    seq = [(s, locked) for s, _note, locked, _t in transitions]
    i = seq.index((READY, False))
    lost = seq[i + 1]
    assert lost == (ASSESSING, True), "READY lost -> assessing, starter inhibited"
    lost_note = transitions[i + 1][1]
    assert lost_note.startswith("no face for"), lost_note
    lost_after = transitions[i + 1][3] - transitions[i][3]
    assert lost_after < 0.5 + m.config.NO_FACE_FAULT_S + 0.5, "dropped at ~2 s of no face"
    assert (READY, False) in seq[i + 2:], "READY regained once the face is back"
    assert rec.calls >= 1, "the driver was re-identified before READY came back"
    events = [e["event"] for e in runs[0].timeline]
    assert events[:3] == ["ready_entered", "ready_lost", "ready_entered"]


class SwitchingRecognizer(FakeRecognizer):
    """Driver 6 for the first ``same`` attempts, then driver 7."""

    def __init__(self, same: int) -> None:
        super().__init__(cost=0.0)
        self.same = same

    def identify(self, frame):
        self.calls += 1
        did = 6 if self.calls <= self.same else 7
        return {"driver_id": did, "name": f"Driver {did}", "confidence": 0.6}


def test_ready_lost_on_driver_change(m, rig, hp, ign, known_driver, transitions, runs,
                                     monkeypatch, caplog):
    """A different enrolled driver: READY lost at once, then the cycle restarts for them."""
    monkeypatch.setattr(m.config, "PREDRIVE_REIDENTIFY_S", 0.5)
    monkeypatch.setattr(m, "show_frame", keys_after(2000, m.KEY_QUIT))     # safety net
    rec = SwitchingRecognizer(same=2)                 # checks at READY +0.5 / +1.0 s pass
    api = FakeAPI()
    am = rig(api, ign)
    caplog.set_level("INFO", logger="fatigue")
    released = m.run_predrive_assessment(api, FakeExtractor(), rec, am, hp, ign, m.LoopRate())
    assert released is False and am.relay_locked, "back to a fresh pre-drive, starter inhibited"
    seq = [(s, locked) for s, _note, locked, _t in transitions]
    i = seq.index((READY, False))
    assert seq[i + 1] == (ASSESSING, True)
    assert transitions[i + 1][1].startswith("confirming the driver (matched driver 7)")
    assert rec.calls == 2 + m.config.PREDRIVE_RECOGNITION_MATCHES
    assert "DRIVER CHANGED from 6 to 7" in caplog.text
    assert runs[0].timeline_summary()["end_reason"] == "driver_changed"


def test_under_minimum_never_ready(m, rig, hp, ign, known_driver, transitions, runs, monkeypatch):
    """Passing frames from the first one, but no READY until the minimum duration."""
    t0 = {}

    def show(frame):
        if m._session_state is ASSESSING:
            t0.setdefault("t", time.monotonic())
        if "t" in t0 and time.monotonic() - t0["t"] > 3.6:     # minimum is 4.0 s
            return m.KEY_QUIT
        return -1
    monkeypatch.setattr(m, "show_frame", show)
    api = FakeAPI()
    am = rig(api, ign)
    with pytest.raises(m.QuitRequested):
        m.run_predrive_assessment(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, ign,
                                  m.LoopRate())
    assert READY not in states(transitions) and am.relay_locked
    assert [e["event"] for e in runs[0].timeline] == ["ended"], "never READY"
    assert "post_assessment" not in api.calls, "not yet judged"
    # The same run, left alone, enters READY only once the minimum has passed.
    monkeypatch.setattr(m, "show_frame", Operator(m, (READY, 0.0, m.KEY_QUIT)))
    with pytest.raises(m.QuitRequested):
        m.run_predrive_assessment(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, ign,
                                  m.LoopRate())
    first = runs[1].timeline[0]
    assert first["event"] == "ready_entered" and first["t_rel"] >= runs[1].duration_s


def test_begin_made_before_ready_is_ignored(m, rig, hp, ign, known_driver, transitions,
                                            monkeypatch):
    """A tap from a stale button (made while ASSESSING) never confirms a later READY."""
    sent = {}

    def show(frame):
        if m._session_state is ASSESSING and "begin" not in sent:
            sent["begin"] = True
            return m.KEY_BEGIN
        if m._session_state is READY:
            sent.setdefault("ready", time.monotonic())
            if time.monotonic() - sent["ready"] > 1.0:
                return m.KEY_QUIT
        return -1
    monkeypatch.setattr(m, "show_frame", show)
    api = FakeAPI()
    am = rig(api, ign)
    with pytest.raises(m.QuitRequested):
        m.run_predrive_assessment(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, ign,
                                  m.LoopRate())
    assert states(transitions)[-1] is READY, "still waiting - the early tap did not count"


# ---- END SESSION (2026-10-19) -------------------------------------------------

def test_end_session_returns_to_idle(patched_main, monkeypatch, tmp_path):
    """
    From READY_TO_START and from MONITORING: starter inhibited, alerts off,
    IDLE, and the process keeps running - START NEXT DRIVER starts again.
    """
    m = patched_main
    op = Operator(m, (READY, 0.3, m.KEY_END_SESSION),       # end while READY
                  (IDLE, 0.5, m.KEY_BEGIN),                 # next driver
                  (READY, 0.3, m.KEY_BEGIN),                # start confirmed
                  (SessionState.MONITORING, 1.0, m.KEY_END_SESSION),
                  (IDLE, 0.5, m.KEY_QUIT))
    monkeypatch.setattr(m, "show_frame", op)
    state_file = tmp_path / "state.json"
    written = []
    real = m._write_state_file

    def spy():
        real()
        written.append(state_file.read_text(encoding="utf-8"))
    monkeypatch.setattr(m, "_write_state_file", spy)
    assert m.main(["--sequence", "--state-file", str(state_file)]) == 0
    assert op.steps == [], f"stopped early: {op.states()}"
    idle = [h for h in op.history if h["state"] is IDLE]
    assert len(idle) == 2
    for h in idle:
        assert h["relay_locked"] is True, "starter inhibited in IDLE"
        assert h["level"] is None
        assert all(v == 0 for pin, v in h["pins"].items()
                   if pin != m.config.RELAY_PIN), "LEDs and buzzer off"
    seq = op.states()
    first_idle, second_idle = [i for i, s in enumerate(seq) if s is IDLE]
    assert READY in seq[:first_idle] and SessionState.MONITORING not in seq[:first_idle]
    assert SessionState.IDENTIFYING in seq[first_idle:] or SessionState.ASSESSING in \
        seq[first_idle:], "a new pre-drive after START NEXT DRIVER"
    assert SessionState.MONITORING in seq[first_idle:second_idle]
    import json
    states_written = [json.loads(w)["state"] for w in written]
    assert states_written.count("IDLE") == 2 and "READY_TO_START" in states_written
    assert all(json.loads(w)["pid"] == __import__("os").getpid() for w in written)


def test_end_session_before_ready_closes_the_run(m, rig, hp, ign, known_driver, runs,
                                                 monkeypatch):
    """END SESSION mid-assessment: SessionEnded propagates, the run is closed and recorded."""
    monkeypatch.setattr(m, "show_frame", Operator(m, (ASSESSING, 1.0, m.KEY_END_SESSION)))
    api = FakeAPI()
    am = rig(api, ign)
    with pytest.raises(m.SessionEnded):
        m.run_predrive_assessment(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, ign,
                                  m.LoopRate())
    assert runs[0].timeline_summary()["end_reason"] == "session_ended"
    m.end_session(am)
    assert am.relay_locked and am.current_level is None


# ---- rolling window, without the loop -----------------------------------------

def _metrics(frs, t):
    """The FrameMetrics fields PredriveAssessment.add reads."""
    from types import SimpleNamespace
    return SimpleNamespace(
        t=t, frs=frs, level="ALERT", level_reason="BAND", frs_result={"components": {}},
        pose=None, ear=0.3, mar=0.4, perclos=5.0, blink_duration_ms=100.0, blink_freq=15.0,
        ear_norm=1.0, ear_norm_frs=1.0, bd_norm=1.0, bf_norm=1.0, perclos_norm=1.0,
        mar_norm=1.0, yawn_norm=0.0, pose_override=False, yawn_status={},
        microsleep_status={}, microsleep_override=False, microsleep_event=None)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rolling_worst_window_equals_the_fixed_computation(seed):
    """window_verdict's worst window == worst_window_mean() on the same frames."""
    rng = np.random.default_rng(seed)
    a = PredriveAssessment(duration_s=30.0, worst_window_s=5.0)
    a.start(1000.0)
    t, ts, fs = 1000.0, [], []
    for _ in range(2500):                         # ~100 s at an uneven ~25 fps
        t += float(rng.uniform(0.02, 0.07))
        f = float(rng.gamma(2.0, 0.08))
        a.add(_metrics(f, t), t)
        ts.append(t)
        fs.append(f)
        if rng.random() < 0.02:
            w = a.window_verdict(t)
            ref = PredriveAssessment(duration_s=30.0, worst_window_s=5.0)
            ref.samples = [{"t_abs": x, "frs_scored": y} for x, y in zip(ts, fs)
                           if x >= t - 30.0]
            assert w["worst_window_mean"] == ref.worst_window_mean()


def test_rolling_window_forgets_old_frames_and_needs_the_minimum():
    a = PredriveAssessment(duration_s=30.0, worst_window_s=5.0)
    a.start(0.0)
    t = 0.0
    while t < 29.0:
        t += 0.05
        a.add(_metrics(0.1, t), t)
    assert not a.window_verdict(t)["eligible"], "under 30 s: never eligible"
    while t < 40.0:                               # 5 s of high FRS, 10-15 s past the start
        t += 0.05
        a.add(_metrics(0.9 if 30.0 <= t < 35.0 else 0.1, t), t)
    w = a.window_verdict(t)
    assert w["eligible"] and not w["passed"] and w["reason"] == "frs"
    while t < 66.0:                               # the bad stretch has left the window
        t += 0.05
        a.add(_metrics(0.1, t), t)
    assert a.window_verdict(t)["passed"]
    for _ in range(200):                          # 10 s with no face: < 80 % valid
        t += 0.05
        a.add_invalid("no_face", t)
    w = a.window_verdict(t)
    assert not w["passed"] and w["reason"] == "invalid"
