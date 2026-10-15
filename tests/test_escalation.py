"""
False-DANGER fixes (2026-10-15): escalation timing, landmark validity, stall
guard, closures across gaps, microsleep routing, scoring floors / warm-up,
head pose relative to neutral, invalid assessments. Fast (no real-time loops).
"""

import numpy as np
import pytest

from fakes import close_eyes, synthetic_landmarks
from modules.assessment import PredriveAssessment
from modules.head_pose import HeadPoseEstimator
from modules.pipeline import LevelEscalator, MetricsPipeline, landmark_problem

# The recorded driver-6 calibration: the tiny baselines that made one
# ordinary blink score DANGER.
TINY = {"ear_baseline": 0.30, "ear_threshold": 0.225, "blink_duration_baseline": 70.7,
        "blink_frequency_baseline": 3.0, "perclos_baseline": 0.5, "mar_baseline": 0.4,
        "yawn_threshold": 0.9}
DT = 0.05  # 20 fps, as on the Pi


def run(pipeline, script, t0=1000.0, thresholds=TINY):
    """``script``: list of (dt, closed); returns the FrameMetrics list."""
    rng = np.random.default_rng(1)
    t, out = t0, []
    for dt, closed in script:
        t += dt
        lm = synthetic_landmarks(rng)
        out.append(pipeline.process(close_eyes(lm) if closed else lm, thresholds, t, t))
    return out


def open_s(seconds):
    return [(DT, False)] * int(round(seconds / DT))


def closed_s(seconds):
    return [(DT, True)] * int(round(seconds / DT))


# ---- LevelEscalator ----------------------------------------------------------

def feed(esc, frs, seconds, t, signals=()):
    levels = []
    for _ in range(int(round(seconds / DT))):
        t += DT
        levels.append((round(t, 3), esc.update(frs, t, signals)))
    return levels, t


def first(levels, level):
    return next((t for t, lv in levels if lv == level), None)


def test_no_direct_alert_to_danger():
    esc = LevelEscalator(1.0, 3.0, 2.0)
    levels, t = feed(esc, 1.5, 6.0, 0.0)
    assert first(levels, "WARNING") == pytest.approx(1.05, abs=0.06)
    assert first(levels, "DANGER") == pytest.approx(4.05, abs=0.11), "1 s to WARNING + 3 s in it"
    assert esc.reason == "BAND"


def test_short_spike_stays_alert():
    esc = LevelEscalator(1.0, 3.0, 2.0)
    levels, t = feed(esc, 2.0, 0.9, 0.0)
    levels2, t = feed(esc, 0.0, 3.0, t)
    assert {lv for _, lv in levels + levels2} == {"ALERT"}


def test_release_waits_and_runs_concurrently():
    esc = LevelEscalator(1.0, 3.0, 2.0)
    _, t = feed(esc, 1.0, 5.0, 0.0)
    assert esc.level == "DANGER"
    levels, t = feed(esc, 0.0, 3.0, t)
    assert first(levels, "WARNING") is None, "collapse goes straight to ALERT"
    assert first(levels, "ALERT") - 5.0 == pytest.approx(2.0, abs=0.11)


def test_danger_needs_continuous_condition_in_warning():
    esc = LevelEscalator(1.0, 3.0, 2.0)
    _, t = feed(esc, 0.7, 2.5, 0.0)          # WARNING at 1.0, 1.5 s of DANGER condition
    _, t = feed(esc, 0.5, 0.2, t)            # dips under 0.65: timer restarts
    levels, t = feed(esc, 0.7, 2.9, t)
    assert first(levels, "DANGER") is None
    levels, t = feed(esc, 0.7, 0.3, t)
    assert first(levels, "DANGER") is not None


def test_head_pose_signal_follows_the_three_second_rule():
    esc = LevelEscalator(1.0, 3.0, 2.0)
    levels, t = feed(esc, 0.0, 4.0, 0.0, ("HEAD_POSE",))
    assert levels[0][1] == "WARNING", "an engaged pose alert enters WARNING at once"
    assert first(levels, "DANGER") == pytest.approx(3.05, abs=0.11)
    assert esc.reason == "HEAD_POSE"


# ---- landmark validity ---------------------------------------------------------

def test_landmark_problem():
    lm = synthetic_landmarks(np.random.default_rng(0))
    assert landmark_problem(lm, (480, 640, 3)) is None
    assert landmark_problem(None, (480, 640, 3)) == "no_face"
    off = lm.copy()
    off[36:42, 0] -= 300                       # left eye off the image
    assert landmark_problem(off, (480, 640, 3)) == "outside_frame"
    flat = lm.copy()
    flat[42:48] = flat[42]                     # right eye collapsed to a point
    assert landmark_problem(flat, (480, 640, 3)) == "right_eye_degenerate"
    shut = lm.copy()
    shut[41], shut[40] = shut[37], shut[38]    # each lid pair on one point: EAR 0
    assert landmark_problem(shut, (480, 640, 3)) == "left_eye_ear_zero"


# ---- single blink with tiny baselines ---------------------------------------------

def test_single_blink_no_longer_escalates(hp):
    p = MetricsPipeline(hp, blink_window_s=30)
    ms = run(p, open_s(3.0) + closed_s(0.25) + open_s(5.0))
    assert {m.level for m in ms} == {"ALERT"}
    assert max(m.frs for m in ms) < 0.40
    assert p.scoring_thresholds(TINY)["perclos_baseline"] == 2.0


def test_blink_terms_warm_up(hp):
    p = MetricsPipeline(hp, blink_window_s=30)
    # Two slow blinks in the first 5 s: < 3 blinks -> no duration term;
    # < 10 s -> no frequency term.
    ms = run(p, open_s(1.0) + closed_s(0.6) + open_s(1.0) + closed_s(0.6) + open_s(1.0))
    assert all(m.bd_norm == 1.0 and m.bf_norm == 1.0 for m in ms)


# ---- closures across gaps ------------------------------------------------------------

def test_stall_never_becomes_a_microsleep_or_blink(hp):
    p = MetricsPipeline(hp, blink_window_s=30)
    ms = run(p, open_s(3.0) + [(DT, True), (1.05, False)] + open_s(1.0))
    assert not any(m.microsleep_event for m in ms)
    assert p.microsleep_detector.get_microsleep_count() == 0
    assert p.blink_detector.blink_durations == [], "the stall is not counted as closure time"


def test_closure_crossing_one_second_between_frames_is_a_slow_blink(hp):
    p = MetricsPipeline(hp, blink_window_s=30)
    # Observed shut 0.95 s, reopening frame 0.3 s later (under MAX_FRAME_GAP_S).
    run(p, open_s(2.0) + closed_s(1.0) + [(0.3, False)] + open_s(0.5))
    assert p.microsleep_detector.get_microsleep_count() == 0
    assert len(p.blink_detector.blink_durations) == 1


def test_no_face_abandons_a_blink_in_progress(hp):
    p = MetricsPipeline(hp, blink_window_s=30)
    run(p, open_s(2.0) + closed_s(0.2))
    p.note_no_face(1002.3, 1002.3)
    run(p, open_s(1.0), t0=1002.7)
    assert p.blink_detector.blink_durations == []


# ---- microsleep routing ----------------------------------------------------------------

def test_microsleep_forces_danger_by_default(hp):
    p = MetricsPipeline(hp, blink_window_s=30, microsleep_requires_warning=False)
    ms = run(p, open_s(3.0) + closed_s(1.3))
    hit = next(m for m in ms if m.microsleep_event)
    assert hit.level == "DANGER" and hit.level_reason == "MICROSLEEP"


def test_microsleep_can_be_routed_through_warning(hp):
    p = MetricsPipeline(hp, blink_window_s=30, microsleep_requires_warning=True)
    ms = run(p, open_s(3.0) + closed_s(1.1) + closed_s(4.0))
    i = next(i for i, m in enumerate(ms) if m.microsleep_event)
    assert ms[i].level == "WARNING"
    assert "MICROSLEEP" in ms[i].level_reason
    assert any(m.level == "DANGER" for m in ms[i:]), "DANGER after 3 s in WARNING"
    assert not any(m.level == "DANGER" for m in ms[i:i + int(2.9 / DT)])


# ---- head pose relative to neutral --------------------------------------------------------

def test_pose_flags_are_relative_to_neutral():
    hp = HeadPoseEstimator(640, 480)
    neutral = {"pitch": 25.0, "yaw": 0.0, "roll": 0.0}
    assert not hp.flag({"pitch": 25.0, "yaw": 0.0, "roll": 0.0}, neutral)["alert"]
    assert not hp.flag({"pitch": 11.0, "yaw": 0.0, "roll": 0.0}, neutral)["alert"]
    assert hp.flag({"pitch": 10.0, "yaw": 0.0, "roll": 0.0}, neutral)["is_nodding"]
    assert hp.flag({"pitch": 25.0, "yaw": 31.0, "roll": 0.0}, neutral)["is_looking_away"]
    assert not hp.flag({"pitch": -40.0, "yaw": 0.0, "roll": 0.0}, None)["alert"], \
        "no neutral -> nothing flagged"


def test_monitoring_seeds_a_neutral_pose(hp):
    p = MetricsPipeline(hp, blink_window_s=60)
    run(p, open_s(3.5))
    assert p.pose_neutral is not None and p.pose_neutral_source == "seeded"


# ---- invalid assessment -----------------------------------------------------------------

def test_invalid_assessment_below_valid_fraction(hp):
    p = MetricsPipeline(hp, blink_window_s=30)
    a = PredriveAssessment(duration_s=5.0, worst_window_s=2.0)
    a.start(1000.0)
    for m in run(p, open_s(3.0)):
        a.add(m)
    for _ in range(20):                       # 60 valid, 20 invalid -> 75 %
        a.add_invalid("outside_frame")
    r = a.result()
    assert r["void"] and not r["passed"]
    assert r["invalid_cause"] == "INVALID_ASSESSMENT"
    assert r["valid_fraction"] == pytest.approx(0.75)
    assert r["invalid_reasons"] == {"outside_frame": 20}


# ---- settling: microsleep detection only ------------------------------------------

def settle_run(watch, script, t0=1000.0):
    """``script``: (dt, state) with state "open" / "closed" / None (invalid)."""
    rng = np.random.default_rng(2)
    t, events = t0, []
    for dt, state in script:
        t += dt
        if state is None:
            watch.invalid(t)
            continue
        lm = synthetic_landmarks(rng)
        ev = watch.valid(close_eyes(lm) if state == "closed" else lm, t, t)
        if ev:
            events.append(round(t - t0, 2))
    return events, t


def test_settling_detects_a_microsleep():
    from modules.pipeline import SettlingWatch
    w = SettlingWatch(3.0, 0.225)
    events, t = settle_run(w, [(DT, "open")] * 30 + [(DT, "closed")] * 30)
    assert len(events) == 1 and events[0] == pytest.approx(2.55, abs=0.06)  # shut at 1.55 s + 1.0 s


def test_settling_microsleep_respects_invalid_frames_and_stalls():
    from modules.pipeline import SettlingWatch
    w = SettlingWatch(3.0, 0.225)
    # 0.6 s shut, one invalid frame, 0.6 s shut: two closures, no microsleep.
    events, _ = settle_run(w, [(DT, "closed")] * 12 + [(DT, None)] + [(DT, "closed")] * 12)
    assert events == []
    w = SettlingWatch(3.0, 0.225)
    # 0.6 s shut, a 0.6 s stall, still shut 0.6 s: the stall is unobserved.
    events, _ = settle_run(w, [(DT, "closed")] * 12 + [(0.6, "closed")] + [(DT, "closed")] * 11)
    assert events == []


def test_settling_waits_for_an_open_eye_and_restarts_on_invalid():
    from modules.pipeline import SettlingWatch
    w = SettlingWatch(3.0, 0.225)
    _, t = settle_run(w, [(DT, "open")] * 50 + [(DT, None)] + [(DT, "open")] * 50)
    assert w.restarts == 1 and not w.settled(t), "2.5 s since the restart"
    _, t = settle_run(w, [(DT, "open")] * 9 + [(DT, "closed")] * 3, t0=t)
    assert w.held(t) >= 3.0 and not w.settled(t), "never hand over mid-closure"
    _, t = settle_run(w, [(DT, "open")], t0=t)
    assert w.settled(t)


# ---- PERCLOS anchored to an absolute reference -------------------------------------

@pytest.mark.parametrize("perclos, baseline, excess", [
    (9.088, 2.0, 0.0),      # the Pi run: ordinary blinking, 2 % floor -> no term
    (20.0, 2.0, 1.0),
    (40.0, 2.0, 3.0),
    (60.0, 2.0, 4.0),       # capped
    (18.0, 12.0, 0.5),      # a baseline above the reference is the anchor
])
def test_perclos_excess_is_anchored(perclos, baseline, excess):
    from modules.frs import FRSCalculator
    from modules.perclos import PERCLOSCalculator
    t = MetricsPipeline.perclos_reference(baseline)
    norm = PERCLOSCalculator().normalize(perclos, t)
    r = FRSCalculator().compute(1.0, 1.0, 1.0, norm)
    assert r["excess"]["perclos_excess"] == pytest.approx(excess, abs=1e-3)


def test_theoretical_max_unchanged():
    from modules.frs import FRSCalculator
    m = FRSCalculator().theoretical_max()
    assert m["bounded"] == pytest.approx(5.05) and m["unbounded_terms"] == []


# ---- no-face: latch for NO_FACE_FAULT_S, then FAULT ----------------------------------

def gap(pipeline, t0, seconds):
    verdicts, t = [], t0
    for _ in range(int(round(seconds / DT))):
        t += DT
        verdicts.append(pipeline.note_no_face(t, t))
    return verdicts, t


def test_danger_latch_becomes_fault_after_two_seconds(hp):
    p = MetricsPipeline(hp, blink_window_s=60, no_face_policy=True)
    ms = run(p, open_s(3.0) + closed_s(1.3))
    assert ms[-1].level == "DANGER"
    t = ms[-1].t
    verdicts, t = gap(p, t, 3.0)
    latched = [v for v in verdicts if v.gap_s < 1.99]
    assert latched and all(v.band == "LATCH" and v.level == "DANGER" for v in latched)
    faults = [v for v in verdicts if v.gap_s > 2.01]
    assert faults and all(v.band == "FAULT" and v.level == "FAULT" for v in faults)
    assert [v.event for v in verdicts if v.event] == ["latch_lost"], "one backend fault"
    back = run(p, open_s(0.5), t0=t)
    assert not back[0].no_face_latch, "the latch is dropped"
    assert {m.level for m in back} == {"ALERT"}, "escalation restarts from the score"


def test_short_gap_keeps_the_existing_latch(hp):
    p = MetricsPipeline(hp, blink_window_s=60, no_face_policy=True)
    ms = run(p, open_s(3.0) + closed_s(1.3))
    verdicts, t = gap(p, ms[-1].t, 1.5)
    assert {v.band for v in verdicts} == {"LATCH"}
    back = run(p, open_s(2.0), t0=t)
    assert all(m.level == "DANGER" and m.no_face_latch for m in back), \
        "held until 3 s below DANGER"


def test_hold_from_warning_faults_at_two_seconds(hp):
    p = MetricsPipeline(hp, blink_window_s=60, no_face_policy=True)
    run(p, open_s(3.0))
    p.last_level = "WARNING"
    verdicts, _ = gap(p, 1003.0, 2.5)
    assert {v.band for v in verdicts if v.gap_s < 1.99} == {"HOLD"}
    assert {v.level for v in verdicts if v.gap_s < 1.99} == {"WARNING"}
    assert [v.event for v in verdicts if v.event] == ["fault_entered"]
    assert verdicts[-1].level == "FAULT"


def test_blink_frequency_is_anchored(hp):
    assert MetricsPipeline.blink_freq_reference(3.0) == 20.0
    assert MetricsPipeline.blink_freq_reference(24.0) == 24.0
    p = MetricsPipeline(hp, blink_window_s=60)
    # ~30 blinks/min for 20 s: against the 6 /min floor this was at the cap.
    ms = run(p, open_s(1.0) + (closed_s(0.15) + open_s(1.85)) * 10)
    m, elapsed = ms[-1], ms[-1].t - ms[0].t
    assert m.blink_freq >= 9
    assert m.bf_norm == pytest.approx(m.blink_freq / (20.0 * elapsed / 60.0), rel=0.01)
    assert m.frs_result["excess"]["blink_frequency_excess"] < 1.0
