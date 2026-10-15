"""Yawn detector hysteresis (2026-10-15): entry 2.0x, release 1.6x, 1.5 s."""

import numpy as np
import pytest

from fakes import synthetic_landmarks
from modules.mar import YawnDetector
from modules.pipeline import MetricsPipeline

BASE = 0.64
THR = 2.0 * BASE
DT = 0.05


def feed(det, ratios, t0=100.0):
    events, t = [], t0
    for r in ratios:
        t += DT
        ev = det.update(r * BASE, THR, t)
        if ev:
            events.append(round(t - t0, 2))
    return events, t


def test_wobble_around_the_entry_threshold_is_one_yawn():
    det = YawnDetector()
    wobble = [2.1, 1.9, 1.8, 2.05, 1.85, 2.2, 1.7, 1.95] * 10       # 4 s at 1.7-2.2x
    events, _ = feed(det, [1.0] * 10 + wobble)
    assert len(events) == 1 and events[0] == pytest.approx(0.55 + 1.5, abs=0.06)
    assert det.is_yawning


def test_peak_below_entry_never_opens():
    det = YawnDetector()
    events, _ = feed(det, [1.0] * 10 + [1.83, 1.83] + [1.7] * 60)
    assert events == [] and not det.mouth_open


def test_release_threshold_ends_the_opening():
    det = YawnDetector()
    events, _ = feed(det, [2.1] * 20 + [1.59] + [2.1] * 20)          # 1.0 s, dip, 1.0 s
    assert events == []


def test_talking_never_confirms():
    det = YawnDetector()
    events, _ = feed(det, ([2.1] * 6 + [1.2] * 6) * 20)
    assert events == []


def test_abandon_ends_the_opening():
    det = YawnDetector()
    _, t = feed(det, [2.1] * 20)
    assert det.abandon() == {"open_s": pytest.approx(0.95), "was_yawning": False}
    events, _ = feed(det, [2.1] * 20, t0=t)
    assert events == []


def test_release_defaults_to_the_configured_ratio():
    det = YawnDetector()
    feed(det, [2.1] * 4)
    det.update(1.61 * BASE, THR, 200.0)
    assert det.mouth_open
    det.update(1.59 * BASE, THR, 200.05)
    assert not det.mouth_open


def _pipeline_with_mar(hp):
    p = MetricsPipeline(hp, blink_window_s=30)
    cur = {}
    p.mar_calc.compute_mar = lambda lm: cur["mar"]
    return p, cur


TH = {"ear_baseline": 0.30, "ear_threshold": 0.225, "blink_duration_baseline": 120.0,
      "blink_frequency_baseline": 15.0, "perclos_baseline": 2.0, "mar_baseline": BASE,
      "yawn_threshold": THR}


def test_pipeline_stall_and_no_face_end_the_opening(hp):
    lm = synthetic_landmarks(np.random.default_rng(0))
    p, cur = _pipeline_with_mar(hp)
    t = 1000.0
    for _ in range(20):                                  # 1.0 s at 2.1x
        t += DT
        cur["mar"] = 2.1 * BASE
        p.process(lm, TH, t, t)
    t += 0.6                                             # stall > MAX_FRAME_GAP_S
    ms = []
    for _ in range(20):
        t += DT
        ms.append(p.process(lm, TH, t, t))
    assert not any(m.yawn_event for m in ms), "the stall split the opening"
    p.note_no_face(t + DT, t + DT)
    assert not p.yawn_detector.mouth_open
    for _ in range(40):                                  # 2.0 s: now a yawn
        t += DT
        ms.append(p.process(lm, TH, t, t))
    assert sum(1 for m in ms if m.yawn_event) == 1
    assert ms[-1].yawn_norm == pytest.approx(2.1)
