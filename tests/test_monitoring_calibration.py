"""
Monitoring start never waits for the backend (2026-10-19): the calibration
GET runs off the frame loop, the loop keeps scoring on what it has, and a
dead backend only gets logged.
"""

import logging

import pytest

from fakes import (FakeRecognizer, ScriptedIgnition, SlowCalibrationAPI, TimedFace,
                   monitoring_summaries)
from modules.phase import Phase

pytestmark = pytest.mark.slow


def closure_at(start, length):
    """Eyes closed from ``start`` for ``length`` s, open otherwise."""
    return lambda t: "closed" if start <= t < start + length else "open"


def test_slow_backend_at_monitoring_start(m, rig, hp, tmp_path, caplog):
    """
    The calibration GET takes 5 s. The frame loop never pauses past
    MAX_FRAME_GAP_S, a 1.1 s closure (>= the 1.0 s microsleep minimum) inside
    those 5 s is confirmed on the self-seeded baseline, and the calibration is
    applied when it arrives.
    """
    api = SlowCalibrationAPI(delay=5.0)
    am = rig(api, ScriptedIgnition([(8.0, Phase.MONITORING), (99, Phase.PREDRIVE)]),
             Phase.MONITORING)
    caplog.set_level(logging.INFO)
    m.run_monitoring(api, TimedFace(closure_at(2.5, 1.1)), FakeRecognizer(0.0), am, hp,
                     m._ignition, m.LoopRate())
    (rec,) = monitoring_summaries(tmp_path)
    assert rec["max_frame_gap_s"] < m.config.MAX_FRAME_GAP_S, rec
    assert rec["frame_gaps_over_limit"] == 0
    assert "MICROSLEEP confirmed" in caplog.text, "detected while the GET was outstanding"
    assert rec["driver_id"] == 6 and rec["identified_after_s"] < 1.0
    assert rec["calibration_source"].startswith("fetched in 5.")
    assert rec["calibration_after_s"] >= 5.0
    assert api.calls.count("get_driver_calibration") == 1
    print(f"\nworst frame gap, 5 s calibration GET at monitoring start: "
          f"{rec['max_frame_gap_s'] * 1000:.0f} ms")


def test_unreachable_backend_keeps_monitoring_running(m, rig, hp, tmp_path, caplog):
    """The GET gives up after 5 s with nothing: logged, monitoring carries on unchanged."""
    api = SlowCalibrationAPI(delay=5.0, calibration=None)
    am = rig(api, ScriptedIgnition([(7.0, Phase.MONITORING), (99, Phase.PREDRIVE)]),
             Phase.MONITORING)
    caplog.set_level(logging.INFO, logger="fatigue")
    m.run_monitoring(api, TimedFace(closure_at(99, 0)), FakeRecognizer(0.0), am, hp,
                     m._ignition, m.LoopRate())
    (rec,) = monitoring_summaries(tmp_path)
    assert rec["max_frame_gap_s"] < m.config.MAX_FRAME_GAP_S
    assert "no usable calibration for driver 6 after 5." in caplog.text
    assert "keeps running on the self-seeded EAR baseline" in caplog.text
    assert rec["calibration_source"] == "none (fetch failed)"
    assert rec["end_reason"] == "ignition_off" and rec["frames"] > 100
