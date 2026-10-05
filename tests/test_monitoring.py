"""Monitoring loop: profiler, recognition cadence, from_release exit semantics."""

import logging
import time

import pytest

from fakes import FakeAPI, FakeExtractor, FakeRecognizer, ScriptedIgnition
from modules.phase import Phase

pytestmark = pytest.mark.slow


def test_profiler_and_recognition_once(m, rig, hp, tmp_path):
    """Recognised driver: identify() runs once; the profiler writes its CSV and clears."""
    api = FakeAPI()
    am = rig(api, ScriptedIgnition([(7.0, Phase.MONITORING), (99, Phase.PREDRIVE)]),
             Phase.MONITORING)
    rec = FakeRecognizer(cost=0.6)
    t = time.monotonic()
    m.run_monitoring(api, FakeExtractor(), rec, am, hp, m._ignition, m.LoopRate(), profile_s=3.0)
    assert 6.5 < time.monotonic() - t < 8.5, "returns when the ignition goes OFF"
    csvs = sorted(tmp_path.glob("loop_profile_*.csv"))
    assert csvs and len(csvs[-1].read_text().splitlines()) > 20, "per-frame CSV written"
    assert m._profiler is None
    assert rec.calls == 1


def test_unrecognised_driver_retried_every_5s(m, rig, hp):
    api = FakeAPI()
    am = rig(api, ScriptedIgnition([(12.0, Phase.MONITORING), (99, Phase.PREDRIVE)]),
             Phase.MONITORING)
    rec = FakeRecognizer(cost=0.0, driver_id=None)
    rate = m.LoopRate()
    m.run_monitoring(api, FakeExtractor(), rec, am, hp, m._ignition, rate)
    assert rec.calls == 3, "attempts at t = 0 / 5 / 10 s"
    assert rate._count > 100
    assert "get_driver_calibration" not in api.calls, "no calibration silently accepted"


class LateRecognizer(FakeRecognizer):
    """No match for the first ``misses`` attempts, then driver 6."""

    def __init__(self, misses: int) -> None:
        super().__init__(cost=0.0)
        self.misses = misses

    def identify(self, frame):
        self.calls += 1
        if self.calls <= self.misses:
            return None
        return {"driver_id": 6, "name": "Test Driver", "confidence": 0.6}


def test_late_match_then_never_again(m, rig, hp, caplog):
    api = FakeAPI()
    am = rig(api, ScriptedIgnition([(16.0, Phase.MONITORING), (99, Phase.PREDRIVE)]),
             Phase.MONITORING)
    rec = LateRecognizer(misses=2)
    caplog.set_level(logging.INFO, logger="fatigue")
    m.run_monitoring(api, FakeExtractor(), rec, am, hp, m._ignition, m.LoopRate())
    lines = [(r.created, r.getMessage()) for r in caplog.records if r.name == "fatigue"]
    ident = [(t, msg) for t, msg in lines if "Driver identified" in msg]
    tries = [t for t, msg in lines if "recognition attempt" in msg or "driver not recognised" in msg]
    assert rec.calls == 3 and len(ident) == 1 and "attempt 3" in ident[0][1]
    gaps = [b - a for a, b in zip(tries, tries[1:] + [ident[0][0]])]
    assert all(4.8 < g < 5.6 for g in gaps), f"attempts spaced ~5 s: {gaps}"
    assert api.calls.count("get_driver_calibration") == 1, "fetched once, after the match"


def test_from_release_waits_for_on_then_off(m, rig, hp):
    api = FakeAPI()
    am = rig(api, ScriptedIgnition([(2.0, Phase.PREDRIVE), (4.0, Phase.MONITORING),
                                    (99, Phase.PREDRIVE)]))
    am.unlock_relay()
    t = time.monotonic()
    m.run_monitoring(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, m._ignition,
                     m.LoopRate(), from_release=True)
    assert 3.7 < time.monotonic() - t < 5.0, "ran through the OFF lead-in, exited at ON -> OFF"
    assert not am.relay_locked, "monitoring never actuates the relay"


def test_ignition_off_returns_immediately(m, rig, hp):
    api = FakeAPI()
    am = rig(api, ScriptedIgnition([(99, Phase.PREDRIVE)]))
    t = time.monotonic()
    m.run_monitoring(api, FakeExtractor(), FakeRecognizer(0.0), am, hp, m._ignition, m.LoopRate())
    assert time.monotonic() - t < 0.3
