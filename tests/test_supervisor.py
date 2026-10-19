"""main() end to end per CLI mode: relay starts inhibited, monitoring never locks it."""

import logging

import pytest

from fakes import Operator, keys_after
from modules.phase import SessionState

pytestmark = pytest.mark.slow

READY, IDLE, MONITORING = (SessionState.READY_TO_START, SessionState.IDLE,
                           SessionState.MONITORING)


@pytest.mark.parametrize("argv, steps", [
    (["--sequence"], [(READY, 0.3, "b"), (MONITORING, 1.0, "q")]),
    (["--force-phase", "predrive"], [(READY, 0.3, "b"), (IDLE, 0.5, "q")]),
    (["--force-phase", "monitoring"], None),
    # No flags = no ignition input: the same session as --sequence.
    (["--mock-gpio"], [(READY, 0.3, "b"), (MONITORING, 1.0, "q")]),
])
def test_supervisor_modes(patched_main, monkeypatch, caplog, argv, steps):
    m = patched_main
    caplog.set_level(logging.INFO)
    if steps is None:
        monkeypatch.setattr(m, "show_frame", keys_after(60, m.KEY_QUIT))
    else:
        op = Operator(m, *[(state, delay, ord(key)) for state, delay, key in steps])
        monkeypatch.setattr(m, "show_frame", op)
    m.main(argv)
    lines = [r.getMessage() for r in caplog.records]
    mon = [line for line in lines if "=== MONITORING" in line]
    pre = [line for line in lines if "=== PRE-DRIVE" in line]
    assert any("Relay starts INHIBITED" in line for line in lines)
    assert not any("Relay lock REFUSED" in line for line in lines), \
        "no relay actuation attempted from monitoring"
    if steps is not None:
        assert op.steps == [], f"stopped early: {op.states()}"
    if argv[0] in ("--sequence", "--mock-gpio"):
        assert len(mon) == 1 and "entered from pre-drive release) - starter UNLOCKED" in mon[0]
        assert any("START CONFIRMED by key 'b'" in line for line in lines)
    elif argv[-1] == "predrive":
        assert not mon, "isolated pre-drive never enters monitoring"
        assert op.states()[-1] is IDLE
    else:
        assert len(mon) == 1 and "starter LOCKED" in mon[0] and not pre


def test_gpio_start_confirmation_is_not_implemented(patched_main, monkeypatch):
    """START_CONFIRM_MODE = 'gpio' stops main() at startup, before any hardware."""
    m = patched_main
    monkeypatch.setattr(m.config, "START_CONFIRM_MODE", "gpio")
    created = []
    monkeypatch.setattr(m, "AlertManager", lambda *a, **k: created.append(1))
    with pytest.raises(NotImplementedError, match="ignition-position sense"):
        m.main(["--sequence"])
    assert created == [], "raised before the GPIO was touched"
