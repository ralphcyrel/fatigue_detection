"""
Starter-relay failsafe for processes that must not claim GPIO.

:func:`force_relay_inhibit` drives the relay pin to INHIBIT (HIGH) with the
``pinctrl`` command-line tool (``raspi-gpio`` on older images) instead of
``RPi.GPIO``. On Bookworm, ``RPi.GPIO`` is the rpi-lgpio shim, and an lgpio
line claim is exclusive to the process that holds it: a launcher that set the
pin up through ``RPi.GPIO`` kept it claimed, and every ``main.py`` child then
failed in ``AlertManager`` with ``lgpio.error: 'GPIO busy'``. ``pinctrl``
writes the pin once and exits, holding no claim.

This module deliberately never imports ``RPi.GPIO`` (``modules.alert`` does,
at import time), so the launcher imports it from here. ``modules.alert``
re-exports it.
"""

import logging
import shutil
import subprocess

from config.config import RELAY_PIN

logger = logging.getLogger(__name__)

# "op dh" = output, driven HIGH. HIGH is _RELAY_INHIBIT in modules/alert.py
# (low-level-trigger relay: HIGH releases the coil, starter inhibited).
_INHIBIT_ARGS = ("set", str(RELAY_PIN), "op", "dh")
_TOOLS = ("pinctrl", "raspi-gpio")
_TIMEOUT_S = 2


def force_relay_inhibit() -> bool:
    """
    Drive the relay pin to INHIBIT (HIGH) without claiming it.

    For the launcher, while no ``main.py`` child is running: at start-up,
    after a child exits and on quit. A child that was killed (SIGKILL) never
    ran ``AlertManager.cleanup()``, and the pin may be left at ALLOW. Never
    call it while a child runs: ``pinctrl`` writes the pin registers directly
    and would override the child's relay state underneath it. Touches the
    relay pin alone (never the LEDs or buzzer).

    Returns:
        ``True`` if the pin was driven, ``False`` if neither tool is present
        (not a Pi - logged at debug) or the command failed (logged).
    """
    tool = next((t for t in _TOOLS if shutil.which(t)), None)
    if tool is None:
        logger.debug("force_relay_inhibit: neither pinctrl nor raspi-gpio found - "
                     "not a Pi, nothing to drive")
        return False
    cmd = [tool, *_INHIBIT_ARGS]
    try:
        result = subprocess.run(cmd, check=False, timeout=_TIMEOUT_S,
                                capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("Could not drive relay pin %d to INHIBIT (%s): %s",
                     RELAY_PIN, " ".join(cmd), exc)
        return False
    if result.returncode != 0:
        logger.error("Could not drive relay pin %d to INHIBIT (%s): exit %d %s",
                     RELAY_PIN, " ".join(cmd), result.returncode,
                     (result.stderr or result.stdout or "").strip())
        return False
    logger.info("Relay pin %d driven to INHIBIT via %s (starter inhibited)", RELAY_PIN, tool)
    return True
