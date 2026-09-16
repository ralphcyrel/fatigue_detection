"""
Module 11 - Ignition sense input.

Reads the vehicle's ignition state from a fuse tap on an ignition-switched
circuit, wired (through an optocoupler or a 3.3 V-clamped divider - never
directly) to ``config.IGNITION_PIN``. The state selects the operating phase:

    ignition OFF -> Phase.PREDRIVE     (starter may be inhibited / released)
    ignition ON  -> Phase.MONITORING   (alerts only; relay never touched)

Follows the same optional-hardware pattern as ``modules.alert``: ``RPi.GPIO``
is imported if available, otherwise (or with ``mock=True``) the sensor runs
in mock mode where the state is set programmatically - by ``--force-phase``
or by the ``i`` key in the preview window.

The raw pin is debounced in wall-clock time: a new reading must persist for
``config.IGNITION_DEBOUNCE_S`` before ``read()`` reports it, so contact
bounce on the tap cannot flap the phase back and forth.
"""

import logging
import time
from typing import Optional

from config import config
from modules.phase import Phase

logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO  # type: ignore

    GPIO_AVAILABLE: bool = True
except ImportError:
    GPIO = None  # type: ignore
    GPIO_AVAILABLE = False


class IgnitionSensor:
    """
    Debounced ignition-ON/OFF input.

    Typical usage::

        ignition = IgnitionSensor(mock=args.mock_gpio)
        if ignition.phase() is Phase.PREDRIVE:
            ...

    Attributes:
        mock: ``True`` when no hardware is being read.
        pin: BCM pin number of the sense input.
    """

    def __init__(
        self,
        mock: bool = False,
        pin: int = config.IGNITION_PIN,
        active_high: bool = config.IGNITION_ACTIVE_HIGH,
        debounce_s: float = config.IGNITION_DEBOUNCE_S,
    ) -> None:
        """
        Configure the input pin (or mock mode).

        Args:
            mock: Force mock mode even if ``RPi.GPIO`` is importable. If the
                library is missing, mock mode is used regardless.
            pin: BCM input pin.
            active_high: ``True`` if the pin reads HIGH when ignition is ON.
            debounce_s: Seconds a new raw reading must persist before it is
                reported.
        """
        self.mock: bool = mock or not GPIO_AVAILABLE
        self.pin: int = pin
        self.active_high: bool = active_high
        self.debounce_s: float = debounce_s

        self._mock_state: bool = False           # OFF -> pre-drive by default
        self._state: bool = False                # last *reported* state
        self._pending: Optional[bool] = None     # raw state waiting to settle
        self._pending_since: float = 0.0

        if self.mock:
            reason = "forced by caller" if mock else "RPi.GPIO not available"
            logger.info("IgnitionSensor running in MOCK mode (%s) - ignition OFF", reason)
        else:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            # Pull towards the OFF level so a disconnected tap reads OFF
            # (pre-drive, starter inhibited) rather than floating.
            pull = GPIO.PUD_DOWN if active_high else GPIO.PUD_UP
            GPIO.setup(pin, GPIO.IN, pull_up_down=pull)
            self._state = self._read_raw()
            logger.info("IgnitionSensor on BCM %d (active %s) - ignition %s",
                        pin, "HIGH" if active_high else "LOW", "ON" if self._state else "OFF")

    # ------------------------------------------------------------------

    def _read_raw(self) -> bool:
        """Undebounced ignition state."""
        if self.mock:
            return self._mock_state
        level = bool(GPIO.input(self.pin))
        return level if self.active_high else not level

    def read(self, now: Optional[float] = None) -> bool:
        """
        Debounced ignition state.

        Args:
            now: ``time.monotonic()`` reference; defaults to the clock.

        Returns:
            ``True`` if the ignition is ON.
        """
        t = time.monotonic() if now is None else now
        raw = self._read_raw()
        if raw == self._state:
            self._pending = None
            return self._state
        if self._pending != raw:
            self._pending = raw
            self._pending_since = t
            return self._state
        if t - self._pending_since >= self.debounce_s:
            self._state = raw
            self._pending = None
            logger.info("%sIgnition %s", "[MOCK] " if self.mock else "",
                        "ON" if raw else "OFF")
        return self._state

    def phase(self, now: Optional[float] = None) -> Phase:
        """Operating phase implied by the (debounced) ignition state."""
        return Phase.MONITORING if self.read(now) else Phase.PREDRIVE

    def set_mock_state(self, on: bool) -> None:
        """
        Set the raw ignition state in mock mode (goes through the debounce).

        Args:
            on: ``True`` for ignition ON.

        Raises:
            RuntimeError: if the sensor is reading real hardware.
        """
        if not self.mock:
            raise RuntimeError("set_mock_state() only works in mock mode")
        self._mock_state = on

    def cleanup(self) -> None:
        """Release the input pin (no-op in mock mode)."""
        if self.mock:
            return
        try:
            GPIO.cleanup(self.pin)
        except Exception as exc:  # pragma: no cover - hardware only
            logger.warning("GPIO.cleanup(%d) failed: %s", self.pin, exc)


class ForcedIgnition:
    """
    Drop-in for :class:`IgnitionSensor` that always reports one phase.

    Used by ``--force-phase`` so either phase can be exercised with no
    ignition hardware and no transitions.
    """

    mock = True

    def __init__(self, phase: Phase) -> None:
        self.forced_phase = Phase(phase)
        logger.info("Ignition FORCED to %s by --force-phase", self.forced_phase.value)

    def read(self, now: Optional[float] = None) -> bool:
        """``True`` if the forced phase is MONITORING."""
        return self.forced_phase is Phase.MONITORING

    def phase(self, now: Optional[float] = None) -> Phase:
        """The forced phase."""
        return self.forced_phase

    def set_mock_state(self, on: bool) -> None:
        """Ignored - the phase is pinned by the command line."""
        logger.info("Ignition toggle ignored: phase is forced to %s", self.forced_phase.value)

    def cleanup(self) -> None:
        """Nothing to release."""
