"""
Module 8 - Physical alerts and relay control.

`AlertManager` is the only module that drives outputs on the Raspberry Pi's
GPIO header. It controls:

* three status LEDs (green / yellow / red) mirroring the FRS alert level,
* a piezo buzzer with short, long or continuous patterns,
* the **starter-inhibit relay**.

The alert level (LEDs + buzzer) and the relay are deliberately independent.
The level follows the fatigue metrics in every phase; the relay is driven
only by the pre-drive verdict:

* ``Phase.PREDRIVE`` (ignition OFF) - the relay starts *inhibited* and is
  released only by ``unlock_relay()`` after a passed assessment or an
  approved operator override.
* ``Phase.MONITORING`` (ignition ON) - ``lock_relay()`` is a no-op that logs
  a WARNING. Nothing in this phase, including a head-pose DANGER, can engage
  the relay. This is enforced here rather than by convention in ``main.py``
  so a future edit cannot reintroduce an engine-interrupt-while-driving bug.
  Entering monitoring does **not** release the relay either: the ignition
  sense is a fuse tap that goes live at key-ON, *before* cranking, so an
  automatic release here would let an unassessed driver start the engine
  simply by turning the key. The relay keeps whatever state pre-drive left
  it in; only a passed assessment or an approved override releases it.

Because development happens on machines without GPIO (Windows / macOS), the
``RPi.GPIO`` import is optional. When it is missing - or when ``mock=True`` is
requested - the manager runs in *mock mode*: every actuation is logged instead
of toggling hardware, and the internal state (which LED is "on", whether the
relay is locked) is still tracked so the rest of the pipeline behaves
identically.

Relay wiring: the relay sits in the **starter solenoid signal** (it can only
prevent a start, never stop a running engine) and is wired **normally-open**
for fail-secure behaviour. The coil must be energised (pin HIGH) to close the
contacts and *allow* a start; de-energised (pin LOW, Pi unpowered, process
crashed, ``GPIO.cleanup()`` run) the starter is inhibited. Swap
``_RELAY_ALLOW`` / ``_RELAY_INHIBIT`` below if your relay board is active-low.
"""

import logging
import threading
import time
from typing import Dict, Optional

from config.config import BUZZER_PIN, LED_GREEN, LED_RED, LED_YELLOW, RELAY_PIN
from modules.phase import Phase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional hardware import. On a non-Pi host this fails and we fall back to
# mock mode; the module-level flag lets callers (and tests) inspect this.
# ---------------------------------------------------------------------------
try:
    import RPi.GPIO as GPIO  # type: ignore

    GPIO_AVAILABLE: bool = True
except ImportError:
    GPIO = None  # type: ignore
    GPIO_AVAILABLE = False

# Buzzer on-times in seconds for the named patterns.
BUZZER_PATTERNS: Dict[str, float] = {
    "short": 0.1,
    "long": 0.5,
}

# Logic levels for the relay (see module docstring for the wiring assumption).
_RELAY_ALLOW: int = 1    # HIGH -> coil energised -> NO contacts closed -> start allowed
_RELAY_INHIBIT: int = 0  # LOW  -> coil released  -> contacts open       -> start inhibited

# All output pins, in a stable order, for setup and cleanup.
_ALL_PINS = (LED_GREEN, LED_YELLOW, LED_RED, BUZZER_PIN, RELAY_PIN)


class AlertManager:
    """
    Drive LEDs and buzzer from the FRS alert level; drive the starter relay
    from the pre-drive verdict.

    Typical usage::

        alerts = AlertManager(phase=Phase.PREDRIVE)   # auto-detects GPIO; relay inhibited
        alerts.set_alert_level("WARNING")             # yellow LED + short beep
        alerts.set_alert_level("DANGER")              # red LED + buzzer (relay untouched)
        alerts.unlock_relay()                         # assessment passed / override approved
        alerts.set_phase(Phase.MONITORING)            # ignition ON: lock_relay() now refused,
                                                      # relay state carried over unchanged
        alerts.cleanup()                              # on shutdown

    Attributes:
        mock: ``True`` when no hardware is being driven.
        phase: Current operating phase; gates :meth:`lock_relay`.
        current_level: Last level passed to :meth:`set_alert_level`.
        relay_locked: ``True`` while the starter is inhibited.
        pin_states: Last logic level written to each pin (mock and real).
    """

    def __init__(self, mock: bool = False, phase: Phase = Phase.PREDRIVE) -> None:
        """
        Configure GPIO (or mock mode) and put every output in its safe state.

        Args:
            mock: Force mock mode even if ``RPi.GPIO`` is importable. If the
                library is missing, mock mode is used regardless.
            phase: Initial operating phase (gates :meth:`lock_relay`). The
                relay starts inhibited in either phase (fail-secure): if the
                process starts with the engine already running, an inhibited
                starter is harmless.
        """
        self.mock: bool = mock or not GPIO_AVAILABLE
        self.phase: Phase = Phase(phase)
        self.current_level: Optional[str] = None
        self.relay_locked: bool = False
        self.pin_states: Dict[int, int] = {pin: 0 for pin in _ALL_PINS}

        # Continuous-buzzer thread control. ``_buzzer_stop`` is set to halt
        # the background thread; ``_buzzer_lock`` serialises start/stop so two
        # frames can't race each other.
        self._buzzer_thread: Optional[threading.Thread] = None
        self._buzzer_stop = threading.Event()
        self._buzzer_lock = threading.Lock()

        if self.mock:
            reason = "forced by caller" if mock else "RPi.GPIO not available"
            logger.info("AlertManager running in MOCK mode (%s)", reason)
        else:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for pin in _ALL_PINS:
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            logger.info("AlertManager running in REAL GPIO mode (BCM numbering)")

        # Safe initial state: green on, everything else off, starter
        # inhibited until a pre-drive pass (fail-secure).
        self._write(LED_GREEN, 1)
        self._write(LED_YELLOW, 0)
        self._write(LED_RED, 0)
        self._write(BUZZER_PIN, 0)
        self._write(RELAY_PIN, _RELAY_INHIBIT)
        self.relay_locked = True
        self.current_level = "ALERT"
        logger.info("%sRelay starts INHIBITED (fail-secure); phase %s",
                    "[MOCK] " if self.mock else "", self.phase.value)

    # ------------------------------------------------------------------
    # Low-level pin access
    # ------------------------------------------------------------------

    def _write(self, pin: int, value: int) -> None:
        """
        Set one output pin, recording the state in both modes.

        Args:
            pin: BCM pin number.
            value: ``1`` for HIGH, ``0`` for LOW.
        """
        self.pin_states[pin] = value
        if not self.mock:
            GPIO.output(pin, GPIO.HIGH if value else GPIO.LOW)

    # ------------------------------------------------------------------
    # Alert level
    # ------------------------------------------------------------------

    def set_alert_level(self, level: str) -> None:
        """
        Apply the LED / buzzer configuration for an alert level.

        The relay is **not** touched here in any phase: DANGER does not
        inhibit the starter and ALERT does not release it. The relay is driven
        only by :meth:`lock_relay` / :meth:`unlock_relay` from the pre-drive
        verdict, so a locked driver who briefly leaves the frame (level
        falls to ALERT) stays locked, and a DANGER while driving changes
        nothing but the LEDs and buzzer.

        Only acts on a *change* of level, so calling this every frame (as
        ``main.py`` does) does not re-trigger the WARNING beep 30x a second
        or restart the continuous buzzer thread.

        Args:
            level: ``"ALERT"``, ``"WARNING"`` or ``"DANGER"``. Unknown values
                are logged and treated as ``"ALERT"`` (fail safe).
        """
        level = level.upper()
        if level not in ("ALERT", "WARNING", "DANGER"):
            logger.warning("Unknown alert level %r - treating as ALERT", level)
            level = "ALERT"

        if level == self.current_level:
            return
        previous = self.current_level
        self.current_level = level

        if level == "ALERT":
            self._stop_continuous_buzzer()
            self._write(LED_GREEN, 1)
            self._write(LED_YELLOW, 0)
            self._write(LED_RED, 0)
            self._write(BUZZER_PIN, 0)

        elif level == "WARNING":
            self._stop_continuous_buzzer()
            self._write(LED_GREEN, 0)
            self._write(LED_YELLOW, 1)
            self._write(LED_RED, 0)
            # Single beep, run on a thread so the 100 ms sleep doesn't stall
            # the frame loop.
            threading.Thread(
                target=self.trigger_buzzer, args=("short",), daemon=True
            ).start()

        else:  # DANGER - LEDs and buzzer only; the relay is never driven here.
            self._write(LED_GREEN, 0)
            self._write(LED_YELLOW, 0)
            self._write(LED_RED, 1)
            self.trigger_buzzer("continuous")

        if self.mock:
            logger.info(
                "[MOCK] level %s -> %s | LEDs G=%d Y=%d R=%d | buzzer=%d | relay=%s",
                previous, level,
                self.pin_states[LED_GREEN], self.pin_states[LED_YELLOW],
                self.pin_states[LED_RED], self.pin_states[BUZZER_PIN],
                self.get_relay_state(),
            )
        else:
            logger.info("Alert level %s -> %s", previous, level)

    # ------------------------------------------------------------------
    # Buzzer
    # ------------------------------------------------------------------

    def trigger_buzzer(self, pattern: str = "short") -> None:
        """
        Sound the buzzer with a named pattern.

        * ``"short"``      - 100 ms on (blocks the caller for 100 ms).
        * ``"long"``       - 500 ms on (blocks the caller for 500 ms).
        * ``"continuous"`` - stays on until the level drops below DANGER or
          :meth:`cleanup` is called. Runs on a daemon thread so it never
          blocks the main loop.

        Args:
            pattern: One of the names above. Unknown names fall back to
                ``"short"``.
        """
        if pattern == "continuous":
            self._start_continuous_buzzer()
            return

        on_time = BUZZER_PATTERNS.get(pattern)
        if on_time is None:
            logger.warning("Unknown buzzer pattern %r - using 'short'", pattern)
            on_time = BUZZER_PATTERNS["short"]

        if self.mock:
            logger.info("[MOCK] buzzer %s (%.0f ms)", pattern, on_time * 1000)

        self._write(BUZZER_PIN, 1)
        time.sleep(on_time)
        # Don't switch the buzzer off underneath a continuous alarm that may
        # have started while we were sleeping.
        if not self._continuous_active():
            self._write(BUZZER_PIN, 0)

    def _continuous_active(self) -> bool:
        """Whether the continuous-buzzer thread is currently running."""
        return self._buzzer_thread is not None and self._buzzer_thread.is_alive()

    def _start_continuous_buzzer(self) -> None:
        """Start the background continuous buzzer if it isn't already running."""
        with self._buzzer_lock:
            if self._continuous_active():
                return
            self._buzzer_stop.clear()
            self._buzzer_thread = threading.Thread(
                target=self._continuous_buzzer_loop, name="buzzer", daemon=True
            )
            self._buzzer_thread.start()
        if self.mock:
            logger.info("[MOCK] buzzer continuous ON")

    def _stop_continuous_buzzer(self) -> None:
        """Signal the continuous buzzer thread to stop and switch the pin off."""
        with self._buzzer_lock:
            if not self._continuous_active():
                return
            self._buzzer_stop.set()
            thread = self._buzzer_thread
        # Join outside the lock; the loop wakes every 100 ms so this is quick.
        if thread is not None:
            thread.join(timeout=0.5)
        self._write(BUZZER_PIN, 0)
        if self.mock:
            logger.info("[MOCK] buzzer continuous OFF")

    def _continuous_buzzer_loop(self) -> None:
        """
        Body of the continuous-buzzer thread.

        Holds the pin HIGH and polls the stop event every 100 ms. Polling
        (rather than a bare ``wait()``) keeps shutdown responsive even if the
        event is set from a signal handler.
        """
        self._write(BUZZER_PIN, 1)
        while not self._buzzer_stop.is_set():
            self._buzzer_stop.wait(0.1)
        self._write(BUZZER_PIN, 0)

    # ------------------------------------------------------------------
    # Phase
    # ------------------------------------------------------------------

    def set_phase(self, phase: Phase) -> None:
        """
        Switch operating phase.

        * ``PREDRIVE``   -> starter inhibited until an explicit
          :meth:`unlock_relay` (fail-secure; every ignition-OFF re-locks).
        * ``MONITORING`` -> :meth:`lock_relay` refused for as long as the
          phase lasts. The relay is deliberately left as it is (see the
          module docstring): key-ON precedes cranking, so releasing here
          would bypass the assessment.

        Args:
            phase: The new :class:`Phase`.
        """
        phase = Phase(phase)
        if phase is self.phase:
            return
        previous, self.phase = self.phase, phase
        logger.info("%sPhase %s -> %s", "[MOCK] " if self.mock else "",
                    previous.value, phase.value)
        if phase is Phase.PREDRIVE:
            self.lock_relay()
        elif self.relay_locked:
            logger.warning("%sEntering monitoring with the starter still INHIBITED - "
                           "no assessment passed; turn ignition OFF for pre-drive",
                           "[MOCK] " if self.mock else "")

    # ------------------------------------------------------------------
    # Relay
    # ------------------------------------------------------------------

    def lock_relay(self) -> None:
        """
        Inhibit the starter. Idempotent. **Refused outside pre-drive.**

        In ``Phase.MONITORING`` this logs a WARNING and returns without
        touching the pin - the vehicle may be moving, and the relay must
        never engage while driving regardless of what the caller thinks the
        fatigue level is. This guard is the structural enforcement of that
        rule; do not bypass it.
        """
        if self.phase is not Phase.PREDRIVE:
            logger.warning("%sRelay lock REFUSED - phase is %s; the starter relay only "
                           "engages during pre-drive",
                           "[MOCK] " if self.mock else "", self.phase.value)
            return
        if self.relay_locked:
            return
        self._write(RELAY_PIN, _RELAY_INHIBIT)
        self.relay_locked = True
        logger.warning("%sRelay LOCKED - starter inhibited",
                       "[MOCK] " if self.mock else "")

    def unlock_relay(self) -> None:
        """Release the starter. Idempotent."""
        if not self.relay_locked:
            return
        self._write(RELAY_PIN, _RELAY_ALLOW)
        self.relay_locked = False
        logger.info("%sRelay UNLOCKED - starter released",
                    "[MOCK] " if self.mock else "")

    def get_relay_state(self) -> str:
        """
        Current relay state.

        Returns:
            ``"LOCKED"`` if the starter is inhibited, else ``"UNLOCKED"``.
        """
        return "LOCKED" if self.relay_locked else "UNLOCKED"

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """
        Silence the buzzer, drive every output LOW and release GPIO.

        Safe to call more than once and in mock mode. Driving the relay pin
        LOW (and ``GPIO.cleanup()`` returning it to a floating input) leaves
        the starter **inhibited** - fail-secure. This is harmless if the
        engine is already running, and the next ignition-OFF returns to
        pre-drive for a fresh assessment anyway.
        """
        self._stop_continuous_buzzer()
        for pin in _ALL_PINS:
            self._write(pin, 0)
        self.relay_locked = True
        self.current_level = None

        if self.mock:
            logger.info("[MOCK] AlertManager cleanup - all outputs LOW")
            return

        try:
            GPIO.cleanup()
            logger.info("GPIO cleanup complete")
        except Exception as exc:  # pragma: no cover - hardware only
            logger.warning("GPIO.cleanup() failed: %s", exc)

    # ------------------------------------------------------------------
    # Head pose (Module 10)
    # ------------------------------------------------------------------

    def handle_head_pose(self, pose: Dict[str, object]) -> None:
        """
        Escalate to DANGER when the head pose estimator flags the driver.

        A nodding, turned-away or slumped head is treated as an immediate
        DANGER regardless of the eye-based FRS, because the eyes may be
        occluded or mis-tracked in exactly those poses. This goes through
        :meth:`set_alert_level`, so it drives the LEDs and buzzer only - it
        cannot engage the relay in any phase.

        Args:
            pose: Dict returned by ``HeadPoseEstimator.estimate()``. Only the
                ``alert`` flag and the three ``is_*`` flags are read.
        """
        if not pose.get("alert"):
            return

        # Name every condition that fired so the log explains *why* the
        # level escalated; several can be true at once (e.g. nod + tilt).
        reasons = [
            name
            for key, name in (
                ("is_nodding", "nodding"),
                ("is_looking_away", "looking away"),
                ("is_tilting", "tilting"),
            )
            if pose.get(key)
        ]
        # Only log on the transition into DANGER; set_alert_level() itself is
        # a no-op on repeats, and this keeps a sustained nod from producing
        # 30 log lines per second.
        if self.current_level != "DANGER":
            logger.warning(
                "Head pose alert (%s): pitch=%.1f yaw=%.1f roll=%.1f -> DANGER",
                ", ".join(reasons) or "unspecified",
                float(pose.get("pitch", 0.0)),
                float(pose.get("yaw", 0.0)),
                float(pose.get("roll", 0.0)),
            )
        self.set_alert_level("DANGER")
