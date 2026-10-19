"""
Operating phases and starter-lock reasons.

The system runs in one of two phases. There is no ignition input yet (see
``modules.ignition``): a session starts in pre-drive and enters monitoring
on an explicit start confirmation.

* ``Phase.PREDRIVE``   — before the start. The driver is recognised, their
  personal thresholds fetched, and a rolling fatigue assessment (30 s
  minimum) run. This is the **only** phase in which the starter relay may
  engage: a pass releases the starter (READY_TO_START), anything else keeps
  it inhibited and a hard failure raises an operator override request.
* ``Phase.MONITORING`` — after the start. The same metrics run continuously,
  but only the LEDs, buzzer and backend notifications react. The relay is
  never touched (``AlertManager.lock_relay`` refuses in this phase).

``LockReason`` names the conditions that keep the starter inhibited after
pre-drive; all of them resolve through the same operator override request
(``APIClient.request_override``). ``MICROSLEEP_DETECTED`` is kept distinct
from ``FATIGUE_DETECTED`` because it is a categorically different finding:
not "scored above the fatigue threshold" but "lost consciousness while
sitting still in a stationary vehicle". ``FOREIGN_DEVICE_BASELINE`` is kept
distinct from ``NO_BASELINE`` for the same reason: the driver *is* enrolled,
but only on another unit. A baseline captured at a different camera mounting
is not trusted to release the starter - one that reads low would let a
fatigued driver pass - so the unit locks without assessing, and the operator
response differs (re-enrol on this unit, versus enrol at all).

Both enums are ``str`` subclasses so they serialise directly into the JSON
payloads sent to the Laravel backend.
"""

from enum import Enum


class Phase(str, Enum):
    """Operating phase (pinned per run until an ignition sense exists)."""

    PREDRIVE = "predrive"
    MONITORING = "monitoring"


class SessionState(str, Enum):
    """
    Where a ``main.py`` session is (2026-10-19). Published to the launcher
    (``main.py --state-file``), which shows its strip buttons from it.

    Pre-drive runs IDENTIFYING -> SETTLING -> ASSESSING -> READY_TO_START and
    enters MONITORING only on a start confirmation. READY_TO_START is the only
    state in which pre-drive leaves the starter released, and it is not a
    latch: losing the rolling pass, the face or the driver returns to
    ASSESSING. LOCKED is a hard failure awaiting an operator override or a
    reset. IDLE follows END SESSION: starter inhibited, awaiting the next
    driver.
    """

    IDLE = "IDLE"
    IDENTIFYING = "IDENTIFYING"
    SETTLING = "SETTLING"
    ASSESSING = "ASSESSING"
    READY_TO_START = "READY_TO_START"
    LOCKED = "LOCKED"
    MONITORING = "MONITORING"


class LockReason(str, Enum):
    """Why the starter relay stayed inhibited after pre-drive."""

    FATIGUE_DETECTED = "fatigue_detected"
    MICROSLEEP_DETECTED = "microsleep_detected"
    DRIVER_NOT_RECOGNIZED = "driver_not_recognized"
    NO_BASELINE = "no_baseline"
    FOREIGN_DEVICE_BASELINE = "foreign_device_baseline"
