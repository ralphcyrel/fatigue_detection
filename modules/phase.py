"""
Operating phases and starter-lock reasons.

The system runs in one of two phases, selected by the vehicle's ignition
state (see ``modules.ignition``):

* ``Phase.PREDRIVE``   — ignition OFF. The driver is recognised, their
  personal thresholds fetched, and a 30 s fatigue assessment run. This is
  the **only** phase in which the starter relay may engage: a pass releases
  the starter, anything else keeps it inhibited and raises an operator
  override request.
* ``Phase.MONITORING`` — ignition ON. The same metrics run continuously,
  but only the LEDs, buzzer and backend notifications react. The relay is
  never touched (``AlertManager.lock_relay`` refuses in this phase).

``LockReason`` names the conditions that keep the starter inhibited after
pre-drive; all of them resolve through the same operator override request
(``APIClient.request_override``). ``MICROSLEEP_DETECTED`` is kept distinct
from ``FATIGUE_DETECTED`` because it is a categorically different finding:
not "scored above the fatigue threshold" but "lost consciousness while
sitting still in a stationary vehicle".

Both enums are ``str`` subclasses so they serialise directly into the JSON
payloads sent to the Laravel backend.
"""

from enum import Enum


class Phase(str, Enum):
    """Operating phase, derived from the ignition state."""

    PREDRIVE = "predrive"
    MONITORING = "monitoring"


class LockReason(str, Enum):
    """Why the starter relay stayed inhibited after pre-drive."""

    FATIGUE_DETECTED = "fatigue_detected"
    MICROSLEEP_DETECTED = "microsleep_detected"
    DRIVER_NOT_RECOGNIZED = "driver_not_recognized"
    NO_BASELINE = "no_baseline"
