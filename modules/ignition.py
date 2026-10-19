"""
Module 11 - Operating phase source (no ignition input yet).

The phase loops ask an *ignition* object for the operating phase::

    Phase.PREDRIVE     (starter may be inhibited / released)
    Phase.MONITORING   (alerts only; relay never touched)

On the current hardware the Pi has NO input from the key switch: its only
connection to the starter circuit is the relay control on GPIO 17 (see
``config.RELAY_PIN``). The phase is therefore pinned for the whole run by
:class:`ForcedIgnition` - ``--force-phase``, ``--sequence`` and the default
all use it - and pre-drive moves into monitoring only on an explicit start
confirmation (``config.START_CONFIRM_MODE``).

TODO(deferred hardware, next semester): the debounced ignition-position
sense that used to live here (``IgnitionSensor``, GPIO 25 through an
optocoupler) was removed on 2026-10-19 because the input is not wired. When
it is, it should provide the ``"gpio"`` start confirmation rather than
switch the phase on its own.
"""

import logging
from typing import Optional

from modules.phase import Phase

logger = logging.getLogger(__name__)


class ForcedIgnition:
    """
    Always reports one phase.

    Used for every run (there is no ignition input to read), so either phase
    can be exercised and nothing flips the phase underneath a loop.
    """

    mock = True

    def __init__(self, phase: Phase, source: str = "--force-phase") -> None:
        self.forced_phase = Phase(phase)
        logger.info("Ignition FORCED to %s by %s", self.forced_phase.value, source)

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
