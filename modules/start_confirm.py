"""
Operator requests and the start confirmation (2026-10-19).

Two requests reach ``main.py`` from outside its frame loop:

* :data:`BEGIN` - the touchscreen's one primary action: START NEXT DRIVER
  in IDLE, BEGIN MONITORING in READY_TO_START.
* :data:`END_SESSION` - END SESSION, after the launcher's confirmation.

``main.py``'s fullscreen window takes no touch input of its own, so the
launcher's strip sends them as signals (SIGUSR1 / SIGUSR2); keys ``b`` / ``e``
in the window do the same. Each request is stamped with the session state it
was made in and honoured only in that state: the launcher draws its buttons
from a state file up to half a second old, and a tap on BEGIN MONITORING
made just as READY_TO_START was lost must never carry into a later one.

The start confirmation is behind ``config.START_CONFIRM_MODE`` so the
ignition sense planned for next semester can replace the button without
touching the state machine: :func:`make_start_confirm` returns an object
whose ``confirmed(state)`` the READY_TO_START loop polls every frame.
"""

import logging
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

BEGIN = "begin"
END_SESSION = "end_session"


def _state_name(state: object) -> str:
    """``SessionState.X`` -> ``"X"`` (``str()`` of a str-Enum is ``"SessionState.X"``)."""
    return str(getattr(state, "value", state))


class OperatorRequests:
    """
    Pending operator requests, at most one per kind (a newer one replaces it).

    :meth:`request` only stores, so it is safe to call from a signal
    handler; everything is logged when the request is taken or discarded.
    """

    def __init__(self) -> None:
        # kind -> (state it was made in, source, time.monotonic())
        self._pending: Dict[str, Tuple[str, str, float]] = {}

    def request(self, kind: str, state: str, source: str) -> None:
        """Record ``kind`` as made in session ``state`` by ``source``."""
        self._pending[kind] = (_state_name(state), source, time.monotonic())

    def pending(self, kind: str) -> bool:
        """Whether ``kind`` is waiting (in any state)."""
        return kind in self._pending

    def take(self, kind: str, state: str) -> Optional[str]:
        """
        Consume ``kind`` if it was made in ``state``.

        Returns:
            The request's source, or ``None`` if none is pending. A request
            made in another state is discarded (logged) and ``None`` returned.
        """
        entry = self._pending.pop(kind, None)
        if entry is None:
            return None
        made_in, source, _t = entry
        if made_in != _state_name(state):
            logger.warning("Operator request %r from %s IGNORED: made in %s, now %s",
                           kind, source, made_in, _state_name(state))
            return None
        logger.info("Operator request %r from %s accepted in %s", kind, source,
                    _state_name(state))
        return source

    def discard(self, kind: Optional[str] = None, why: str = "") -> None:
        """Drop ``kind`` (every kind if ``None``), logging what was dropped."""
        for k in ([kind] if kind is not None else list(self._pending)):
            entry = self._pending.pop(k, None)
            if entry is not None:
                logger.info("Operator request %r from %s (made in %s) discarded%s",
                            k, entry[1], entry[0], f": {why}" if why else "")


class TouchStartConfirm:
    """The BEGIN MONITORING button (launcher strip / key ``b``)."""

    mode = "touch"
    prompt = "then tap BEGIN MONITORING"

    def __init__(self, requests: OperatorRequests) -> None:
        self.requests = requests

    def confirmed(self, state: str) -> Optional[str]:
        """The confirmation's source if one was made in ``state``, else ``None``."""
        return self.requests.take(BEGIN, state)

    def reset(self, why: str) -> None:
        """Forget a confirmation made before the current READY_TO_START."""
        self.requests.discard(BEGIN, why)


def make_start_confirm(mode: str, requests: OperatorRequests) -> TouchStartConfirm:
    """
    The start confirmation for ``config.START_CONFIRM_MODE``.

    Raises:
        NotImplementedError: for ``"gpio"`` (the ignition sense is deferred
            hardware).
        ValueError: for any other unknown mode.
    """
    if mode == "touch":
        return TouchStartConfirm(requests)
    if mode == "gpio":
        raise NotImplementedError(
            "START_CONFIRM_MODE = 'gpio' is not implemented: the ignition-position sense "
            "is deferred hardware (next semester) and nothing is wired to read it. Set "
            "START_CONFIRM_MODE = 'touch' in config/config.py (the BEGIN MONITORING button).")
    raise ValueError(f"START_CONFIRM_MODE must be 'touch' or 'gpio', got {mode!r}")
