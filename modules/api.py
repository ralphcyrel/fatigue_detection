"""
Module 9 - HTTP client for the Laravel backend.

`APIClient` is the single point of contact between the Pi and the Laravel
API. It owns a ``requests.Session`` (so the TCP connection and auth headers
are reused across calls) and wraps every endpoint the other modules need:

    GET  /drivers/encodings                 -> DriverRecognizer.load_encodings()
    GET  /drivers/{id}/thresholds           -> per-driver calibration profile
    POST /drivers/{id}/enroll               -> save encoding + baselines
    POST /fatigue-events                    -> log a DANGER event (monitoring)
    POST /assessments                       -> pre-drive verdict + per-frame series
    POST /override-requests                 -> starter stays inhibited; ask operator
    GET  /override-requests/{id}            -> operator decision (pending/approved/denied)
    GET  /drivers/{id}/relay-override       -> (deprecated) per-driver unlock flag
    GET  /ping                              -> backend reachability

``config.API_BASE_URL`` already ends in ``/api`` (e.g.
``http://localhost:8000/api``), so paths here are written relative to it.

Design rules, because this runs inside a real-time detection loop:

* every request uses ``config.API_TIMEOUT`` so a dead link can never hang
  a frame for more than a few seconds;
* every method catches ``requests.exceptions.RequestException`` (and bad
  JSON), logs it, and returns ``None`` / ``False`` - the loop must keep
  running with the last known state if the backend is unreachable;
* :meth:`push_fatigue_event` is fire-and-forget on a daemon thread so a
  DANGER frame is never delayed by network latency.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from config import config
from modules.phase import LockReason, Phase

logger = logging.getLogger(__name__)

# HTTP status codes accepted as "created / ok" for the POST endpoints.
_ENROLL_OK = (200, 201)
_EVENT_OK = (201,)
_CREATED_OK = (200, 201)

# Statuses the operator portal may return for an override request.
OVERRIDE_STATUSES = ("pending", "approved", "denied")


class APIClient:
    """
    Thin, fault-tolerant wrapper around the Laravel REST API.

    Typical usage::

        api = APIClient()                      # reads URL + token from config
        if not api.ping():
            logger.warning("backend offline - running with cached data")
        thresholds = api.get_driver_thresholds(driver_id)

    Attributes:
        base_url: API root without a trailing slash.
        token: Bearer token sent on every request (may be empty).
        session: Shared ``requests.Session`` with auth/JSON headers set.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self, base_url: Optional[str] = None, token: Optional[str] = None
    ) -> None:
        """
        Build the session and default headers.

        Args:
            base_url: API root. Defaults to ``config.API_BASE_URL``.
            token: Bearer token. Defaults to ``config.API_TOKEN``.
        """
        self.base_url: str = (base_url or config.API_BASE_URL).rstrip("/")
        self.token: str = token if token is not None else config.API_TOKEN
        self.timeout: int = config.API_TIMEOUT

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        if not self.token:
            logger.warning("APIClient: no API token configured (FATIGUE_API_TOKEN)")
        logger.info("APIClient targeting %s (timeout %ss)", self.base_url, self.timeout)

    # ------------------------------------------------------------------
    # Internal request helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        """Join ``path`` (with or without a leading slash) onto the base URL."""
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> Optional[requests.Response]:
        """
        Perform one HTTP request, swallowing transport errors.

        Args:
            method: ``"GET"`` or ``"POST"``.
            path: Endpoint path relative to ``base_url``.
            **kwargs: Passed through to ``Session.request`` (e.g. ``json=``).

        Returns:
            The ``Response`` (any status code), or ``None`` on a transport
            level failure (DNS, connection refused, timeout, ...).
        """
        url = self._url(path)
        try:
            return self.session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.exceptions.RequestException as exc:
            logger.error("%s %s failed: %s", method, url, exc)
            return None

    def _get_json(self, path: str) -> Optional[Any]:
        """
        GET an endpoint and decode its JSON body.

        Returns:
            Decoded JSON on HTTP 2xx, otherwise ``None`` (logged).
        """
        resp = self._request("GET", path)
        if resp is None:
            return None
        if not resp.ok:
            logger.error("GET %s returned HTTP %s: %s", path, resp.status_code, resp.text[:200])
            return None
        try:
            return resp.json()
        except ValueError as exc:
            logger.error("GET %s returned invalid JSON: %s", path, exc)
            return None

    @staticmethod
    def _unwrap(payload: Any) -> Any:
        """
        Strip Laravel's conventional ``{"data": ...}`` envelope if present.

        Laravel API resources wrap responses in ``data``; plain controllers
        don't. Accept both so the backend can evolve without touching the Pi.
        """
        if isinstance(payload, dict) and "data" in payload and len(payload) == 1:
            return payload["data"]
        return payload

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def get_face_encodings(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch every enrolled driver's face encoding.

        ``GET /drivers/encodings``

        Returns:
            List of ``{"driver_id": int, "name": str, "encoding": str|list}``
            records, or ``None`` on failure.
        """
        payload = self._unwrap(self._get_json("/drivers/encodings"))
        if payload is None:
            return None
        if not isinstance(payload, list):
            logger.error("GET /drivers/encodings: expected a list, got %s", type(payload).__name__)
            return None
        logger.info("Fetched %d face encoding records", len(payload))
        return payload

    def get_driver_thresholds(self, driver_id: int) -> Optional[Dict[str, float]]:
        """
        Fetch a driver's calibrated baselines and thresholds.

        ``GET /drivers/{driver_id}/thresholds``

        Args:
            driver_id: Primary key of the driver in the Laravel DB.

        Returns:
            ``{"ear_baseline", "ear_threshold", "perclos_baseline",
            "perclos_threshold", "blink_duration_baseline",
            "blink_frequency_baseline", "mar_baseline", "yawn_threshold"}``
            as floats, or ``None`` on failure. The two MAR keys are absent
            for drivers enrolled before yawn detection existed; ``main.py``
            fills them from ``DEFAULT_THRESHOLDS``.
        """
        payload = self._unwrap(self._get_json(f"/drivers/{driver_id}/thresholds"))
        if payload is None:
            return None
        if not isinstance(payload, dict):
            logger.error("GET thresholds for driver %s: expected an object", driver_id)
            return None

        # Coerce to float - Laravel may serialise decimals as strings.
        thresholds: Dict[str, float] = {}
        for key, value in payload.items():
            try:
                thresholds[key] = float(value)
            except (TypeError, ValueError):
                logger.debug("Ignoring non-numeric threshold field %r=%r", key, value)
        logger.info("Loaded thresholds for driver %s: %s", driver_id, thresholds)
        return thresholds

    def save_driver_enrollment(
        self, driver_id: int, face_encoding: List[float], thresholds: Dict[str, float]
    ) -> bool:
        """
        Persist a newly enrolled driver's encoding and calibration profile.

        ``POST /drivers/{driver_id}/enroll``

        Args:
            driver_id: Driver to attach the data to (must already exist).
            face_encoding: 128-d list of floats.
            thresholds: Baselines dict from ``CalibrationManager``. Every
                key is forwarded verbatim, so this now includes
                ``mar_baseline`` and ``yawn_threshold``; the Laravel side
                must accept (and store) those two extra fields.

        Returns:
            ``True`` on HTTP 200/201, ``False`` otherwise.
        """
        body = {
            # Ensure plain Python floats - numpy scalars are not JSON-serialisable.
            "face_encoding": [float(x) for x in face_encoding],
            "thresholds": {k: float(v) for k, v in thresholds.items()},
        }
        resp = self._request("POST", f"/drivers/{driver_id}/enroll", json=body)
        if resp is None:
            return False
        if resp.status_code in _ENROLL_OK:
            logger.info("Enrollment saved for driver %s", driver_id)
            return True
        logger.error(
            "Enrollment for driver %s rejected: HTTP %s %s",
            driver_id, resp.status_code, resp.text[:200],
        )
        return False

    def push_fatigue_event(
        self,
        driver_id: Optional[int],
        frs_result: Dict[str, Any],
        ear: float,
        perclos: float,
        relay_triggered: bool,
        blocking: bool = False,
        phase: Phase = Phase.MONITORING,
    ) -> bool:
        """
        Log a fatigue event to the backend.

        ``POST /fatigue-events``

        Sent from the monitoring phase (ignition ON) as the operator
        notification for DANGER; the relay is never engaged there, so
        ``relay_triggered`` is kept only for backward compatibility and is
        always ``False`` from that phase.

        By default the request is dispatched on a daemon thread and this
        method returns immediately, so the detection loop is never held up
        by network latency. The HTTP outcome is logged by the thread.

        Args:
            driver_id: Driver the event belongs to, or ``None`` if the
                driver was not recognised.
            frs_result: Dict returned by ``FRSCalculator.compute()``.
            ear: Raw EAR at the time of the event.
            perclos: PERCLOS at the time of the event.
            relay_triggered: Whether the starter relay was inhibited.
            blocking: If ``True``, send synchronously and return the real
                outcome (useful for tests and the enrollment flow).
            phase: Operating phase the event was raised in.

        Returns:
            Non-blocking: ``True`` if the request was queued.
            Blocking: ``True`` on HTTP 201, ``False`` otherwise.
        """
        body = {
            # None when the driver was never recognised (monitoring with defaults).
            "driver_id": int(driver_id) if driver_id is not None else None,
            "frs_score": float(frs_result["frs"]),
            "alert_level": str(frs_result["level"]),
            "ear_value": float(ear),
            "perclos_value": float(perclos),
            "relay_triggered": bool(relay_triggered),
            "phase": Phase(phase).value,
            "ignition_on": Phase(phase) is Phase.MONITORING,
            "device_id": config.DEVICE_ID,
            # ISO 8601 with explicit UTC offset, e.g. 2026-09-15T09:31:07.123456+00:00
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if blocking:
            return self._send_fatigue_event(body)

        threading.Thread(
            target=self._send_fatigue_event, args=(body,),
            name="fatigue-event", daemon=True,
        ).start()
        return True

    def _send_fatigue_event(self, body: Dict[str, Any]) -> bool:
        """POST one fatigue event body; log and return the outcome."""
        resp = self._request("POST", "/fatigue-events", json=body)
        if resp is None:
            return False
        if resp.status_code in _EVENT_OK:
            logger.info(
                "Fatigue event logged for driver %s (FRS %.3f)",
                body["driver_id"], body["frs_score"],
            )
            return True
        logger.error(
            "Fatigue event rejected: HTTP %s %s", resp.status_code, resp.text[:200]
        )
        return False

    def check_relay_override(self, driver_id: int) -> Optional[bool]:
        """
        Ask whether an operator has requested the relay be unlocked.

        ``GET /drivers/{driver_id}/relay-override``

        .. deprecated::
            Keyed by driver, so it cannot serve the "driver not recognised"
            lock reason. The pre-drive flow now uses
            :meth:`request_override` + :meth:`check_override_request`.
            Kept for backend compatibility; no longer called by ``main.py``.

        Accepts either a bare JSON boolean or an object containing an
        ``override`` / ``relay_override`` / ``unlock`` key.

        Args:
            driver_id: Driver whose vehicle is being queried.

        Returns:
            ``True`` if an override is active, ``False`` if not, ``None`` on
            error (callers should treat ``None`` as "no change").
        """
        payload = self._unwrap(self._get_json(f"/drivers/{driver_id}/relay-override"))
        if payload is None:
            return None
        if isinstance(payload, bool):
            return payload
        if isinstance(payload, dict):
            for key in ("override", "relay_override", "unlock"):
                if key in payload:
                    return bool(payload[key])
        logger.error("Unexpected relay-override payload: %r", payload)
        return None

    # ------------------------------------------------------------------
    # Pre-drive assessment + operator override
    # ------------------------------------------------------------------

    def post_assessment(
        self,
        device_id: str,
        driver_id: Optional[int],
        result: Dict[str, Any],
        lock_reason: Optional[LockReason] = None,
    ) -> Optional[int]:
        """
        Persist a pre-drive assessment: verdict, aggregates and the full
        per-frame series.

        ``POST /assessments``

        Args:
            device_id: This vehicle unit (``config.DEVICE_ID``).
            driver_id: Recognised driver, or ``None`` if recognition failed.
            result: Dict from ``PredriveAssessment.result()``; ``samples`` is
                sent verbatim (one dict per frame, keys as in
                ``modules.assessment.SAMPLE_FIELDS``).
            lock_reason: Why the starter stays inhibited, or ``None`` on a pass.

        Returns:
            The backend's assessment id, or ``None`` on failure.
        """
        body: Dict[str, Any] = {
            "device_id": device_id,
            "driver_id": driver_id,
            "started_at": datetime.fromtimestamp(result["started_at"], timezone.utc).isoformat(),
            "ended_at": datetime.fromtimestamp(result["ended_at"], timezone.utc).isoformat(),
            "duration_s": float(result["duration_s"]),
            "n_samples": int(result["n_samples"]),
            "n_no_face": int(result["n_no_face"]),
            "mean_frs": float(result["mean_frs"]),
            "median_frs": float(result["median_frs"]),
            "max_frs": float(result["max_frs"]),
            "worst_window_mean": float(result["worst_window_mean"]),
            "worst_window_s": float(result["worst_window_s"]),
            "pass_threshold": float(result["pass_threshold"]),
            "pose_override_frames": int(result["pose_override_frames"]),
            "yawns": int(result["yawns"]),
            "void": bool(result["void"]),
            "passed": bool(result["passed"]),
            "lock_reason": LockReason(lock_reason).value if lock_reason else None,
            "samples": [
                {k: (None if v is None else (bool(v) if isinstance(v, bool) else
                     (v if isinstance(v, str) else float(v))))
                 for k, v in sample.items()}
                for sample in result["samples"]
            ],
        }
        resp = self._request("POST", "/assessments", json=body)
        if resp is None:
            return None
        if resp.status_code not in _CREATED_OK:
            logger.error("Assessment rejected: HTTP %s %s", resp.status_code, resp.text[:200])
            return None
        payload = self._unwrap(self._safe_json(resp))
        assessment_id = payload.get("id") if isinstance(payload, dict) else None
        logger.info("Assessment saved (id=%s, passed=%s, worst_window=%.3f, %d samples)",
                    assessment_id, body["passed"], body["worst_window_mean"], body["n_samples"])
        return int(assessment_id) if assessment_id is not None else None

    def request_override(
        self,
        device_id: str,
        driver_id: Optional[int],
        reason: LockReason,
        assessment_id: Optional[int] = None,
    ) -> Optional[int]:
        """
        Tell the operator portal the starter is inhibited and why.

        ``POST /override-requests``

        Args:
            device_id: This vehicle unit.
            driver_id: Recognised driver, or ``None`` (``DRIVER_NOT_RECOGNIZED``).
            reason: One of the three :class:`LockReason` values.
            assessment_id: Id returned by :meth:`post_assessment`, if any.

        Returns:
            The request id to poll with :meth:`check_override_request`, or
            ``None`` on failure.
        """
        body = {
            "device_id": device_id,
            "driver_id": driver_id,
            "reason": LockReason(reason).value,
            "assessment_id": assessment_id,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        resp = self._request("POST", "/override-requests", json=body)
        if resp is None:
            return None
        if resp.status_code not in _CREATED_OK:
            logger.error("Override request rejected: HTTP %s %s", resp.status_code, resp.text[:200])
            return None
        payload = self._unwrap(self._safe_json(resp))
        request_id = payload.get("id") if isinstance(payload, dict) else None
        logger.warning("Override request %s raised (device=%s, driver=%s, reason=%s)",
                       request_id, device_id, driver_id, body["reason"])
        return int(request_id) if request_id is not None else None

    def check_override_request(self, request_id: int) -> Optional[str]:
        """
        Poll the operator's decision on an override request.

        ``GET /override-requests/{request_id}``

        Accepts a bare JSON string or an object with a ``status`` key.

        Returns:
            ``"pending"``, ``"approved"`` or ``"denied"``; ``None`` on error
            or an unexpected payload (callers treat ``None`` as "no change").
        """
        payload = self._unwrap(self._get_json(f"/override-requests/{request_id}"))
        if payload is None:
            return None
        status = payload.get("status") if isinstance(payload, dict) else payload
        if isinstance(status, str) and status.lower() in OVERRIDE_STATUSES:
            return status.lower()
        logger.error("Unexpected override-request payload: %r", payload)
        return None

    @staticmethod
    def _safe_json(resp: Any) -> Any:
        """``resp.json()`` or ``None`` if the body is not JSON."""
        try:
            return resp.json()
        except ValueError:
            return None

    def ping(self) -> bool:
        """
        Check whether the backend is reachable.

        ``GET /ping``

        Returns:
            ``True`` on any HTTP 2xx, ``False`` on error or non-2xx.
        """
        resp = self._request("GET", "/ping")
        reachable = resp is not None and resp.ok
        if reachable:
            logger.info("Backend reachable at %s", self.base_url)
        else:
            logger.warning("Backend NOT reachable at %s", self.base_url)
        return reachable
