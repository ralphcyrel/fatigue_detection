"""
Module 9 - HTTP client for the Laravel backend.

`APIClient` is the single point of contact between the Pi and the Laravel
API. It owns a ``requests.Session`` (so the TCP connection and auth headers
are reused across calls) and wraps every endpoint the other modules need:

    GET  /drivers/encodings                 -> DriverRecognizer.load_encodings()
    GET  /devices/{device_id}/calibration   -> active calibration of the driver
                                               assigned to this device
    POST /drivers/{id}/enroll               -> save encoding + baselines
    POST /fatigue-events                    -> log a DANGER event (monitoring)
    POST /monitoring-faults                 -> open / refresh / resolve a no-face
                                               or danger-latched fault
    POST /assessments                       -> pre-drive verdict + per-frame series
    POST /override-requests                 -> starter stays inhibited; ask operator
    GET  /override-requests/{id}            -> operator decision (pending/approved/denied)
    GET  /drivers/{id}/relay-override       -> (deprecated) per-driver unlock flag
    GET  /ping                              -> backend reachability
    POST /devices/{device_id}/heartbeat     -> "still online" + actual relay state

``config.API_BASE_URL`` already ends in ``/api`` (e.g.
``http://localhost:8000/api``), so paths here are written relative to it.

Design rules, because this runs inside a real-time detection loop:

* every request uses ``config.API_TIMEOUT`` so a dead link can never hang
  a frame for more than a few seconds;
* every method catches ``requests.exceptions.RequestException`` (and bad
  JSON), logs it, and returns ``None`` / ``False`` - the loop must keep
  running with the last known state if the backend is unreachable;
* :meth:`push_fatigue_event` and :meth:`post_heartbeat` are fire-and-forget
  on a daemon thread so a DANGER frame (or the 30 s heartbeat) is never
  delayed by network latency.
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
# /fatigue-events: 201 = created, 200 = duplicate deduped server-side,
# 202 = accepted and queued for retry. All three mean the data was taken.
_EVENT_OK = (200, 201, 202)
_EVENT_OUTCOME = {200: "deduplicated", 201: "created", 202: "queued for retry"}
_CREATED_OK = (200, 201)
_HEARTBEAT_OK = (200, 201, 204)

# AlertManager.get_relay_state() -> the heartbeat's ``confirmed_state``
# vocabulary. "interrupted" = starter circuit inhibited. Anything else
# (``None`` before the relay has been driven) is reported as "unknown".
RELAY_CONFIRMED_STATE = {
    "UNLOCKED": "normal",
    "LOCKED": "interrupted",
}
CONFIRMED_STATE_UNKNOWN = "unknown"

# POST /monitoring-faults: fault_type -> severity. danger_latched is critical:
# the last thing the unit saw was DANGER and it can no longer see the driver.
MONITORING_FAULT_SEVERITY = {
    "no_face": "warning",
    "danger_latched": "critical",
}

# Statuses the operator portal may return for an override request.
OVERRIDE_STATUSES = ("pending", "approved", "denied")

# How GET /devices/{id}/calibration lays out the numbers -> the flat names the
# rest of the Pi uses (``modules.pipeline``, ``main.DEFAULT_THRESHOLDS``).
# The backend groups them and drops the suffix from the baselines but keeps
# it on the thresholds, so the mapping is spelled out rather than derived.
CALIBRATION_FIELD_MAP = (
    # (group, backend key,      pipeline key)
    ("baselines",  "ear",             "ear_baseline"),
    ("baselines",  "perclos",         "perclos_baseline"),
    ("baselines",  "blink_duration",  "blink_duration_baseline"),
    ("baselines",  "blink_frequency", "blink_frequency_baseline"),
    ("baselines",  "mar",             "mar_baseline"),
    ("thresholds", "ear_threshold",   "ear_threshold"),
    ("thresholds", "yawn_threshold",  "yawn_threshold"),
)


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

        # Outcome of the last heartbeat (``None`` until one has been sent).
        # Only used to log the offline -> online transition once rather
        # than every 30 s; races between heartbeat threads are harmless.
        self._heartbeat_ok: Optional[bool] = None

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
        self,
        method: str,
        path: str,
        *,
        error_level: int = logging.ERROR,
        **kwargs: Any,
    ) -> Optional[requests.Response]:
        """
        Perform one HTTP request, swallowing transport errors.

        Args:
            method: ``"GET"`` or ``"POST"``.
            path: Endpoint path relative to ``base_url``.
            error_level: Log level for a transport failure. ERROR by
                default; the periodic heartbeat passes DEBUG so an offline
                backend does not write an error line every 30 s.
            **kwargs: Passed through to ``Session.request`` (e.g. ``json=``).

        Returns:
            The ``Response`` (any status code), or ``None`` on a transport
            level failure (DNS, connection refused, timeout, ...).
        """
        url = self._url(path)
        try:
            return self.session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.exceptions.RequestException as exc:
            logger.log(error_level, "%s %s failed: %s", method, url, exc)
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

    def get_drivers(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch the driver roster (used by the touchscreen launcher's picker).

        ``GET /drivers``

        Returns:
            List of ``{"id": int, "full_name": str, "device_id": str|None,
            "is_enrolled": bool}`` records, or ``None`` on failure.
            ``device_id`` is the unit's string identifier (e.g. ``pi-01``),
            comparable to ``config.DEVICE_ID``; ``is_enrolled`` reflects an
            active calibration on the backend.
        """
        payload = self._unwrap(self._get_json("/drivers"))
        if payload is None:
            return None
        if not isinstance(payload, list):
            logger.error("GET /drivers: expected a list, got %s", type(payload).__name__)
            return None
        logger.info("Fetched %d driver records", len(payload))
        return payload

    def get_face_encodings(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch every enrolled driver's face encoding.

        ``GET /drivers/encodings``

        Returns:
            List of ``{"driver_id": int, "name": str, "face_encoding": str|list}``
            records, or ``None`` on failure. Older backends used the key
            ``encoding``; ``DriverRecognizer`` accepts either.
        """
        payload = self._unwrap(self._get_json("/drivers/encodings"))
        if payload is None:
            return None
        if not isinstance(payload, list):
            logger.error("GET /drivers/encodings: expected a list, got %s", type(payload).__name__)
            return None
        logger.info("Fetched %d face encoding records", len(payload))
        return payload

    def get_device_calibration(
        self,
        device_id: Optional[str] = None,
        expected_driver_id: Optional[int] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Fetch the active calibration profile for the driver assigned to
        this device.

        ``GET /devices/{device_id}/calibration``

        The backend resolves device -> assigned driver -> active calibration
        itself, so the Pi never handles calibration or driver database ids
        here. It answers 404 in three operationally distinct situations,
        each with ``lock_reason: "no_baseline"`` and its own ``status``
        (device not registered, no driver assigned, no active calibration).
        All three are the same lock condition for the Pi, so this method
        returns ``None`` for each, but logs the ``status`` so an operator
        can tell which one it was.

        Args:
            device_id: Device string (``pi-01``). Defaults to
                ``config.DEVICE_ID``.
            expected_driver_id: The driver the Pi recognised on camera. If
                the record names a different ``driver_id`` a warning is
                logged - the calibration belongs to whoever is *assigned* to
                the device, not necessarily whoever is in the seat.

        Returns:
            ``{"ear_baseline", "ear_threshold", "perclos_baseline",
            "blink_duration_baseline", "blink_frequency_baseline",
            "mar_baseline", "yawn_threshold"}`` as floats, or ``None`` on
            404 / failure. The two MAR keys are absent for drivers enrolled
            before yawn detection existed; ``main.py`` fills them from
            ``DEFAULT_THRESHOLDS``.
        """
        device_id = device_id or config.DEVICE_ID
        path = f"/devices/{device_id}/calibration"
        resp = self._request("GET", path)
        if resp is None:
            return None

        if resp.status_code == 404:
            payload = self._unwrap(self._safe_json(resp))
            info = payload if isinstance(payload, dict) else {}
            if info.get("lock_reason") == LockReason.NO_BASELINE.value:
                logger.warning(
                    "No baseline for device %s: status=%r (%s)",
                    device_id, info.get("status"), info.get("message", "no message"),
                )
            else:
                logger.error("GET %s returned HTTP 404 without a no_baseline lock reason: %s",
                             path, resp.text[:200])
            return None

        if not resp.ok:
            logger.error("GET %s returned HTTP %s: %s", path, resp.status_code, resp.text[:200])
            return None

        # Success body is {"status": "ok", "data": {...}} - two keys, so the
        # generic single-key ``_unwrap`` does not apply.
        payload = self._safe_json(resp)
        record = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(record, dict):
            logger.error("GET %s: expected {\"data\": {...}}, got %s", path, resp.text[:200])
            return None

        record_driver = record.get("driver_id")
        if (expected_driver_id is not None and record_driver is not None
                and int(record_driver) != int(expected_driver_id)):
            logger.warning(
                "Device %s is assigned to driver %s (%s) but the camera recognised driver %s - "
                "using the assigned driver's calibration",
                device_id, record_driver, record.get("driver_name", "?"), expected_driver_id,
            )

        # Flatten the grouped record into pipeline names, coercing to float -
        # Laravel may serialise decimals as strings. A missing or null field
        # is simply absent (main.py fills MAR keys from DEFAULT_THRESHOLDS).
        thresholds: Dict[str, float] = {}
        for group, key, name in CALIBRATION_FIELD_MAP:
            section = record.get(group)
            value = section.get(key) if isinstance(section, dict) else None
            if value is None:
                continue
            try:
                thresholds[name] = float(value)
            except (TypeError, ValueError):
                logger.debug("Ignoring non-numeric calibration field %s.%s=%r", group, key, value)
        logger.info("Loaded calibration for device %s (driver %s): %s",
                    device_id, record_driver, thresholds)
        return thresholds

    def get_driver_thresholds(self, driver_id: int) -> Optional[Dict[str, float]]:
        """
        Backward-compatible alias for :meth:`get_device_calibration`.

        ``GET /drivers/{id}/thresholds`` no longer exists; the calibration
        is keyed by this device (``config.DEVICE_ID``). ``driver_id`` is
        only used to warn if the assigned driver differs.
        """
        return self.get_device_calibration(expected_driver_id=driver_id)

    def save_driver_enrollment(
        self,
        driver_id: int,
        face_encoding: List[float],
        thresholds: Dict[str, float],
        sample_duration_s: Optional[int] = None,
        device_id: Optional[str] = None,
    ) -> bool:
        """
        Persist a newly enrolled driver's encoding and calibration profile.

        ``POST /drivers/{driver_id}/enroll``

        The endpoint takes every baseline as a top-level field (not nested
        under ``thresholds``) plus three provenance fields: how long the
        calibration ran, which device captured it and when.

        Args:
            driver_id: Driver to attach the data to (must already exist).
            face_encoding: 128-d list of floats.
            thresholds: Baselines dict from ``CalibrationManager``. Every
                key is forwarded, so this includes ``mar_baseline`` and
                ``yawn_threshold`` when MAR was sampled.
            sample_duration_s: Seconds the calibration observed the driver.
                Defaults to ``config.CALIBRATION_DURATION``.
            device_id: Device string that captured the calibration. Must be
                registered in the backend. Defaults to ``config.DEVICE_ID``.

        Returns:
            ``True`` on HTTP 200/201, ``False`` otherwise.
        """
        body: Dict[str, Any] = {
            # Ensure plain Python floats - numpy scalars are not JSON-serialisable.
            "face_encoding": [float(x) for x in face_encoding],
            "sample_duration_s": int(sample_duration_s if sample_duration_s is not None
                                     else config.CALIBRATION_DURATION),
            "captured_on_device_id": device_id or config.DEVICE_ID,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        body.update({k: float(v) for k, v in thresholds.items()})
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
            logger.debug(
                "Fatigue event outcome: HTTP %s (%s)",
                resp.status_code, _EVENT_OUTCOME[resp.status_code],
            )
            return True
        logger.error(
            "Fatigue event rejected: HTTP %s %s", resp.status_code, resp.text[:200]
        )
        return False

    def push_monitoring_fault(
        self,
        fault_uuid: str,
        status: str,
        fault_type: str,
        driver_id: Optional[int],
        entry_level: str,
        started_at: datetime,
        gap_s: float,
        last_known: Optional[Dict[str, Any]] = None,
        resolved_at: Optional[datetime] = None,
        resolution: Optional[str] = None,
        blocking: bool = False,
    ) -> bool:
        """
        Open, refresh or resolve a monitoring fault (driver not observable).

        ``POST /monitoring-faults``

        Not a fatigue event: the driver must not be accused of fatigue for
        something the camera failed to see. Both fault types share one body
        shape (see :data:`MONITORING_FAULT_SEVERITY`):

        * ``no_face`` - a gap from ALERT / WARNING crossed ``NO_FACE_FAULT_S``.
        * ``danger_latched`` - a gap from DANGER passed ``NO_FACE_HOLD_S``;
          the unit is holding DANGER for a driver it can no longer see.

        Every POST for a fault carries the same client-generated
        ``fault_uuid`` and the backend upserts on it: ``"open"`` when the
        fault begins, ``"open"`` again every ``MONITORING_FAULT_REFRESH_S``
        with an updated ``gap_s``, and ``"resolved"`` once. Keyed that way
        the POSTs can go fire-and-forget from threads (no id round trip) and
        a retried POST is idempotent. The portal should derive live duration
        from ``started_at`` and use ``gap_s`` / ``timestamp`` of the latest
        refresh only as a liveness check against the device heartbeat.

        Args:
            fault_uuid: Identifies the fault across all its POSTs.
            status: ``"open"`` or ``"resolved"``.
            fault_type: ``"no_face"`` or ``"danger_latched"``.
            driver_id: Driver being monitored, or ``None`` if unrecognised.
            entry_level: Level the gap began at.
            started_at: When the face was lost (UTC), not when the fault
                opened.
            gap_s: Seconds of continuous no-face as of this POST.
            last_known: :meth:`FrameMetrics.last_known` of the last scored
                frame before the gap, or ``None`` if there was none.
            resolved_at: When the fault ended (UTC); ``"resolved"`` only.
            resolution: Why it ended - ``"face_reacquired"`` or
                ``"ignition_off"``; ``"resolved"`` only.
            blocking: If ``True``, send synchronously (tests).

        Returns:
            Non-blocking: ``True`` if queued. Blocking: ``True`` on 2xx.
        """
        body = {
            "fault_uuid": fault_uuid,
            "status": status,
            "fault_type": fault_type,
            "severity": MONITORING_FAULT_SEVERITY[fault_type],
            "device_id": config.DEVICE_ID,
            "driver_id": int(driver_id) if driver_id is not None else None,
            "phase": Phase.MONITORING.value,
            "entry_level": entry_level,
            "started_at": started_at.isoformat(),
            "gap_s": round(float(gap_s), 1),
            "last_known": last_known,
            "resolved_at": resolved_at.isoformat() if resolved_at else None,
            "resolution": resolution,
            "duration_s": (round((resolved_at - started_at).total_seconds(), 1)
                           if resolved_at else None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if blocking:
            return self._send_monitoring_fault(body)
        threading.Thread(
            target=self._send_monitoring_fault, args=(body,),
            name="monitoring-fault", daemon=True,
        ).start()
        return True

    def _send_monitoring_fault(self, body: Dict[str, Any]) -> bool:
        """POST one monitoring-fault body; log and return the outcome."""
        resp = self._request("POST", "/monitoring-faults", json=body)
        if resp is None:
            return False
        if resp.status_code in _CREATED_OK:
            logger.info("Monitoring fault %s %s %s gap=%.1fs (driver %s)",
                        body["fault_type"], body["fault_uuid"], body["status"],
                        body["gap_s"], body["driver_id"])
            return True
        logger.error("Monitoring fault rejected: HTTP %s %s",
                     resp.status_code, resp.text[:200])
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
            "microsleep_override_frames": int(result["microsleep_override_frames"]),
            "yawns": int(result["yawns"]),
            "microsleeps": int(result["microsleeps"]),
            "failed_on_microsleep": bool(result["failed_on_microsleep"]),
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

    # ------------------------------------------------------------------
    # Device heartbeat
    # ------------------------------------------------------------------

    def post_heartbeat(
        self,
        relay_state: Optional[str],
        device_id: Optional[str] = None,
        firmware_version: Optional[str] = None,
        blocking: bool = False,
    ) -> bool:
        """
        Tell the portal this unit is online and what the relay actually is.

        ``POST /devices/{device_id}/heartbeat``

        ``confirmed_state`` must be the relay state *as driven* by
        ``AlertManager`` (``get_relay_state()``), not what was commanded:
        the portal compares it with its own ``commanded_state`` to spot a
        unit that has not applied a command. The mapping is
        :data:`RELAY_CONFIRMED_STATE` (``UNLOCKED`` -> ``normal``,
        ``LOCKED`` -> ``interrupted``); ``None`` -> ``unknown``.

        The response carries the portal's ``commanded_state``. It is only
        logged at DEBUG for now - nothing acts on it yet.

        Sent from the main loop every ``config.HEARTBEAT_INTERVAL_SECONDS``
        in both phases. By default the request runs on a daemon thread and
        this returns immediately. An unreachable backend is logged at DEBUG
        only (it recurs every 30 s and ``ping()`` already warned at
        start-up); recovery is logged once at INFO, and a rejected body
        (HTTP 4xx/5xx - a contract problem, not an outage) once at WARNING.

        Args:
            relay_state: ``"LOCKED"`` / ``"UNLOCKED"`` from
                ``AlertManager.get_relay_state()``, or ``None`` if the
                relay has not been driven yet (reported as ``unknown``).
            device_id: Defaults to ``config.DEVICE_ID``.
            firmware_version: Defaults to ``config.FIRMWARE_VERSION``.
            blocking: If ``True``, send synchronously and return the real
                outcome (tests).

        Returns:
            Non-blocking: ``True`` if the request was queued.
            Blocking: ``True`` on HTTP 2xx, ``False`` otherwise.
        """
        path = f"/devices/{device_id or config.DEVICE_ID}/heartbeat"
        body = {
            "confirmed_state": RELAY_CONFIRMED_STATE.get(
                str(relay_state).upper() if relay_state is not None else "",
                CONFIRMED_STATE_UNKNOWN,
            ),
            "firmware_version": firmware_version or config.FIRMWARE_VERSION,
        }
        if blocking:
            return self._send_heartbeat(path, body)

        threading.Thread(
            target=self._send_heartbeat, args=(path, body),
            name="heartbeat", daemon=True,
        ).start()
        return True

    def _send_heartbeat(self, path: str, body: Dict[str, Any]) -> bool:
        """POST one heartbeat; log only transitions (see :meth:`post_heartbeat`)."""
        was_ok = self._heartbeat_ok
        resp = self._request("POST", path, json=body, error_level=logging.DEBUG)
        ok = resp is not None and resp.status_code in _HEARTBEAT_OK
        self._heartbeat_ok = ok

        if ok:
            # Not acted on yet - surfaced at DEBUG so it can be seen arriving.
            payload = self._unwrap(self._safe_json(resp))
            commanded = payload.get("commanded_state") if isinstance(payload, dict) else None
            if was_ok is False:
                logger.info("Heartbeat restored: backend reachable again "
                            "(confirmed=%s, commanded=%r)", body["confirmed_state"], commanded)
            else:
                logger.debug("Heartbeat sent (confirmed=%s, commanded=%r)",
                             body["confirmed_state"], commanded)
        elif resp is None:
            # Transport failure - _request already logged the cause at DEBUG.
            logger.debug("Heartbeat not delivered: backend unreachable")
        else:
            # Reachable but refused: one WARNING, then DEBUG until it clears.
            logger.log(
                logging.DEBUG if was_ok is False else logging.WARNING,
                "Heartbeat rejected: HTTP %s %s", resp.status_code, resp.text[:200],
            )
        return ok
