"""
Per-frame FRS debug log (``config.DEBUG_FRS_CSV``; off by default).

A diagnostic, not results data: one CSV row per processed frame, in both
phases, with the raw metrics, every normalised input and weighted FRS term,
the score, the effective level and *why* the level is what it is (the FRS
band, or the override forcing DANGER). It exists to answer "why did the level
say DANGER here?" frame by frame, which the assessment CSVs (pre-drive only,
no reason column) and the session log (level changes only) cannot.

Written to ``logs/debug_frs_<UTC stamp>.csv``, one file per process; the file
is created on the first row, so a run with the flag on but no face never
leaves an empty file behind. Line-buffered, so a crash or power cut loses at
most the row being written.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config import config

logger = logging.getLogger(__name__)

# Column order. ``*_norm`` are the normalised FRS inputs (1.0 = the driver's
# calibrated baseline; ``ear_norm_frs`` is the rolling median the FRS sees),
# ``c_*`` the weighted excess terms, which sum to ``frs``. ``d_pitch`` is the
# pitch relative to the driver's neutral pose (``neutral`` names its source),
# ``esc_*`` the escalation timers (modules.pipeline.LevelEscalator.timers).
# ``reason`` is the level reason for a scored frame, else why the frame was
# not scored (NO_FACE, INVALID_LANDMARKS:<problem>, SETTLING) or a verdict.
FIELDS = [
    "timestamp", "dt_ms", "phase", "face_valid",
    "ear", "perclos", "blink_dur_ms", "blink_freq", "mar", "pitch", "yaw", "roll",
    "d_pitch", "neutral",
    "ear_norm_frs", "perclos_norm", "bd_norm", "bf_norm", "yawn_norm", "microsleep_count",
    "c_ear", "c_perclos", "c_blink_dur", "c_blink_freq", "c_yawn", "c_microsleep",
    "frs", "raw_band", "band", "level", "reason",
    "esc_warn_s", "esc_danger_s", "esc_calm_s",
    "pose_alert", "pose_override", "microsleep_state", "closed_s", "microsleep_override",
]


def _r(value: Any, ndigits: int = 4) -> Any:
    """Round a number for the CSV; pass anything else (None, str) through."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return round(float(value), ndigits)


class FrsDebugLog:
    """
    Append one row per frame to ``logs/debug_frs_<stamp>.csv``.

    Every method is a no-op when ``enabled`` is false, so callers need no
    guard of their own.

    Args:
        enabled: Defaults to ``config.DEBUG_FRS_CSV``.
        directory: Where the file goes (``config.LOGS_DIR``).
    """

    def __init__(self, enabled: Optional[bool] = None,
                 directory: Path = config.LOGS_DIR) -> None:
        self.enabled = config.DEBUG_FRS_CSV if enabled is None else bool(enabled)
        self.directory = directory
        self.path: Optional[Path] = None
        self._fh: Any = None
        self._writer: Optional[csv.DictWriter] = None
        self._last_t: Optional[float] = None   # time.monotonic() of the last row

    def _open(self, now: float) -> bool:
        """Create the file on the first row. Returns False (and disables) on error."""
        if self._writer is not None:
            return True
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
            self.path = self.directory / f"debug_frs_{stamp}.csv"
            self._fh = self.path.open("w", newline="", encoding="utf-8", buffering=1)
            self._writer = csv.DictWriter(self._fh, fieldnames=FIELDS)
            self._writer.writeheader()
        except OSError as exc:
            logger.error("FRS debug CSV disabled - could not open %s: %s", self.path, exc)
            self.enabled = False
            return False
        logger.info("FRS debug CSV: %s", self.path)
        return True

    def _write(self, row: Dict[str, Any], now: float) -> None:
        if not self.enabled or not self._open(now):
            return
        row["timestamp"] = round(now, 3)            # wall clock: for reading only
        # The gap since the previous row on the monotonic clock (2026-10-19),
        # so a wall-clock step never shows up as a frame gap.
        mono = time.monotonic()
        row["dt_ms"] = None if self._last_t is None else round((mono - self._last_t) * 1000.0, 1)
        self._last_t = mono
        try:
            self._writer.writerow(row)  # type: ignore[union-attr]
        except OSError as exc:
            logger.error("FRS debug CSV disabled - write failed: %s", exc)
            self.close()
            self.enabled = False

    def log_frame(self, m: Any, phase: str, neutral: str = "") -> None:
        """
        Record one processed frame.

        Args:
            m: The frame's ``FrameMetrics``.
            phase: Free-form phase label, e.g. ``"predrive_assessing"``.
            neutral: Source of the neutral head pose (``stored`` /
                ``settling`` / ``seeded``; empty = none yet).
        """
        if not self.enabled:
            return
        comps = m.frs_result.get("components", {})
        pose = m.pose or {}
        ms = m.microsleep_status or {}
        esc = m.escalation or {}
        self._write({
            "phase": phase, "face_valid": True,
            "ear": _r(m.ear), "perclos": _r(m.perclos, 3),
            "blink_dur_ms": _r(m.blink_duration_ms, 1), "blink_freq": _r(m.blink_freq, 1),
            "mar": _r(m.mar), "pitch": _r(pose.get("pitch"), 2),
            "yaw": _r(pose.get("yaw"), 2), "roll": _r(pose.get("roll"), 2),
            "d_pitch": _r(pose.get("d_pitch"), 2), "neutral": neutral,
            "ear_norm_frs": _r(m.ear_norm_frs), "perclos_norm": _r(m.perclos_norm),
            "bd_norm": _r(m.bd_norm), "bf_norm": _r(m.bf_norm), "yawn_norm": _r(m.yawn_norm),
            "microsleep_count": m.microsleep_count,
            "c_ear": _r(comps.get("ear_excess")),
            "c_perclos": _r(comps.get("perclos_excess")),
            "c_blink_dur": _r(comps.get("blink_duration_excess")),
            "c_blink_freq": _r(comps.get("blink_frequency_excess")),
            "c_yawn": _r(comps.get("yawn_excess")),
            "c_microsleep": _r(comps.get("microsleep_excess")),
            "frs": _r(m.frs), "raw_band": m.frs_result.get("level"), "band": m.band,
            "level": m.level, "reason": m.level_reason,
            "esc_warn_s": esc.get("warn_s"), "esc_danger_s": esc.get("danger_s"),
            "esc_calm_s": esc.get("calm_s"),
            "pose_alert": bool(pose.get("alert", False)), "pose_override": m.pose_override,
            "microsleep_state": ms.get("state", ""), "closed_s": _r(ms.get("closed_s"), 3),
            "microsleep_override": m.microsleep_override,
        }, m.t)

    def log_unscored(self, now: float, phase: str, face_valid: bool, level: str,
                     reason: str, pose: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a frame that was not scored: no face, invalid landmarks, or a
        valid frame during settling.

        Args:
            now: ``time.time()`` of the frame.
            phase: As for :meth:`log_frame`.
            face_valid: Whether the face and landmarks were usable.
            level: The level driven for the frame.
            reason: E.g. ``"NO_FACE"``, ``"INVALID_LANDMARKS:outside_frame"``,
                ``"SETTLING"``; monitoring appends the no-face band.
            pose: Head pose, if one was estimated (settling).
        """
        if not self.enabled:
            return
        pose = pose or {}
        self._write({"phase": phase, "face_valid": face_valid, "level": level,
                     "reason": reason, "pitch": _r(pose.get("pitch"), 2),
                     "yaw": _r(pose.get("yaw"), 2), "roll": _r(pose.get("roll"), 2)}, now)

    def log_event(self, now: float, phase: str, level: str, reason: str) -> None:
        """Record a non-frame event, e.g. the pre-drive verdict and its cause."""
        if not self.enabled:
            return
        self._write({"phase": phase, "level": level, "reason": reason}, now)

    def close(self) -> None:
        """Close the file (safe to call repeatedly)."""
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
        self._fh = None
        self._writer = None
