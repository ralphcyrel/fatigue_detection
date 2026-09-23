"""
Module 13 - Pre-drive fatigue assessment scoring.

At ignition-OFF the driver is observed for ``config.PREDRIVE_ASSESSMENT_SECONDS``
and every processed frame's :class:`~modules.pipeline.FrameMetrics` is
recorded here. When the window closes the verdict is taken on the **worst
rolling window** of the scored FRS series:

    worst_window_mean = max over t of mean(frs_scored[t, t + W])   (W = 5 s)
    passed            = worst_window_mean < WARNING_THRESHOLD (0.40)
                        and microsleeps == 0

Why this aggregate and not the mean or median
---------------------------------------------
The purpose of a pre-drive check is to catch a driver who is *already*
drowsy enough to microsleep. A 4-5 s eyes-closed episode inside 25 s of
otherwise alert behaviour is the strongest possible evidence of that, and
both the mean and the median dilute it - the median discards it entirely.
A single-frame maximum goes too far the other way: one mis-placed eyelid
landmark can spike the EAR term for a frame. A 5 s window needs a
*sustained* episode, which is what a microsleep is. The threshold reuses
the existing WARNING band edge (any window that would have shown a yellow
LED fails) so no new tunable is introduced.

Microsleep: an unconditional fail
--------------------------------
A single confirmed microsleep (:class:`~modules.blink.MicrosleepDetector`,
≥ 1.0 s eyes closed) fails the assessment outright, whatever
``worst_window_mean`` says. Pre-drive exists to decide whether someone should
start driving at all, and sleep intrusion while sitting still in a stationary
vehicle is disqualifying on its own - the aggregate is a measure of *risk*,
whereas a microsleep is the event the whole system is trying to prevent,
already happening. It is reported as ``failed_on_microsleep`` so the caller
can raise ``LockReason.MICROSLEEP_DETECTED`` rather than a generic fatigue
lock.

Per-frame scoring
-----------------
Each frame is recorded twice: ``frs_raw`` (the pipeline's eye + mouth FRS)
and ``frs_scored`` = ``max(frs_raw, DANGER_THRESHOLD)`` while a DANGER
override (head pose or microsleep) is active, else ``frs_raw``. A sustained
nod during the assessment therefore fails it, and the raw series is still
available for analysis. Frames with no face are counted but not scored; if
too many are missing the assessment is void.

Every sample is kept and written to ``logs/assessments/<device>_<driver>_
<timestamp>.csv`` (plus a ``.json`` summary) and returned in
:meth:`PredriveAssessment.result` so the backend receives the full series.
"""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import config
from modules.frs import DANGER_THRESHOLD, WARNING_THRESHOLD
from modules.pipeline import FrameMetrics

logger = logging.getLogger(__name__)

# Column order of the per-frame CSV; also the keys of each ``samples`` entry.
SAMPLE_FIELDS: List[str] = [
    "t_rel", "t_abs", "ear", "mar", "perclos", "blink_duration_ms", "blink_freq",
    "ear_norm", "bd_norm", "bf_norm", "perclos_norm", "mar_norm", "yawn_norm",
    "ear_excess", "blink_duration_excess", "blink_frequency_excess", "perclos_excess",
    "yawn_excess", "microsleep_excess", "frs_raw", "frs_scored", "level",
    "pitch", "yaw", "roll", "pose_alert", "pose_override", "yawn_state",
    "microsleep_state", "microsleep_override", "microsleep_confirmed",
]


class PredriveAssessment:
    """
    Accumulate per-frame metrics for a fixed window and decide pass / fail.

    Typical usage::

        assessment = PredriveAssessment()
        assessment.start(now)
        while not assessment.is_complete(now):
            ...                                   # pipeline.process(...)
            assessment.add(metrics, now)          # or assessment.add_no_face(now)
        result = assessment.result()
        assessment.write_files(device_id, driver_id)

    Args:
        duration_s: Length of the scored window.
        worst_window_s: Length of the rolling window for the verdict.
        pass_threshold: ``worst_window_mean`` must be strictly below this.
        max_no_face_fraction: Void the assessment if more than this fraction
            of frames had no face.
    """

    def __init__(
        self,
        duration_s: float = config.PREDRIVE_ASSESSMENT_SECONDS,
        worst_window_s: float = config.PREDRIVE_WORST_WINDOW_SECONDS,
        pass_threshold: float = WARNING_THRESHOLD,
        max_no_face_fraction: float = config.PREDRIVE_MAX_NO_FACE_FRACTION,
    ) -> None:
        self.duration_s = float(duration_s)
        self.worst_window_s = float(worst_window_s)
        self.pass_threshold = float(pass_threshold)
        self.max_no_face_fraction = float(max_no_face_fraction)

        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.samples: List[Dict[str, Any]] = []
        self.no_face_frames: int = 0
        self._started = False

    # ------------------------------------------------------------------

    def start(self, now: Optional[float] = None) -> None:
        """Reset and open the window at ``now`` (``time.time()`` style)."""
        self.start_time = time.time() if now is None else now
        self.end_time = 0.0
        self.samples = []
        self.no_face_frames = 0
        self._started = True

    def progress(self, now: Optional[float] = None) -> float:
        """Fraction of the window elapsed, clamped to ``[0, 1]``."""
        if not self._started or self.duration_s <= 0:
            return 1.0 if self._started else 0.0
        t = time.time() if now is None else now
        return float(min(max((t - self.start_time) / self.duration_s, 0.0), 1.0))

    def seconds_remaining(self, now: Optional[float] = None) -> float:
        """Seconds until the window closes (0 once complete)."""
        t = time.time() if now is None else now
        return max(0.0, self.duration_s - (t - self.start_time))

    def is_complete(self, now: Optional[float] = None) -> bool:
        """Whether the window has elapsed."""
        if not self._started:
            return False
        done = self.progress(now) >= 1.0
        if done and self.end_time == 0.0:
            self.end_time = time.time() if now is None else now
        return done

    def add(self, m: FrameMetrics, now: Optional[float] = None) -> None:
        """
        Record one processed frame.

        Args:
            m: Pipeline output for the frame.
            now: Frame time (``time.time()``); defaults to ``m.t``.
        """
        if not self._started:
            raise RuntimeError("PredriveAssessment.add() called before start()")
        t = m.t if now is None else now
        frs_raw = m.frs
        frs_scored = max(frs_raw, DANGER_THRESHOLD) if m.overridden else frs_raw
        comps = m.frs_result.get("components", {})
        pose = m.pose or {}
        self.samples.append({
            "t_rel": round(t - self.start_time, 3),
            "t_abs": t,
            "ear": m.ear, "mar": m.mar, "perclos": m.perclos,
            "blink_duration_ms": m.blink_duration_ms, "blink_freq": m.blink_freq,
            "ear_norm": m.ear_norm, "bd_norm": m.bd_norm, "bf_norm": m.bf_norm,
            "perclos_norm": m.perclos_norm, "mar_norm": m.mar_norm, "yawn_norm": m.yawn_norm,
            "ear_excess": comps.get("ear_excess", 0.0),
            "blink_duration_excess": comps.get("blink_duration_excess", 0.0),
            "blink_frequency_excess": comps.get("blink_frequency_excess", 0.0),
            "perclos_excess": comps.get("perclos_excess", 0.0),
            "yawn_excess": comps.get("yawn_excess", 0.0),
            "microsleep_excess": comps.get("microsleep_excess", 0.0),
            "frs_raw": frs_raw, "frs_scored": frs_scored, "level": m.level,
            "pitch": pose.get("pitch"), "yaw": pose.get("yaw"), "roll": pose.get("roll"),
            "pose_alert": bool(pose.get("alert", False)), "pose_override": bool(m.pose_override),
            "yawn_state": m.yawn_status.get("state", ""),
            "microsleep_state": m.microsleep_status.get("state", ""),
            "microsleep_override": bool(m.microsleep_override),
            # True only on the frame a microsleep was confirmed, so counting
            # these needs no edge detection.
            "microsleep_confirmed": m.microsleep_event is not None,
        })

    def add_no_face(self, now: Optional[float] = None) -> None:
        """Record a frame in which no face was found (counted, not scored)."""
        self.no_face_frames += 1

    # ------------------------------------------------------------------

    def worst_window_mean(self) -> float:
        """
        Maximum over all windows of length ``worst_window_s`` of the mean
        ``frs_scored``, using sample timestamps (the loop rate varies).

        Returns:
            ``0.0`` if there are no samples.
        """
        if not self.samples:
            return 0.0
        t = np.array([s["t_abs"] for s in self.samples], dtype=float)
        f = np.array([s["frs_scored"] for s in self.samples], dtype=float)
        if len(t) == 1 or t[-1] - t[0] <= self.worst_window_s:
            return round(float(f.mean()), 6)
        worst = 0.0
        j = 0
        for i in range(len(t)):
            # Advance j to the first sample outside [t[i], t[i] + W].
            while j < len(t) and t[j] - t[i] <= self.worst_window_s:
                j += 1
            worst = max(worst, float(f[i:j].mean()))
            if j >= len(t):
                break
        # Round away float noise (as frs.py does) so a window sitting exactly
        # on the pass threshold is judged correctly.
        return round(worst, 6)

    def _count_yawns(self) -> int:
        """Number of confirmed yawns = transitions into the ``yawning`` state."""
        count, prev = 0, ""
        for s in self.samples:
            if s["yawn_state"] == "yawning" and prev != "yawning":
                count += 1
            prev = s["yawn_state"]
        return count

    def _count_microsleeps(self) -> int:
        """Number of microsleeps confirmed during the assessment window."""
        return sum(1 for s in self.samples if s["microsleep_confirmed"])

    def no_face_fraction(self) -> float:
        """Fraction of all frames (scored + unscored) that had no face."""
        total = len(self.samples) + self.no_face_frames
        return self.no_face_frames / total if total else 1.0

    def is_void(self) -> bool:
        """``True`` if too few scored frames to trust the verdict."""
        return not self.samples or self.no_face_fraction() > self.max_no_face_fraction

    def result(self) -> Dict[str, Any]:
        """
        Summary + full per-frame series.

        Returns:
            A dict with ``passed``, ``void``, the aggregates, timing and
            ``samples`` (list of dicts keyed by :data:`SAMPLE_FIELDS`).
        """
        f = np.array([s["frs_scored"] for s in self.samples], dtype=float)
        raw = np.array([s["frs_raw"] for s in self.samples], dtype=float)
        worst = self.worst_window_mean()
        void = self.is_void()
        microsleeps = self._count_microsleeps()
        return {
            "started_at": self.start_time,
            "ended_at": self.end_time or self.start_time + self.duration_s,
            "duration_s": self.duration_s,
            "n_samples": len(self.samples),
            "n_no_face": self.no_face_frames,
            "no_face_fraction": round(self.no_face_fraction(), 4),
            "mean_frs": float(f.mean()) if len(f) else 0.0,
            "median_frs": float(np.median(f)) if len(f) else 0.0,
            "max_frs": float(f.max()) if len(f) else 0.0,
            "mean_frs_raw": float(raw.mean()) if len(raw) else 0.0,
            "worst_window_mean": worst,
            "worst_window_s": self.worst_window_s,
            "pass_threshold": self.pass_threshold,
            "pose_override_frames": int(sum(1 for s in self.samples if s["pose_override"])),
            "microsleep_override_frames": int(
                sum(1 for s in self.samples if s["microsleep_override"])),
            "yawns": self._count_yawns(),
            "microsleeps": microsleeps,
            # Reported separately from ``passed`` so the caller can raise a
            # distinguishable lock reason rather than a generic fatigue fail.
            "failed_on_microsleep": microsleeps > 0,
            "void": void,
            "passed": (not void) and worst < self.pass_threshold and microsleeps == 0,
            "samples": list(self.samples),
        }

    # ------------------------------------------------------------------

    def write_files(
        self,
        device_id: str,
        driver_id: Optional[int],
        directory: Path = config.ASSESSMENTS_DIR,
    ) -> Path:
        """
        Write ``<device>_<driver|unknown>_<UTC stamp>.csv`` (one row per
        frame) and a ``.json`` summary (without the samples) next to it.

        Returns:
            Path of the CSV file.
        """
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(self.start_time))
        stem = f"{device_id}_{driver_id if driver_id is not None else 'unknown'}_{stamp}"
        csv_path = directory / f"{stem}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=SAMPLE_FIELDS)
            writer.writeheader()
            for s in self.samples:
                writer.writerow({k: s.get(k) for k in SAMPLE_FIELDS})
        summary = {k: v for k, v in self.result().items() if k != "samples"}
        summary["device_id"] = device_id
        summary["driver_id"] = driver_id
        summary["csv"] = csv_path.name
        (directory / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Assessment data written: %s (%d samples)", csv_path, len(self.samples))
        return csv_path
