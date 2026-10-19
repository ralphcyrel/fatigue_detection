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
and ``frs_scored`` = ``max(frs_raw, DANGER_THRESHOLD)`` while the level is
DANGER for a reason other than the FRS band (a microsleep, or a head-pose
alert that has escalated through WARNING), else ``frs_raw``. A sustained nod
during the assessment therefore fails it, and the raw series is still
available for analysis.

Valid frames (2026-10-15)
-------------------------
Frames with no face or with invalid landmarks (``pipeline.landmark_problem``)
are counted, by reason, but never scored - never treated as closed eyes. If
fewer than ``config.PREDRIVE_MIN_VALID_FRACTION`` of the assessment's frames
were valid, the assessment is void: an *invalid assessment* (FAULT), not a
fatigue verdict. ``result()`` reports it as ``invalid_cause`` =
``"INVALID_ASSESSMENT"`` with ``valid_fraction``; the backend still receives
it as ``void`` (no API change).

Every sample is kept and written to ``logs/assessments/<device>_<driver>_
<timestamp>.csv`` (plus a ``.json`` summary) and returned in
:meth:`PredriveAssessment.result` so the backend receives the full series.

Rolling assessment (2026-10-19)
-------------------------------
``duration_s`` (30 s) is a *minimum*, not the length of the run. The run
goes on until the driver confirms the start (or the session ends), and:

* once it has lasted ``duration_s`` (:meth:`is_complete`), its first
  ``duration_s`` are judged by :meth:`result` exactly as the fixed run was,
  and the caller :meth:`freeze`\\ s that verdict - it is what goes to the
  backend. Later frames are no longer added to ``samples`` (memory stays
  bounded however long the driver waits); after :meth:`write_files` and
  :meth:`stream_to_csv` they go straight to the CSV instead.
* :meth:`window_verdict` applies the same pass rule - valid fraction,
  microsleeps, worst ``worst_window_s`` mean of ``frs_scored`` - to the most
  recent ``duration_s`` only. It is evaluated every frame, so it caches the
  mean of each window once the window is complete (the identical
  ``f[i:j].mean()`` the fixed run computes) instead of rescanning.
* :meth:`mark` records the READY_TO_START entries / losses, the start
  confirmation and the end of the run in ``timeline``, which the JSON
  summary carries for reporting the real pre-drive duration of each trial.

Clocks (2026-10-19)
-------------------
Every ``now`` here is ``time.monotonic()``: the 30 s minimum, the rolling
window, the worst-window scan and every duration in the timeline are
measured on it, so a wall-clock step (an NTP sync after a cold boot - the Pi
has no RTC) cannot end the assessment early or hold it open. The wall clock
is only *recorded*: ``t_abs`` of each sample (the frame's ``FrameMetrics.t``),
``started_at`` / ``ended_at`` (sent to the backend), the timeline's ``t_abs``
and the file names.
"""

import bisect
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
    "ear_norm", "ear_norm_frs", "bd_norm", "bf_norm", "perclos_norm", "mar_norm", "yawn_norm",
    "ear_excess", "blink_duration_excess", "blink_frequency_excess", "perclos_excess",
    "yawn_excess", "microsleep_excess", "frs_raw", "frs_scored", "level",
    "pitch", "yaw", "roll", "pose_alert", "pose_override", "yawn_state",
    "microsleep_state", "microsleep_override", "microsleep_confirmed",
]

# window_verdict(): once this many samples have fallen out of the rolling
# window, they are dropped from its arrays (the frozen ``samples`` keep the
# first window regardless).
_PRUNE_AT: int = 2000


class PredriveAssessment:
    """
    Accumulate per-frame metrics and decide pass / fail - on the first
    ``duration_s`` once (:meth:`result`), then on a rolling window
    (:meth:`window_verdict`).

    Typical usage::

        assessment = PredriveAssessment()
        assessment.start(now)
        while not assessment.is_complete(now):
            ...                                   # pipeline.process(...)
            assessment.add(metrics, now)          # or assessment.add_no_face(now)
        result = assessment.result()
        assessment.freeze()
        assessment.write_files(device_id, driver_id)
        assessment.stream_to_csv()
        while waiting_for_the_start:
            assessment.add(metrics, now)
            ready = assessment.window_verdict(now)["passed"]
        assessment.finish("confirmed", now)

    Args:
        duration_s: Minimum length of the run, and the rolling window.
        worst_window_s: Length of the rolling window for the verdict.
        pass_threshold: ``worst_window_mean`` must be strictly below this.
        min_valid_fraction: Void the assessment if fewer than this fraction
            of frames had a face with valid landmarks.
    """

    def __init__(
        self,
        duration_s: float = config.PREDRIVE_ASSESSMENT_SECONDS,
        worst_window_s: float = config.PREDRIVE_WORST_WINDOW_SECONDS,
        pass_threshold: float = WARNING_THRESHOLD,
        min_valid_fraction: float = config.PREDRIVE_MIN_VALID_FRACTION,
    ) -> None:
        self.duration_s = float(duration_s)
        self.worst_window_s = float(worst_window_s)
        self.pass_threshold = float(pass_threshold)
        self.min_valid_fraction = float(min_valid_fraction)

        # time.monotonic() of start() / of completion (0 = not yet), and the
        # wall clock at the same two moments, for the record.
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.started_wall: float = 0.0
        self.ended_wall: float = 0.0
        self.samples: List[Dict[str, Any]] = []
        # time.monotonic() of each entry of ``samples``.
        self._sample_mono: List[float] = []
        # Frames not scored (no face, invalid landmarks), in total and by reason.
        self.no_face_frames: int = 0
        self.invalid_reasons: Dict[str, int] = {}
        self._started = False
        # Which calibration the run was scored against (api.Calibration
        # .provenance()); set by the caller, echoed in result() so it lands in
        # the JSON summary and POST /assessments.
        self.calibration: Dict[str, Any] = {}
        # When the pre-drive cycle began (recognition), set by the caller:
        # the start of the trial's pre-drive duration. Monotonic for the
        # duration, wall clock for the record.
        self.predrive_started_mono: Optional[float] = None
        self.predrive_started_at: Optional[float] = None
        self.timeline: List[Dict[str, Any]] = []
        self._reset_rolling()
        self._files: Optional[Dict[str, Any]] = None

    def _reset_rolling(self) -> None:
        self._frozen = False
        # Rolling-window arrays (every frame, frozen or not): sample times,
        # frs_scored, times of confirmed microsleeps, times of invalid frames.
        self._t: List[float] = []
        self._f: List[float] = []
        self._ms_t: List[float] = []
        self._invalid_t: List[float] = []
        # Mean frs_scored of the complete worst-window starting at sample i,
        # for every i < _next_open; _next_open is the first window still open.
        self._win_means: List[float] = []
        self._next_open = 0
        self._csv_fh = None
        self._csv_writer = None
        # Frame loop health: the longest gap between two frames handed to
        # add() / add_invalid(), on the monotonic clock.
        self.max_frame_gap_s = 0.0
        self._last_frame_t: Optional[float] = None

    def _frame_at(self, t: float) -> None:
        if self._last_frame_t is not None:
            self.max_frame_gap_s = max(self.max_frame_gap_s, t - self._last_frame_t)
        self._last_frame_t = t

    # ------------------------------------------------------------------

    def start(self, now: Optional[float] = None) -> None:
        """Reset and open the window at ``now`` (``time.monotonic()``)."""
        self.close_stream()
        self.start_time = time.monotonic() if now is None else now
        self.started_wall = time.time()
        self.end_time = self.ended_wall = 0.0
        self.samples = []
        self._sample_mono = []
        self.no_face_frames = 0
        self.invalid_reasons = {}
        self.timeline = []
        self._reset_rolling()
        self._files = None
        self._started = True

    def progress(self, now: Optional[float] = None) -> float:
        """Fraction of the window elapsed, clamped to ``[0, 1]``."""
        if not self._started or self.duration_s <= 0:
            return 1.0 if self._started else 0.0
        t = time.monotonic() if now is None else now
        return float(min(max((t - self.start_time) / self.duration_s, 0.0), 1.0))

    def seconds_remaining(self, now: Optional[float] = None) -> float:
        """Seconds until the window closes (0 once complete)."""
        t = time.monotonic() if now is None else now
        return max(0.0, self.duration_s - (t - self.start_time))

    def is_complete(self, now: Optional[float] = None) -> bool:
        """Whether the window has elapsed."""
        if not self._started:
            return False
        done = self.progress(now) >= 1.0
        if done and self.end_time == 0.0:
            self.end_time = time.monotonic() if now is None else now
            self.ended_wall = time.time()
        return done

    def add(self, m: FrameMetrics, now: Optional[float] = None) -> None:
        """
        Record one processed frame.

        Args:
            m: Pipeline output for the frame.
            now: Frame time (``time.monotonic()``); defaults to ``m.t_mono``.
                The sample's ``t_abs`` is the frame's wall clock, ``m.t``.
        """
        if not self._started:
            raise RuntimeError("PredriveAssessment.add() called before start()")
        t = m.t_mono if now is None else now
        frs_raw = m.frs
        # DANGER held by the FRS band alone scores the FRS itself; DANGER from
        # a microsleep or an escalated head-pose alert scores at least the
        # DANGER threshold, as the overrides always did.
        forced = m.level == "DANGER" and m.level_reason != "BAND"
        frs_scored = max(frs_raw, DANGER_THRESHOLD) if forced else frs_raw
        comps = m.frs_result.get("components", {})
        pose = m.pose or {}
        sample = {
            "t_rel": round(t - self.start_time, 3),
            "t_abs": m.t,
            "ear": m.ear, "mar": m.mar, "perclos": m.perclos,
            "blink_duration_ms": m.blink_duration_ms, "blink_freq": m.blink_freq,
            "ear_norm": m.ear_norm, "ear_norm_frs": m.ear_norm_frs,
            "bd_norm": m.bd_norm, "bf_norm": m.bf_norm,
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
        }
        if not self._frozen:
            self.samples.append(sample)
            self._sample_mono.append(t)
        elif self._csv_writer is not None:
            self._csv_writer.writerow({k: sample.get(k) for k in SAMPLE_FIELDS})
        self._push(t, frs_scored, sample["microsleep_confirmed"])
        self._frame_at(t)

    def add_invalid(self, reason: str = "no_face", now: Optional[float] = None) -> None:
        """
        Record a frame that could not be measured - no face, or invalid
        landmarks (``reason`` from ``pipeline.landmark_problem``). Counted,
        never scored. Counts in :meth:`result` stop at :meth:`freeze`; the
        rolling window counts it by ``now`` (``time.monotonic()``; the clock
        if omitted).
        """
        t = time.monotonic() if now is None else now
        self._invalid_t.append(t)
        self._frame_at(t)
        if self._frozen:
            return
        self.no_face_frames += 1
        self.invalid_reasons[reason] = self.invalid_reasons.get(reason, 0) + 1

    def add_no_face(self, now: Optional[float] = None) -> None:
        """Record a frame in which no face was found (counted, not scored)."""
        self.add_invalid("no_face", now)

    # ------------------------------------------------------------------

    def worst_window_mean(self) -> float:
        """
        Maximum over all windows of length ``worst_window_s`` of the mean
        ``frs_scored``, using sample timestamps (the loop rate varies) - the
        monotonic ones recorded by :meth:`add`, or ``t_abs`` for a sample list
        built elsewhere (a replay from a CSV).

        Returns:
            ``0.0`` if there are no samples.
        """
        if not self.samples:
            return 0.0
        times = (self._sample_mono if len(self._sample_mono) == len(self.samples)
                 else [s["t_abs"] for s in self.samples])
        t = np.array(times, dtype=float)
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

    # ---- rolling window (2026-10-19) ---------------------------------

    def _push(self, t: float, frs_scored: float, microsleep: bool) -> None:
        """Add one scored frame to the rolling arrays, closing finished windows."""
        self._t.append(t)
        self._f.append(frs_scored)
        if microsleep:
            self._ms_t.append(t)
        # A window starting at sample i is complete at the first sample more
        # than worst_window_s after it - this one, since earlier frames were
        # checked as they arrived - and spans [i, newest): the same j and the
        # same f[i:j].mean() as worst_window_mean()'s scan.
        newest = len(self._t) - 1
        while self._next_open < newest and t - self._t[self._next_open] > self.worst_window_s:
            i = self._next_open
            self._win_means.append(float(np.asarray(self._f[i:newest], dtype=float).mean()))
            self._next_open += 1

    def _prune(self, cutoff: float) -> None:
        """Drop rolling samples older than ``cutoff`` once enough have piled up."""
        s = bisect.bisect_left(self._t, cutoff)
        if s >= _PRUNE_AT and s <= self._next_open:
            del self._t[:s], self._f[:s], self._win_means[:s]
            self._next_open -= s
        for times in (self._ms_t, self._invalid_t):
            k = bisect.bisect_left(times, cutoff)
            if k >= _PRUNE_AT:
                del times[:k]

    def elapsed(self, now: Optional[float] = None) -> float:
        """Seconds since :meth:`start` (``now``: ``time.monotonic()``)."""
        return (time.monotonic() if now is None else now) - self.start_time

    def window_verdict(self, now: Optional[float] = None) -> Dict[str, Any]:
        """
        The pass rule on the most recent ``duration_s``: frames with
        ``t >= now - duration_s``. Same aggregate as :meth:`worst_window_mean`
        (the worst window is computed exactly as there, over this subset),
        same valid-fraction and microsleep conditions as :meth:`result`.

        Returns:
            ``passed``, ``reason`` (``None`` on a pass, else ``"microsleep"``
            > ``"invalid"`` > ``"frs"`` - the hard-failure priority),
            ``eligible`` (the run has lasted ``duration_s``),
            ``worst_window_mean``, ``valid_fraction``, ``microsleeps``,
            ``n_samples`` and ``n_invalid`` in the window.
        """
        now = time.monotonic() if now is None else now
        cutoff = now - self.duration_s
        self._prune(cutoff)
        s = bisect.bisect_left(self._t, cutoff)
        n = len(self._t)
        valid = n - s
        invalid = len(self._invalid_t) - bisect.bisect_left(self._invalid_t, cutoff)
        total = valid + invalid
        valid_fraction = valid / total if total else 0.0
        worst = 0.0
        if valid:
            # Every window from s up to the first one still open, which (as
            # in the scan) is the last one counted: it runs to the newest.
            first_open = max(self._next_open, s)
            worst = float(np.asarray(self._f[first_open:n], dtype=float).mean())
            if s < self._next_open:
                worst = max(worst, max(self._win_means[s:self._next_open]))
            worst = round(worst, 6)
        microsleeps = len(self._ms_t) - bisect.bisect_left(self._ms_t, cutoff)
        void = valid == 0 or valid_fraction < self.min_valid_fraction
        if microsleeps:
            reason: Optional[str] = "microsleep"
        elif void:
            reason = "invalid"
        elif worst >= self.pass_threshold:
            reason = "frs"
        else:
            reason = None
        return {
            "passed": reason is None,
            "reason": reason,
            "eligible": self._started and now - self.start_time >= self.duration_s,
            "worst_window_mean": worst,
            "valid_fraction": round(valid_fraction, 4),
            "microsleeps": microsleeps,
            "n_samples": valid,
            "n_invalid": invalid,
        }

    def freeze(self, now: Optional[float] = None) -> None:
        """
        Fix :meth:`result` on the frames so far (the first ``duration_s``).
        Later frames feed only the rolling window (and the CSV stream).
        """
        if not self._frozen:
            self.is_complete(now)        # stamps end_time
            self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    def mark(self, event: str, now: Optional[float] = None, **detail: Any) -> None:
        """
        Record a pre-drive milestone in ``timeline`` (``ready_entered``,
        ``ready_lost``, ``confirmed``, ``ended``) at ``now``
        (``time.monotonic()``) and refresh the JSON summary if it has been
        written. ``t_abs`` is the wall clock, for the record.
        """
        t = time.monotonic() if now is None else now
        self.timeline.append({"event": event, "t_mono": t, "t_abs": time.time(),
                              "t_rel": round(t - self.start_time, 3), **detail})
        self._write_summary()

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
        """Fraction of all frames (scored + unscored) that could not be measured."""
        total = len(self.samples) + self.no_face_frames
        return self.no_face_frames / total if total else 1.0

    def valid_fraction(self) -> float:
        """Fraction of all frames that had a face with valid landmarks."""
        return 1.0 - self.no_face_fraction()

    def is_void(self) -> bool:
        """``True`` if too few valid frames to trust the verdict (invalid assessment)."""
        return not self.samples or self.valid_fraction() < self.min_valid_fraction

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
            "started_at": self.started_wall,
            "ended_at": self.ended_wall or self.started_wall + self.duration_s,
            "duration_s": self.duration_s,
            "n_samples": len(self.samples),
            "n_no_face": self.no_face_frames,
            "no_face_fraction": round(self.no_face_fraction(), 4),
            "valid_fraction": round(self.valid_fraction(), 4),
            "min_valid_fraction": self.min_valid_fraction,
            "invalid_reasons": dict(self.invalid_reasons),
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
            # Local record of why a void run was void (the backend gets "void").
            "invalid_cause": "INVALID_ASSESSMENT" if void else None,
            "passed": (not void) and worst < self.pass_threshold and microsleeps == 0,
            "calibration": dict(self.calibration),
            # Everything above is the first duration_s (the backend's
            # record). The rest of the run, local only (2026-10-19):
            **self.timeline_summary(),
            "samples": list(self.samples),
        }

    def timeline_summary(self) -> Dict[str, Any]:
        """
        When the trial reached READY_TO_START and when the start was
        confirmed, for reporting the actual pre-drive duration.

        ``predrive_duration_s`` runs from the start of the pre-drive cycle
        (recognition; else the assessment) to the start confirmation, and is
        ``None`` if the start was never confirmed.
        """
        def first(event: str) -> Optional[Dict[str, Any]]:
            return next((e for e in self.timeline if e["event"] == event), None)

        def wall(e: Optional[Dict[str, Any]]) -> Optional[float]:
            return None if e is None else e["t_abs"]

        # Durations on the monotonic clock; the *_at fields are wall clock.
        began = (self.start_time if self.predrive_started_mono is None
                 else self.predrive_started_mono)
        ready, confirmed = first("ready_entered"), first("confirmed")
        ended = next((e for e in reversed(self.timeline) if e["event"] == "ended"), None)
        return {
            "min_duration_s": self.duration_s,
            "predrive_started_at": self.predrive_started_at or self.started_wall,
            "first_ready_at": wall(ready),
            "confirmed_at": wall(confirmed),
            "time_to_first_ready_s": (None if ready is None
                                      else round(ready["t_mono"] - began, 3)),
            "ready_entries": sum(1 for e in self.timeline if e["event"] == "ready_entered"),
            "predrive_duration_s": (None if confirmed is None
                                    else round(confirmed["t_mono"] - began, 3)),
            "end_reason": ended.get("reason") if ended else None,
            # Longest wait between two frames of this run (monotonic).
            "max_frame_gap_s": round(self.max_frame_gap_s, 3),
            "timeline": [dict(e) for e in self.timeline],
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
        Later :meth:`mark` / :meth:`finish` calls rewrite the summary.

        Returns:
            Path of the CSV file.
        """
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(self.started_wall))
        stem = f"{device_id}_{driver_id if driver_id is not None else 'unknown'}_{stamp}"
        csv_path = directory / f"{stem}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=SAMPLE_FIELDS)
            writer.writeheader()
            for s in self.samples:
                writer.writerow({k: s.get(k) for k in SAMPLE_FIELDS})
        self._files = {"csv": csv_path, "json": directory / f"{stem}.json",
                       "device_id": device_id, "driver_id": driver_id}
        self._write_summary()
        logger.info("Assessment data written: %s (%d samples)", csv_path, len(self.samples))
        return csv_path

    def _write_summary(self) -> None:
        """(Re)write the JSON summary, if :meth:`write_files` has run."""
        if self._files is None:
            return
        summary = {k: v for k, v in self.result().items() if k != "samples"}
        summary["device_id"] = self._files["device_id"]
        summary["driver_id"] = self._files["driver_id"]
        summary["csv"] = self._files["csv"].name
        try:
            self._files["json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("Could not update assessment summary %s: %s", self._files["json"], exc)

    def stream_to_csv(self) -> None:
        """
        After :meth:`freeze` and :meth:`write_files`: append every later
        frame to the CSV as it is added (line-buffered), so a run of any
        length is recorded without keeping it in memory. No-op if the files
        were not written.
        """
        if self._files is None or self._csv_fh is not None:
            return
        try:
            self._csv_fh = self._files["csv"].open("a", newline="", encoding="utf-8",
                                                   buffering=1)
        except OSError as exc:
            logger.error("Could not reopen %s for the rest of the run: %s",
                         self._files["csv"], exc)
            return
        self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=SAMPLE_FIELDS)

    def close_stream(self) -> None:
        """Close the CSV stream (idempotent)."""
        fh = getattr(self, "_csv_fh", None)
        if fh is not None:
            fh.close()
        self._csv_fh = self._csv_writer = None

    @property
    def files_written(self) -> bool:
        return self._files is not None

    def finish(self, reason: str, now: Optional[float] = None, **detail: Any) -> None:
        """End of the run: mark ``ended`` with ``reason``, close the stream, update the summary."""
        self.close_stream()
        self.mark("ended", now, reason=reason, **detail)
