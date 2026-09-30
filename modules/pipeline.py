"""
Module 12 - Shared per-frame metrics pipeline.

Both operating phases (pre-drive assessment and continuous monitoring) turn
one frame's landmarks into one :class:`FrameMetrics` through exactly the same
code path::

    landmarks -> EAR / MAR -> blink / microsleep / PERCLOS / yawn
              -> normalise -> FRS -> head pose -> debounce
              -> effective level

Keeping this in one place is what guarantees the two phases can never
compute the metrics differently: ``main.py`` only decides what to *do* with
the result (score an assessment, or drive alerts).

Frames with no face go through :meth:`MetricsPipeline.note_no_face`, which
returns a :class:`NoFaceVerdict` - the level to drive for that frame - from
the same :class:`NoFacePolicy` in both phases. The policy is only *enabled*
in monitoring; pre-drive keeps its own answer to a missing face (the
assessment is voided and routed to an operator) and gets the old ALERT.

The module imports no cv2 / dlib. The :class:`HeadPoseEstimator` (which
needs cv2 for ``solvePnP``) is injected, so the pipeline can be exercised
with a stub estimator on a machine without those libraries.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np

from modules.blink import BlinkDetector, MicrosleepDetector
from modules.calibration import EAR_THRESHOLD_RATIO, EarSelfSeed, finite_or_none
from modules.ear import EARCalculator, RollingMedian
from modules.frs import (DANGER_THRESHOLD, EAR_MEDIAN_WINDOW_S, LEVEL_COLORS,
                         MICROSLEEP_COUNT_WINDOW_S, WARNING_THRESHOLD, FRSCalculator)
from modules.mar import MARCalculator, YawnDetector
from modules.perclos import PERCLOSCalculator

logger = logging.getLogger(__name__)

# Head-pose debounce, in wall-clock seconds (time.monotonic()) rather than
# frames: the real loop rate on a Pi is well below config.CAMERA_FPS because
# of dlib landmarks + periodic face recognition, so a frame count would
# stretch unpredictably. A glance at a side mirror is well under 1.5 s.
HEAD_POSE_DANGER_HOLD_S: float = 1.5
# Hysteresis on release: the pose must be normal for this long before a
# head-pose DANGER is dropped, so a head bobbing through the normal range
# while nodding off does not make the level chatter.
HEAD_POSE_RELEASE_HOLD_S: float = 0.5

# Blink-frequency baselines are calibrated over this window (Module 7). A
# phase that counts blinks over a shorter window scales the baseline by
# ``window / CALIBRATION_BLINK_WINDOW_S`` so the ratio keeps its meaning.
CALIBRATION_BLINK_WINDOW_S: float = 60.0

# Log the head pose every N processed frames to avoid log spam.
POSE_LOG_INTERVAL: int = 30

# Hysteresis on the microsleep DANGER override. There is no engage hold: the
# 1.0 s closure ``MicrosleepDetector`` already required *is* the hold, so the
# override engages on the frame the microsleep is confirmed. The release hold
# keeps the level at DANGER for this long after the eyes reopen, so a 1.2 s
# microsleep does not drop straight back to ALERT the instant the driver
# blinks awake.
MICROSLEEP_DANGER_HOLD_S: float = 0.0
MICROSLEEP_RELEASE_HOLD_S: float = 3.0

# Hysteresis on the FRS bands (2026-09-28; see BandHysteresis). A band is
# entered on the first frame the score reaches its threshold - no engage
# hold, so escalation is never delayed - and left only once the score has
# stayed below (threshold - margin) for the release hold. Sized on the
# 2026-09-23 11:49 flicker rebuilt from real EAR (driver 6's calibration
# series against the population baseline, on the log's blink-duration
# plateau): with the 0.5 s EAR median the smoothed EAR term still spans
# ~0.06 (p5-p95), and 6-26 changes/min remained at 0.40; 0.06 / 1.0 s took
# that to 0, and to 1-3/min at 0.65.
BAND_RELEASE_MARGIN: float = 0.06
BAND_RELEASE_HOLD_S: float = 1.0

# No-face policy (monitoring only; see NoFacePolicy). All wall-clock seconds
# of *continuous* no-face, measured with time.monotonic().
# Up to this long a gap is a detector dropout: the level is held silently.
NO_FACE_HOLD_S: float = 2.0
# From ALERT / WARNING, a gap this long means the system can no longer see the
# driver at all - a FAULT for the operator, not a fatigue verdict.
NO_FACE_FAULT_S: float = 10.0
# A DANGER latched across a gap is released only after the face is back and
# its own level has been below DANGER for this long continuously. Matches
# MICROSLEEP_RELEASE_HOLD_S, the other "don't drop DANGER on one good frame".
NO_FACE_LATCH_RELEASE_S: float = 3.0

# Colour name for the FAULT level in FrameMetrics.effective_result() (the
# FRS bands' names are in frs.LEVEL_COLORS; FAULT is not an FRS band).
FAULT_COLOR: str = "magenta"

# TEMPORARY DIAGNOSTIC (``MetricsPipeline(diag_ear=True)``, CLI ``--diag-ear``).
# With it on, every change of the closed/open decision is logged, plus a
# periodic heartbeat every this many frames so an unchanging decision is still
# visible. Off by default: one line per frame at 30 fps is unreadable.
DIAG_EAR_HEARTBEAT_FRAMES: int = 30


class HeadPoseDebounce:
    """
    Wall-clock hold/release debounce for a DANGER override.

    The override engages once the input has been *continuously* alerting for
    ``danger_hold`` seconds and releases once it has been *continuously*
    normal for ``release_hold`` seconds. Any frame of the opposite state
    restarts the respective timer. Both use ``time.monotonic()`` so the
    behaviour does not depend on the (variable) loop rate.

    Named for its first user, the head-pose override, but the logic is
    generic and :class:`MetricsPipeline` also drives the microsleep override
    with it (``danger_hold=0``, so that one engages immediately).
    """

    def __init__(self, danger_hold: float, release_hold: float) -> None:
        self.danger_hold = danger_hold
        self.release_hold = release_hold
        self.active = False
        self._alert_since: Optional[float] = None
        self._normal_since: Optional[float] = None

    def update(self, alerting: bool, now: float) -> Optional[str]:
        """
        Feed one pose observation.

        Args:
            alerting: ``pose["alert"]`` for this frame (``False`` if the pose
                could not be solved).
            now: ``time.monotonic()`` timestamp of the observation.

        Returns:
            ``"engaged"`` on the frame the override switches on,
            ``"released"`` on the frame it switches off, else ``None``.
        """
        if alerting:
            self._normal_since = None
            if self._alert_since is None:
                self._alert_since = now
            if not self.active and now - self._alert_since >= self.danger_hold:
                self.active = True
                return "engaged"
        else:
            self._alert_since = None
            if self.active:
                if self._normal_since is None:
                    self._normal_since = now
                if now - self._normal_since >= self.release_hold:
                    self.active = False
                    self._normal_since = None
                    return "released"
        return None

    def reset(self) -> None:
        """Forget all timers and drop any active override (e.g. face lost, new driver)."""
        self.active = False
        self.clear_timers()

    def clear_timers(self) -> None:
        """Forget both timers but keep ``active`` (a no-face gap under band hysteresis)."""
        self._alert_since = None
        self._normal_since = None

    def alert_elapsed(self, now: float) -> float:
        """Seconds the pose has been continuously alerting (0 if it is not)."""
        return 0.0 if self._alert_since is None else now - self._alert_since

    def normal_elapsed(self, now: float) -> float:
        """Seconds the pose has been continuously normal *while the override is active*."""
        return 0.0 if self._normal_since is None else now - self._normal_since


class BandHysteresis:
    """
    FRS band (ALERT / WARNING / DANGER) with release hysteresis.

    One :class:`HeadPoseDebounce` per threshold, with no engage hold. A gate
    switches on the first frame ``frs >= threshold`` and off once
    ``frs < threshold - margin`` has held for ``release_hold`` seconds, so a
    score sitting on a threshold holds the higher band instead of flipping
    every frame. Only the way *down* is slowed; the overrides (microsleep,
    head pose, no-face latch) sit on top of the band in
    :attr:`FrameMetrics.level` and are not affected by it.
    """

    def __init__(self, margin: float = BAND_RELEASE_MARGIN,
                 release_hold: float = BAND_RELEASE_HOLD_S) -> None:
        self.margin = margin
        self.gates = ((WARNING_THRESHOLD, HeadPoseDebounce(0.0, release_hold)),
                      (DANGER_THRESHOLD, HeadPoseDebounce(0.0, release_hold)))

    @property
    def level(self) -> str:
        """Current band."""
        warning, danger = (gate.active for _, gate in self.gates)
        return "DANGER" if danger else "WARNING" if warning else "ALERT"

    def update(self, frs: float, now: float) -> str:
        """Feed one score (``time.monotonic()`` timestamp); returns the band."""
        for threshold, gate in self.gates:
            gate.update(frs >= threshold - (self.margin if gate.active else 0.0), now)
        return self.level

    def clear_timers(self) -> None:
        """Keep the band, forget the timers (no-face gap: only the level carries)."""
        for _, gate in self.gates:
            gate.clear_timers()

    def reset(self) -> None:
        """Back to ALERT (new driver)."""
        for _, gate in self.gates:
            gate.reset()


@dataclass(frozen=True)
class NoFaceVerdict:
    """
    What a no-face frame means for the alert level.

    ``band`` is one of:

    * ``"HOLD"``  - the gap began at ALERT / WARNING (or at FAULT from a
      self-seed that gave up) and is shorter than :data:`NO_FACE_FAULT_S`;
      ``level`` is the level the gap began at.
      ``face_lost`` turns true past :data:`NO_FACE_HOLD_S`, when it stops
      being a detector dropout and is reported as a lost face.
    * ``"LATCH"`` - the gap began at DANGER; ``level`` is DANGER for as long
      as the gap lasts and never becomes FAULT.
    * ``"FAULT"`` - the gap began at ALERT / WARNING and has lasted
      :data:`NO_FACE_FAULT_S` or more; ``level`` is ``"FAULT"``.
    * ``"OFF"``   - policy disabled (pre-drive); ``level`` is ALERT, as
      before the policy existed.
    """

    band: str
    level: str
    gap_s: float
    entry_level: str
    face_lost: bool = False
    # Set on the one frame a gap opens a backend fault, else None:
    # "fault_entered" when a HOLD gap crosses into FAULT (fault_type
    # "no_face"), "latch_lost" when a LATCH gap passes NO_FACE_HOLD_S
    # (fault_type "danger_latched").
    event: Optional[str] = None


class NoFacePolicy:
    """
    Three-band handling of frames with no face, for the monitoring phase.

    The rule it enforces is an asymmetry: **a missing face may never lower
    the level, and may never manufacture DANGER from a calm baseline.**

    * HOLD  - from ALERT / WARNING the last level is held (not lowered to
      ALERT, not raised). Past :data:`NO_FACE_HOLD_S` the gap is reported
      as a lost face; the level is still held.
    * LATCH - from DANGER the level stays DANGER for the whole gap, and after
      the face returns until its own level has been below DANGER for
      :data:`NO_FACE_LATCH_RELEASE_S`. A driver whose head drops out of
      frame mid-microsleep is exactly the case this exists for.
    * FAULT - from ALERT / WARNING, past :data:`NO_FACE_FAULT_S` the level
      becomes ``"FAULT"``: the unit cannot see the driver, which is a
      condition for an operator, not evidence of fatigue. Never entered from
      DANGER (that would replace the strongest signal with a weaker one).

    The detectors' own state (pose debounce, microsleep closure) is still
    cleared on every no-face frame by :meth:`MetricsPipeline.note_no_face`;
    bridging *those* across a gap is what would manufacture DANGER. Only the
    *level* is carried.

    Args:
        enabled: ``False`` makes every verdict ``OFF`` / ALERT (pre-drive).
        hold_s, fault_s, latch_release_s: See the module constants.
    """

    def __init__(
        self,
        enabled: bool = True,
        hold_s: float = NO_FACE_HOLD_S,
        fault_s: float = NO_FACE_FAULT_S,
        latch_release_s: float = NO_FACE_LATCH_RELEASE_S,
    ) -> None:
        if not 0.0 <= hold_s <= fault_s:
            raise ValueError(f"need 0 <= hold_s <= fault_s, got {hold_s}, {fault_s}")
        self.enabled = enabled
        self.hold_s = hold_s
        self.fault_s = fault_s
        # Engaged only from on_no_face (never by a face-present DANGER, so
        # ordinary FRS DANGER gets no extra hysteresis); released by the same
        # hold/release debounce the overrides use.
        self.latch = HeadPoseDebounce(0.0, latch_release_s)
        self._gap_start: Optional[float] = None
        self._entry_level: str = "ALERT"
        self._last: Optional[NoFaceVerdict] = None

    @property
    def in_gap(self) -> bool:
        """Whether the previous frame had no face."""
        return self._gap_start is not None

    def on_no_face(self, last_level: Optional[str], now_mono: float) -> NoFaceVerdict:
        """
        Classify one no-face frame.

        Args:
            last_level: Effective level of the last scored frame before this
                gap (``None`` if no face has been scored yet - treated as
                ALERT). Only read on the first frame of a gap.
            now_mono: ``time.monotonic()`` for the frame.
        """
        if self._gap_start is None:
            self._gap_start = now_mono
            # FAULT: the self-seed gave up (face present, no EAR baseline);
            # the gap holds it rather than showing ALERT.
            entry = (last_level if last_level in ("ALERT", "WARNING", "DANGER", "FAULT")
                     else "ALERT")
            # A gap that begins inside a latch release window re-arms it.
            if self.latch.active:
                entry = "DANGER"
            self._entry_level = entry
        gap = now_mono - self._gap_start
        entry = self._entry_level

        if not self.enabled:
            verdict = NoFaceVerdict("OFF", "ALERT", gap, entry)
        elif entry == "DANGER":
            self.latch.update(True, now_mono)
            lost = gap > self.hold_s
            was_lost = self._last is not None and self._last.face_lost
            verdict = NoFaceVerdict("LATCH", "DANGER", gap, entry, lost,
                                    "latch_lost" if lost and not was_lost else None)
        elif gap >= self.fault_s:
            was_fault = self._last is not None and self._last.band == "FAULT"
            verdict = NoFaceVerdict("FAULT", "FAULT", gap, entry, True,
                                    None if was_fault else "fault_entered")
        else:
            verdict = NoFaceVerdict("HOLD", entry, gap, entry, gap > self.hold_s)
        self._last = verdict
        return verdict

    def on_face(self, level: str, now_mono: float) -> Tuple[bool, Optional[NoFaceVerdict]]:
        """
        Feed a face frame's own level (FRS band, or DANGER under an override).

        Args:
            level: The frame's level before the latch is applied.
            now_mono: ``time.monotonic()`` for the frame.

        Returns:
            ``(latched, ended_gap)``: whether the latch still forces DANGER
            this frame, and the last verdict of the gap that just ended (on
            the first face frame after one, else ``None``).
        """
        ended = self._last if self._gap_start is not None else None
        self._gap_start = None
        self._last = None
        if self.latch.active:
            if self.latch.update(level == "DANGER", now_mono) == "released":
                logger.info("Face back and below DANGER for %.1fs - releasing no-face "
                            "DANGER latch", self.latch.release_hold)
        return self.latch.active, ended

    def reset(self) -> None:
        """Drop any gap and latch (new driver)."""
        self.latch.reset()
        self._gap_start = None
        self._entry_level = "ALERT"
        self._last = None


@dataclass
class FrameMetrics:
    """Everything the pipeline derived from one frame."""

    t: float                       # time.time() of the frame
    ear: float
    mar: float
    perclos: float                 # percent
    blink_duration_ms: float       # mean of recent blinks
    blink_freq: float              # blinks in the detector's window
    ear_norm: float
    bd_norm: float
    bf_norm: float
    perclos_norm: float
    mar_norm: float
    yawn_norm: float               # mar_norm while yawning, else 1.0
    frs_result: Dict[str, Any]     # FRSCalculator.compute() output (raw, un-overridden)
    pose: Optional[Dict[str, Any]]
    pose_override: bool            # head-pose DANGER override active
    pose_event: Optional[str]      # "engaged" | "released" | None
    yawn_event: Optional[Dict[str, float]]
    yawn_status: Dict[str, Any] = field(default_factory=dict)
    # ear_norm as the FRS saw it: the EAR_MEDIAN_WINDOW_S rolling median.
    # ``ear_norm`` stays the frame's own value, so ear / ear_norm is still the
    # baseline a recording was scored against.
    ear_norm_frs: float = 0.0
    # FRS band after BandHysteresis; frs_result["level"] is the raw band.
    band: str = "ALERT"
    # False while an unrecognised driver's EAR baseline is still being
    # self-seeded: no FRS, frs_result is a zero placeholder.
    scored: bool = True
    microsleep_override: bool = False           # microsleep DANGER override active
    microsleep_event: Optional[Dict[str, float]] = None   # set on confirmation
    microsleep_count: int = 0                   # confirmed in the trailing window
    microsleep_status: Dict[str, Any] = field(default_factory=dict)
    no_face_latch: bool = False                 # DANGER carried over from a no-face gap
    gap_ended: Optional[NoFaceVerdict] = None   # last verdict of a gap that just ended
    # The EAR self-seed gave up (EarSelfSeed.gave_up): no baseline, level
    # FAULT unless a DANGER override is active.
    seed_failed: bool = False

    @property
    def frs(self) -> float:
        """Raw (eye + mouth) FRS for this frame."""
        return float(self.frs_result["frs"])

    @property
    def overridden(self) -> bool:
        """Whether any DANGER override (head pose, microsleep, no-face latch) is active."""
        return self.pose_override or self.microsleep_override or self.no_face_latch

    @property
    def override_reason(self) -> str:
        """Human-readable cause of the override, or ``""`` when there is none."""
        reasons = []
        if self.microsleep_override:
            reasons.append("microsleep")
        if self.pose_override:
            reasons.append("head pose")
        if self.no_face_latch:
            reasons.append("no-face latch")
        return " + ".join(reasons)

    @property
    def level(self) -> str:
        """
        Effective alert level: ``DANGER`` while any override is active, else
        ``FAULT`` if the EAR self-seed gave up (the eyes cannot be measured -
        ALERT would claim a driver checked and fine), else the band.
        """
        if self.overridden:
            return "DANGER"
        return "FAULT" if self.seed_failed else self.band

    def last_known(self) -> Dict[str, Any]:
        """
        The measurements a monitoring-fault report carries as ``last_known``.

        Taken from the last scored frame before a no-face gap, so the portal
        can show what state the driver was in when they stopped being seen.
        """
        return {
            "at": datetime.fromtimestamp(self.t, timezone.utc).isoformat(),
            "level": self.level,
            "frs": round(self.frs, 4),
            "ear": round(float(self.ear), 4),
            "perclos": round(float(self.perclos), 2),
            "blink_duration_ms": round(float(self.blink_duration_ms), 1),
            "blink_freq": round(float(self.blink_freq), 1),
            "mar": round(float(self.mar), 4),
            "microsleep_count": int(self.microsleep_count),
            "override_reason": self.override_reason or None,
        }

    def effective_result(self) -> Dict[str, Any]:
        """``frs_result`` with ``level``/``color`` set to the effective level."""
        return dict(self.frs_result, level=self.level,
                    color=LEVEL_COLORS.get(self.level, FAULT_COLOR))


class MetricsPipeline:
    """
    Owns every per-frame calculator and runs them in a fixed order.

    Typical usage (one instance per driver session)::

        pipeline = MetricsPipeline(head_pose, blink_window_s=60)
        metrics = pipeline.process(landmarks, thresholds, now, now_mono)
        alert_manager.set_alert_level(metrics.level)

    Args:
        head_pose: A ``HeadPoseEstimator`` (or any object with
            ``estimate(landmarks) -> dict | None``).
        blink_window_s: Window over which blink frequency is counted. Use the
            phase length for a short assessment; the frequency baseline is
            scaled by ``blink_window_s / 60`` automatically.
        pose_release_hold: Seconds of normal pose before a head-pose DANGER
            override is released.
        microsleep_release_hold: Seconds after the eyes reopen before a
            microsleep DANGER override is released.
        no_face_policy: Enable the HOLD / LATCH / FAULT handling of no-face
            frames (:class:`NoFacePolicy`). Monitoring only; with it off,
            :meth:`note_no_face` returns ALERT as it always did.
        diag_ear: TEMPORARY DIAGNOSTIC. Log the raw EAR, the active closure
            threshold and the resulting closed/open decision on every change
            and periodically in between. Use when PERCLOS / blink / microsleep
            all report nothing, which means the closure test is never firing.
    """

    def __init__(
        self,
        head_pose: Any,
        blink_window_s: float = CALIBRATION_BLINK_WINDOW_S,
        pose_release_hold: float = HEAD_POSE_RELEASE_HOLD_S,
        pose_danger_hold: float = HEAD_POSE_DANGER_HOLD_S,
        microsleep_release_hold: float = MICROSLEEP_RELEASE_HOLD_S,
        no_face_policy: bool = False,
        diag_ear: bool = False,
    ) -> None:
        self.head_pose = head_pose
        self.blink_window_s = float(blink_window_s)
        self.ear_calc = EARCalculator()
        self.ear_median = RollingMedian(EAR_MEDIAN_WINDOW_S)
        self.mar_calc = MARCalculator()
        self.blink_detector = BlinkDetector(frequency_window=int(blink_window_s))
        self.perclos_calc = PERCLOSCalculator()
        self.yawn_detector = YawnDetector()
        self.microsleep_detector = MicrosleepDetector()
        self.frs_calc = FRSCalculator()
        self.pose_debounce = HeadPoseDebounce(pose_danger_hold, pose_release_hold)
        # Engages on the frame the microsleep is confirmed (no engage hold -
        # the 1.0 s closure was the hold) and lingers for the release hold.
        self.microsleep_debounce = HeadPoseDebounce(
            MICROSLEEP_DANGER_HOLD_S, microsleep_release_hold
        )
        self.no_face = NoFacePolicy(enabled=no_face_policy)
        self.band = BandHysteresis()
        # EAR baseline for a driver with no usable calibration (process(self_seed=True)).
        self.ear_seed = EarSelfSeed()
        # Effective level of the last scored frame: what a no-face gap is
        # judged against. None until the first face.
        self.last_level: Optional[str] = None
        self._frames = 0
        self.diag_ear = diag_ear
        # Last closed/open decision, for edge-triggered diagnostic logging.
        self._diag_last_closed: Optional[bool] = None

    # ------------------------------------------------------------------

    def process(
        self,
        landmarks: np.ndarray,
        thresholds: Dict[str, float],
        now: float,
        now_mono: float,
        self_seed: bool = False,
    ) -> FrameMetrics:
        """
        Run the full metric chain on one frame's landmarks.

        Args:
            landmarks: ``(68, 2)`` dlib landmarks.
            thresholds: The driver's baseline dict (all keys present).
            now: ``time.time()`` for the frame (blink / yawn timestamps).
            now_mono: ``time.monotonic()`` for the frame (pose debounce).
            self_seed: No usable calibration (monitoring only):
                ``ear_baseline`` / ``ear_threshold`` are taken from
                :attr:`ear_seed` instead of ``thresholds``, which then need
                only the other keys. Until the seed freezes the frame is not
                scored (``FrameMetrics.scored`` is ``False``): the level is
                ALERT unless an override forces DANGER.

        Returns:
            A :class:`FrameMetrics`.
        """
        self._frames += 1

        # Head pose + debounce. A sustained nod / look-away / tilt overrides
        # the level to DANGER independent of the eye metrics.
        pose = self.head_pose.estimate(landmarks)
        alerting = pose is not None and bool(pose["alert"])
        pose_event = self.pose_debounce.update(alerting, now_mono)
        if pose_event == "engaged":
            logger.warning(
                "Head pose alert (%s): pitch=%.1f yaw=%.1f roll=%.1f held %.2fs -> DANGER override",
                ", ".join(k[3:].replace("_", " ") for k in
                          ("is_nodding", "is_looking_away", "is_tilting") if pose.get(k))
                or "unspecified",
                pose["pitch"], pose["yaw"], pose["roll"], self.pose_debounce.danger_hold,
            )
        elif pose_event == "released":
            logger.info("Head pose normal for %.2fs - releasing DANGER override",
                        self.pose_debounce.release_hold)
        if pose is not None and self._frames % POSE_LOG_INTERVAL == 0:
            logger.info(
                "Head pose: pitch=%.1f yaw=%.1f roll=%.1f alert=%s (alerting %.2fs, override=%s)",
                pose["pitch"], pose["yaw"], pose["roll"], pose["alert"],
                self.pose_debounce.alert_elapsed(now_mono), self.pose_debounce.active,
            )

        # Raw metrics
        ear = self.ear_calc.compute_average_ear(landmarks)

        # No calibration: the EAR baseline and threshold come from this
        # driver's own EAR. Until the seed freezes there is no FRS (overrides
        # only), and until it has a provisional threshold not even microsleep
        # detection - see modules.calibration.EarSelfSeed.
        scored = True
        if self_seed:
            if self.ear_seed.update(ear, now_mono):
                logger.info("Scoring on self-seeded EAR baseline %.4f / threshold %.4f "
                            "(ratio %.2f) - FRS starts now", self.ear_seed.baseline,
                            self.ear_seed.threshold, EAR_THRESHOLD_RATIO)
            thresholds = dict(thresholds, ear_baseline=self.ear_seed.baseline,
                              ear_threshold=self.ear_seed.provisional_threshold(now_mono))
            scored = self.ear_seed.frozen
        # After SELF_SEED_MAX_ATTEMPTS discarded seeds: no baseline, no
        # threshold (so no microsleep detection either), level FAULT.
        seed_failed = self_seed and self.ear_seed.gave_up
        # None while self-seeding without a provisional threshold.
        ear_threshold = finite_or_none(thresholds.get("ear_threshold"))

        # TEMPORARY DIAGNOSTIC. This single comparison gates PERCLOS, blink
        # detection and microsleep detection alike - all three call
        # ``ear < thresholds["ear_threshold"]`` - so when all three report
        # nothing at once, this line shows whether the closure test is firing
        # at all, and against what.
        if self.diag_ear and ear_threshold is not None:
            self._log_ear_decision(ear, thresholds, now)

        if scored:
            self.blink_detector.update(ear, ear_threshold, now)
        # Same EAR and same threshold as the blink detector, so the two can
        # never disagree about whether the eyes are shut. Closures of
        # MICROSLEEP_MIN_DURATION_S or more are reported here and rejected
        # there, which is the whole split.
        microsleep_event = (None if ear_threshold is None else
                            self.microsleep_detector.update(ear, ear_threshold, now))
        if microsleep_event:
            # min_ear, not the confirming frame's ear: when the closure
            # crosses 1.0 s between two frames the confirming frame is the
            # *reopening* one, whose EAR is above the threshold and would
            # make this line read "EAR 0.266 < threshold 0.191".
            logger.warning(
                "MICROSLEEP confirmed: eyes closed %.2fs (min EAR %.3f < threshold %.3f) - "
                "%d in the last %.0fs -> DANGER override",
                microsleep_event["duration_ms"] / 1000.0,
                microsleep_event["min_ear"], ear_threshold,
                self.microsleep_detector.get_microsleep_count(
                    MICROSLEEP_COUNT_WINDOW_S, now),
                MICROSLEEP_COUNT_WINDOW_S,
            )
        ms_event = self.microsleep_debounce.update(
            self.microsleep_detector.is_microsleeping, now_mono
        )
        if ms_event == "released":
            logger.info(
                "Microsleep over for %.2fs - releasing DANGER override (total closure %.2fs)",
                self.microsleep_debounce.release_hold,
                (self.microsleep_detector.microsleep_durations[-1] / 1000.0
                 if self.microsleep_detector.microsleep_durations else 0.0),
            )
        mar = self.mar_calc.compute_mar(landmarks)
        microsleep_count = self.microsleep_detector.get_microsleep_count(
            MICROSLEEP_COUNT_WINDOW_S, now
        )
        if scored:
            perclos = self.perclos_calc.update(ear, ear_threshold, now_mono,
                                               baseline=thresholds["perclos_baseline"])
            yawn_event = self.yawn_detector.update(mar, thresholds["yawn_threshold"], now)
            if yawn_event:
                logger.info("Yawn detected (mouth open %.1fs, MAR %.3f vs threshold %.3f) - "
                            "%d this session",
                            yawn_event["duration_ms"] / 1000.0, mar,
                            thresholds["yawn_threshold"], self.yawn_detector.get_yawn_count())

            # Normalise against this driver's calibration
            blink_duration_ms = self.blink_detector.get_average_duration(now)
            blink_freq = self.blink_detector.get_blink_frequency(now)
            ear_norm = self.ear_calc.normalize(ear, thresholds["ear_baseline"])
            # The FRS gets the rolling median, so a blink - already counted by
            # PERCLOS and the blink terms - does not also spike the EAR term.
            ear_norm_frs = self.ear_median.update(ear_norm, now_mono)
            bd_norm = self.blink_detector.normalize_duration(
                blink_duration_ms, thresholds["blink_duration_baseline"]
            )
            # The frequency baseline was calibrated over 60 s; if this pipeline
            # counts over a shorter window, scale the baseline to the same units.
            bf_baseline = thresholds["blink_frequency_baseline"] * (
                self.blink_window_s / CALIBRATION_BLINK_WINDOW_S
            )
            bf_norm = self.blink_detector.normalize_frequency(blink_freq, bf_baseline)
            perclos_norm = self.perclos_calc.normalize(perclos, thresholds["perclos_baseline"])
            # Yawn-gated: the mouth only contributes to the FRS while a confirmed
            # (sustained) yawn is in progress. Talking, laughing and the 1.5 s
            # of a yawn before it is confirmed all pass 1.0 = zero excess.
            mar_norm = self.mar_calc.normalize(mar, thresholds["mar_baseline"])
            yawn_norm = mar_norm if self.yawn_detector.is_yawning else 1.0

            # Persistence term: repeated microsleeps keep the *score* elevated
            # between episodes, which the override alone cannot do.
            frs_result = self.frs_calc.compute(
                ear_norm_frs, bd_norm, bf_norm, perclos_norm, yawn_norm, microsleep_count
            )
            band = self.band.update(float(frs_result["frs"]), now_mono)
        else:
            # Self-seeding: no baseline yet, so nothing is normalised and
            # the score is a zero placeholder the level never reads.
            perclos, yawn_event = 0.0, None
            blink_duration_ms = blink_freq = 0.0
            nan = float("nan")
            ear_norm = ear_norm_frs = bd_norm = bf_norm = perclos_norm = mar_norm = nan
            yawn_norm = 1.0
            frs_result = self.frs_calc.compute(1.0, 1.0, 1.0, 1.0)
            band = self.band.level

        # The latch is released on this frame's own level, overrides included.
        own_level = ("DANGER" if self.pose_debounce.active or self.microsleep_debounce.active
                     else "FAULT" if seed_failed else str(frs_result["level"]))
        no_face_latch, gap_ended = self.no_face.on_face(own_level, now_mono)

        metrics = FrameMetrics(
            t=now, ear=ear, mar=mar, perclos=perclos,
            blink_duration_ms=blink_duration_ms, blink_freq=blink_freq,
            ear_norm=ear_norm, bd_norm=bd_norm, bf_norm=bf_norm,
            perclos_norm=perclos_norm, mar_norm=mar_norm, yawn_norm=yawn_norm,
            frs_result=frs_result, pose=pose,
            pose_override=self.pose_debounce.active, pose_event=pose_event,
            yawn_event=yawn_event, yawn_status=self.yawn_detector.get_status(now),
            microsleep_override=self.microsleep_debounce.active,
            microsleep_event=microsleep_event,
            microsleep_count=microsleep_count,
            microsleep_status=self.microsleep_detector.get_status(now),
            no_face_latch=no_face_latch,
            gap_ended=gap_ended,
            ear_norm_frs=ear_norm_frs,
            band=band,
            scored=scored,
            seed_failed=seed_failed,
        )
        self.last_level = metrics.level
        return metrics

    def _log_ear_decision(
        self, ear: float, thresholds: Dict[str, float], now: float
    ) -> None:
        """
        TEMPORARY DIAGNOSTIC: report the per-frame closed/open decision.

        Logged on every transition (so each closure is bracketed exactly) and
        every :data:`DIAG_EAR_HEARTBEAT_FRAMES` frames in between (so a
        decision that never changes is still visible). Enabled by
        ``diag_ear``; see :meth:`__init__`.

        Args:
            ear: Raw EAR for this frame.
            thresholds: The driver's threshold dict.
            now: ``time.time()`` for the frame.
        """
        # Either may be None while self-seeding (the provisional threshold
        # exists from 2 s, the baseline only once the seed freezes at 5 s).
        threshold = finite_or_none(thresholds.get("ear_threshold"))
        baseline = finite_or_none(thresholds.get("ear_baseline")) or 0.0
        if threshold is None:
            return
        closed = ear < threshold
        changed = closed != self._diag_last_closed
        if not changed and self._frames % DIAG_EAR_HEARTBEAT_FRAMES != 0:
            return
        self._diag_last_closed = closed
        logger.info(
            "DIAG ear=%.4f threshold=%.4f -> %s | ear/baseline=%.3f "
            "threshold/baseline=%.3f | perclos=%.1f%% closure=%.2fs "
            "blinks_in_window=%d microsleeps=%d%s",
            ear, threshold, "CLOSED" if closed else "open",
            (ear / baseline) if baseline > 0 else float("nan"),
            (threshold / baseline) if baseline > 0 else float("nan"),
            self.perclos_calc.get_perclos(),
            self.microsleep_detector.get_closed_duration(now),
            int(self.blink_detector.get_blink_frequency(now)),
            self.microsleep_detector.get_microsleep_count(MICROSLEEP_COUNT_WINDOW_S, now),
            "  <-- transition" if changed else "",
        )

    def note_no_face(
        self, now: Optional[float] = None, now_mono: Optional[float] = None
    ) -> NoFaceVerdict:
        """
        Call on frames with no landmarks; returns the level to drive.

        Two separate things happen here, deliberately kept apart:

        1. **Detector state is cleared.** No pose observation is possible, so
           the debounce must not keep a stale "alerting since" time - or a
           live override - across the gap. Likewise the microsleep closure
           timer: left running, the next face frame could confirm a
           "microsleep" that was really a lost face. Bridging either across a
           gap is how a missing face would manufacture DANGER.
        2. **The level is decided by** :class:`NoFacePolicy` from the level
           the gap began at. A DANGER in force when the face went (an active
           microsleep or head-pose override included) is latched, so
           clearing the overrides in step 1 no longer drops the level.

        Args:
            now: ``time.time()`` for the frame, used to report how long the
                eyes had been shut. Defaults to the wall clock.
            now_mono: ``time.monotonic()`` for the frame (policy timing).
                Defaults to the monotonic clock.

        Returns:
            A :class:`NoFaceVerdict`; drive ``verdict.level``.
        """
        if now_mono is None:
            now_mono = time.monotonic()
        # Judge the gap against the level before step 1 clears anything.
        verdict = self.no_face.on_no_face(self.last_level, now_mono)
        if verdict.event == "fault_entered":
            logger.warning(
                "NO FACE for %.1fs from %s - FAULT: the driver cannot be observed",
                verdict.gap_s, verdict.entry_level,
            )
        elif verdict.event == "latch_lost":
            logger.warning(
                "NO FACE for %.1fs at DANGER - DANGER latched, driver unobserved",
                verdict.gap_s,
            )

        if self.pose_debounce.active:
            logger.info("Face lost - clearing head-pose DANGER override%s",
                        " (level latched at DANGER)" if verdict.band == "LATCH" else "")
        self.pose_debounce.reset()

        abandoned = self.microsleep_detector.note_no_face(now)
        if abandoned and abandoned["was_microsleeping"]:
            logger.warning(
                "FACE LOST MID-MICROSLEEP after %.2fs of eye closure - abandoning the "
                "closure; level %s. The driver may still be microsleeping; nothing "
                "can be measured without landmarks.",
                abandoned["closed_s"],
                "latched at DANGER" if verdict.band == "LATCH" else verdict.level,
            )
        elif self.microsleep_debounce.active:
            # Eyes had already reopened; this was only the release hold.
            logger.info("Face lost - clearing microsleep DANGER override (release hold)")
        elif abandoned and float(abandoned["closed_s"]) >= self.microsleep_detector.min_duration / 2:
            # A closure well on its way to a microsleep, interrupted. Too
            # common to warn about, useful when reconstructing a session.
            logger.debug("Face lost %.2fs into a closure - closure abandoned unconfirmed",
                         abandoned["closed_s"])
        self.microsleep_debounce.reset()
        # Pre-gap EAR samples say nothing about the face that comes back.
        self.ear_median.reset()
        # The band is a level, which carries; its release timer is not.
        self.band.clear_timers()
        return verdict

    def reset(self) -> None:
        """Fresh per-driver history (blinks, microsleeps, PERCLOS, yawns, debounces)."""
        self.blink_detector.reset()
        self.perclos_calc.reset()
        self.yawn_detector.reset()
        self.microsleep_detector.reset()
        self.microsleep_debounce.reset()
        self.pose_debounce.reset()
        self.ear_median.reset()
        self.band.reset()
        self.ear_seed.reset()
        self.no_face.reset()
        self.last_level = None
        self._frames = 0
