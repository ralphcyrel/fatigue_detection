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

The module imports no cv2 / dlib. The :class:`HeadPoseEstimator` (which
needs cv2 for ``solvePnP``) is injected, so the pipeline can be exercised
with a stub estimator on a machine without those libraries.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from modules.blink import BlinkDetector, MicrosleepDetector
from modules.ear import EARCalculator
from modules.frs import MICROSLEEP_COUNT_WINDOW_S, FRSCalculator
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
        self._alert_since = None
        self._normal_since = None

    def alert_elapsed(self, now: float) -> float:
        """Seconds the pose has been continuously alerting (0 if it is not)."""
        return 0.0 if self._alert_since is None else now - self._alert_since

    def normal_elapsed(self, now: float) -> float:
        """Seconds the pose has been continuously normal *while the override is active*."""
        return 0.0 if self._normal_since is None else now - self._normal_since


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
    microsleep_override: bool = False           # microsleep DANGER override active
    microsleep_event: Optional[Dict[str, float]] = None   # set on confirmation
    microsleep_count: int = 0                   # confirmed in the trailing window
    microsleep_status: Dict[str, Any] = field(default_factory=dict)

    @property
    def frs(self) -> float:
        """Raw (eye + mouth) FRS for this frame."""
        return float(self.frs_result["frs"])

    @property
    def overridden(self) -> bool:
        """Whether either DANGER override (head pose or microsleep) is active."""
        return self.pose_override or self.microsleep_override

    @property
    def override_reason(self) -> str:
        """Human-readable cause of the override, or ``""`` when there is none."""
        reasons = []
        if self.microsleep_override:
            reasons.append("microsleep")
        if self.pose_override:
            reasons.append("head pose")
        return " + ".join(reasons)

    @property
    def level(self) -> str:
        """Effective alert level: ``DANGER`` while either override is active."""
        return "DANGER" if self.overridden else str(self.frs_result["level"])

    def effective_result(self) -> Dict[str, Any]:
        """``frs_result`` with ``level``/``color`` forced to DANGER under an override."""
        if self.overridden:
            return dict(self.frs_result, level="DANGER", color="red")
        return self.frs_result


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
    """

    def __init__(
        self,
        head_pose: Any,
        blink_window_s: float = CALIBRATION_BLINK_WINDOW_S,
        pose_release_hold: float = HEAD_POSE_RELEASE_HOLD_S,
        pose_danger_hold: float = HEAD_POSE_DANGER_HOLD_S,
        microsleep_release_hold: float = MICROSLEEP_RELEASE_HOLD_S,
    ) -> None:
        self.head_pose = head_pose
        self.blink_window_s = float(blink_window_s)
        self.ear_calc = EARCalculator()
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
        self._frames = 0

    # ------------------------------------------------------------------

    def process(
        self,
        landmarks: np.ndarray,
        thresholds: Dict[str, float],
        now: float,
        now_mono: float,
    ) -> FrameMetrics:
        """
        Run the full metric chain on one frame's landmarks.

        Args:
            landmarks: ``(68, 2)`` dlib landmarks.
            thresholds: The driver's baseline dict (all keys present).
            now: ``time.time()`` for the frame (blink / yawn timestamps).
            now_mono: ``time.monotonic()`` for the frame (pose debounce).

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
        self.blink_detector.update(ear, thresholds["ear_threshold"], now)
        # Same EAR and same threshold as the blink detector, so the two can
        # never disagree about whether the eyes are shut. Closures of
        # MICROSLEEP_MIN_DURATION_S or more are reported here and rejected
        # there, which is the whole split.
        microsleep_event = self.microsleep_detector.update(
            ear, thresholds["ear_threshold"], now
        )
        if microsleep_event:
            logger.warning(
                "MICROSLEEP confirmed: eyes closed %.2fs (EAR %.3f < threshold %.3f) - "
                "%d in the last %.0fs -> DANGER override",
                microsleep_event["duration_ms"] / 1000.0, ear,
                thresholds["ear_threshold"],
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
        perclos = self.perclos_calc.update(ear, thresholds["ear_threshold"])
        mar = self.mar_calc.compute_mar(landmarks)
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
        microsleep_count = self.microsleep_detector.get_microsleep_count(
            MICROSLEEP_COUNT_WINDOW_S, now
        )
        frs_result = self.frs_calc.compute(
            ear_norm, bd_norm, bf_norm, perclos_norm, yawn_norm, microsleep_count
        )

        return FrameMetrics(
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
        )

    def note_no_face(self, now: Optional[float] = None) -> None:
        """
        Call on frames with no landmarks.

        No pose observation is possible, so the debounce must not keep a
        stale "alerting since" time - or a live override - across the gap.
        The same applies to the microsleep closure timer: without clearing it
        the timer would keep running across a gap in which no EAR was
        observed at all, and the next frame with a face could confirm a
        "microsleep" that was really a lost face.

        Dropping a *live* microsleep override this way is the conservative
        reading and it is logged at WARNING, because a driver whose head
        leaves the frame mid-microsleep is exactly the case worth seeing in
        the log - the alert level falls back to ALERT on a no-face frame
        regardless, so nothing here changes behaviour, only visibility.

        Args:
            now: ``time.time()`` for the frame, used only to report how long
                the eyes had been shut. Defaults to the wall clock.
        """
        if self.pose_debounce.active:
            logger.info("Face lost - clearing head-pose DANGER override")
        self.pose_debounce.reset()

        abandoned = self.microsleep_detector.note_no_face(now)
        if abandoned and abandoned["was_microsleeping"]:
            logger.warning(
                "FACE LOST MID-MICROSLEEP after %.2fs of eye closure - dropping the "
                "microsleep DANGER override and abandoning the closure. The driver may "
                "still be microsleeping; nothing can be measured without landmarks.",
                abandoned["closed_s"],
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

    def reset(self) -> None:
        """Fresh per-driver history (blinks, microsleeps, PERCLOS, yawns, debounces)."""
        self.blink_detector.reset()
        self.perclos_calc.reset()
        self.yawn_detector.reset()
        self.microsleep_detector.reset()
        self.microsleep_debounce.reset()
        self.pose_debounce.reset()
        self._frames = 0
