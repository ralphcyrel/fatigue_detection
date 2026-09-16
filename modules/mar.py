"""
Module 3b — Mouth Aspect Ratio (MAR) and yawn detection.

The MAR is the mouth's analogue of the EAR (Module 3): a scale-invariant
measure of how open the mouth is, computed from the dlib outer-lip landmarks
(48–59)::

    MAR = (|p50 - p58| + |p51 - p57| + |p52 - p56|) / (2 * |p48 - p54|)

``p48`` and ``p54`` are the mouth corners; ``p50/p58``, ``p51/p57`` and
``p52/p56`` are the three central vertical pairs on the upper and lower outer
lip. The *outer* lip is used rather than the inner lip (60–67) on purpose:
lip thickness gives a stable, non-zero closed-mouth MAR (roughly 0.4–0.6
depending on the driver), so ``MAR / baseline`` behaves like ``EAR /
baseline``. The inner-lip MAR of a closed mouth is ~0.05 and that ratio is
numerically useless. Being a ratio, MAR is unaffected by the driver's
distance from the camera.

Typical outer-lip values (read the real ones off the ``--debug`` overlay):

    closed / resting      ≈ 0.40 – 0.60   (1.0× baseline)
    talking               ≈ 1.3  – 1.8× baseline, opening/closing several
                                             times per second
    yawn                  ≈ 2.5  – 3.5× baseline, held for 4–7 s

:class:`YawnDetector` separates yawns from speech with **two** conditions
that must both hold: the MAR must exceed ``YAWN_MAR_RATIO`` (2.0) times the
driver's calibrated closed-mouth baseline — above talking, below a yawn —
*and* stay above it continuously for ``MIN_YAWN_DURATION_S`` (1.5 s). A
single frame below the threshold restarts the timer, so speech, laughter
and singing (which open and close the mouth far faster than that) never
register, while a real yawn is confirmed ~1.5 s in and remains "in
progress" until the mouth closes.

Downstream, the FRS (Module 6) consumes ``mar_norm`` **only while a yawn is
in progress** (the caller passes ``1.0`` — zero excess — otherwise), and the
calibration module (Module 7) uses the per-frame MAR to learn each driver's
personal closed-mouth baseline.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import euclidean

# ---------------------------------------------------------------------------
# Mouth landmark index range (half-open, matching ``modules.landmark``).
# Duplicated here rather than imported so this module stays free of the
# dlib/cv2 dependency that ``modules.landmark`` pulls in. Keep in sync.
# ---------------------------------------------------------------------------
MOUTH: Tuple[int, int] = (48, 68)

# Number of landmark points that outline the mouth (outer 48–59 + inner 60–67).
MOUTH_POINTS: int = 20

# Yawn threshold as a multiple of the driver's calibrated closed-mouth MAR.
# Talking peaks at ~1.3–1.8× and a yawn at ~2.5–3.5×; 2.0 sits in the gap.
YAWN_MAR_RATIO: float = 2.0

# Seconds the MAR must stay continuously above the yawn threshold before the
# opening is confirmed as a yawn. Speech and laughter never hold a
# yawn-sized opening this long; real yawns last 4–7 s.
MIN_YAWN_DURATION_S: float = 1.5


class MARCalculator:
    """
    Compute the Mouth Aspect Ratio from dlib landmarks.

    Typical usage::

        mar_calc = MARCalculator()
        mar = mar_calc.compute_mar(landmarks)                 # landmarks: (68, 2)
        mar_norm = mar_calc.normalize(mar, baselines["mar_baseline"])
        opened = mar_calc.is_mouth_open(mar, baselines["yawn_threshold"])
    """

    def __init__(self) -> None:
        """Initialise the calculator. No configuration is required."""
        # Stateless by design, like EARCalculator — every call is a pure
        # function of its inputs so one instance can be shared across
        # frames and drivers.
        pass

    def compute_mouth_mar(self, mouth_points: np.ndarray) -> float:
        """
        Compute the MAR from the 20 mouth landmarks.

        Args:
            mouth_points: ``(20, 2)`` array of ``(x, y)`` coordinates ordered
                as dlib emits them (points 48–67): outer lip 0–11 with 0 and
                6 the corners, inner lip 12–19 (unused here).

        Returns:
            The MAR as a float. Returns ``0.0`` when the mouth width is zero
            (degenerate landmarks), so callers never see a division error.

        Raises:
            ValueError: if ``mouth_points`` is not of shape ``(20, 2)``.
        """
        pts = np.asarray(mouth_points, dtype=np.float64)
        if pts.shape != (MOUTH_POINTS, 2):
            raise ValueError(
                f"mouth_points must have shape ({MOUTH_POINTS}, 2), got {pts.shape}"
            )

        # Vertical outer-lip distances: p50-p58, p51-p57, p52-p56
        # (0-based within the mouth slice: 2-10, 3-9, 4-8).
        vertical_a = euclidean(pts[2], pts[10])
        vertical_b = euclidean(pts[3], pts[9])
        vertical_c = euclidean(pts[4], pts[8])

        # Horizontal mouth width: p48-p54 (0-based: 0-6).
        horizontal = euclidean(pts[0], pts[6])

        # A zero-width mouth can only come from a broken detection; treat it
        # as closed rather than crashing the pipeline.
        if horizontal == 0.0:
            return 0.0

        return float((vertical_a + vertical_b + vertical_c) / (2.0 * horizontal))

    def compute_mar(self, landmarks: np.ndarray) -> float:
        """
        Compute the MAR from a full 68-point landmark array.

        Args:
            landmarks: ``(68, 2)`` array of dlib facial landmarks.

        Returns:
            The MAR as a float.

        Raises:
            ValueError: if ``landmarks`` does not contain at least 68 rows.
        """
        lm = np.asarray(landmarks, dtype=np.float64)
        if lm.ndim != 2 or lm.shape[0] < MOUTH[1] or lm.shape[1] != 2:
            raise ValueError(
                f"landmarks must be a (68, 2) array, got shape {lm.shape}"
            )
        return self.compute_mouth_mar(lm[MOUTH[0]:MOUTH[1]])

    def normalize(self, mar: float, baseline: float) -> float:
        """
        Express a MAR value relative to the driver's calibrated baseline.

        A normalised value of ``1.0`` means "mouth as closed as during
        calibration"; values well above ``1.0`` indicate an open mouth.

        Args:
            mar: Raw MAR for the current frame.
            baseline: The driver's closed-mouth MAR from calibration.

        Returns:
            ``mar / baseline``, or ``1.0`` if ``baseline`` is zero.
        """
        # Guard against a zero baseline (e.g. calibration never ran). Unlike
        # EAR, where 0.0 maps to the worst case, a missing mouth baseline
        # should contribute *nothing* to the FRS, so return the baseline
        # ratio (zero excess) rather than 0.0.
        if baseline == 0.0:
            return 1.0
        return float(mar / baseline)

    def is_mouth_open(self, mar: float, threshold: float) -> bool:
        """
        Decide whether the mouth is open wide enough to be a candidate yawn.

        Args:
            mar: Raw MAR for the current frame.
            threshold: Yawn threshold, normally ``mar_baseline *
                YAWN_MAR_RATIO`` from the calibration module.

        Returns:
            ``True`` if ``mar`` is strictly above ``threshold``.
        """
        return mar > threshold


class YawnDetector:
    """
    Duration-gated yawn state machine over per-frame MAR values.

    Mirrors :class:`modules.blink.BlinkDetector`: feed it one MAR per frame
    and it tracks whether the mouth is currently open past the yawn
    threshold, for how long, and whether that opening has lasted long
    enough to be a yawn.

    Typical usage::

        yawns = YawnDetector()
        event = yawns.update(mar, thresholds["yawn_threshold"], now)
        if event:                      # a yawn was just *confirmed*
            log(event["duration_ms"])
        if yawns.is_yawning:           # confirmed yawn still in progress
            frs_input = mar_norm
        else:
            frs_input = 1.0

    Attributes:
        min_duration: Seconds the mouth must stay open before an opening
            counts as a yawn.
        mouth_open: Whether the MAR was above threshold on the last frame.
        open_time: Timestamp at which the current opening started.
        is_yawning: ``True`` from the moment the opening is confirmed as a
            yawn until the mouth closes again.
        yawn_timestamps: Confirmation time of every yawn since ``reset()``.
        yawn_durations: Total open duration (ms) of every completed yawn.
    """

    def __init__(self, min_duration: float = MIN_YAWN_DURATION_S) -> None:
        """
        Create a detector with the mouth closed.

        Args:
            min_duration: Seconds of continuous opening required to confirm
                a yawn.
        """
        self.min_duration: float = min_duration
        self.mouth_open: bool = False
        self.open_time: float = 0.0
        self.is_yawning: bool = False
        self.yawn_timestamps: List[float] = []
        self.yawn_durations: List[float] = []

    def update(
        self,
        mar: float,
        threshold: float,
        timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Feed one frame's MAR into the state machine.

        Transitions:

        * closed → open when ``mar > threshold``: remember the open time.
        * open, not yet yawning, held ≥ ``min_duration``: confirm the yawn
          and return an event (once per opening).
        * open → closed when ``mar <= threshold``: end the opening; if it
          was a confirmed yawn, record its total duration.

        A single frame at or below the threshold closes the opening, so
        speech (which dips below the threshold many times a second) never
        accumulates towards ``min_duration``.

        Args:
            mar: Raw MAR for the current frame.
            threshold: The driver's calibrated yawn threshold.
            timestamp: Frame time in seconds (``time.time()`` style).
                Defaults to the wall clock; pass explicitly for deterministic
                tests or video replay.

        Returns:
            ``{"timestamp": float, "duration_ms": float}`` on the frame a
            yawn is *confirmed* (``duration_ms`` is the opening so far,
            i.e. ≈ ``min_duration``), otherwise ``None``.
        """
        now = time.time() if timestamp is None else timestamp
        currently_open = mar > threshold

        if currently_open and not self.mouth_open:
            # Mouth just opened past the yawn threshold — start timing.
            self.mouth_open = True
            self.open_time = now
            return None

        if currently_open and self.mouth_open and not self.is_yawning:
            held = now - self.open_time
            if held >= self.min_duration:
                self.is_yawning = True
                self.yawn_timestamps.append(now)
                return {"timestamp": now, "duration_ms": held * 1000.0}
            return None

        if not currently_open and self.mouth_open:
            # Mouth closed — the opening is over.
            self.mouth_open = False
            if self.is_yawning:
                self.is_yawning = False
                self.yawn_durations.append((now - self.open_time) * 1000.0)
            return None

        # No transition (still closed, or a confirmed yawn still in progress).
        return None

    def get_open_duration(self, current_time: Optional[float] = None) -> float:
        """
        Seconds the mouth has been continuously open past the threshold.

        Args:
            current_time: Reference "now". Defaults to ``time.time()``.

        Returns:
            ``0.0`` if the mouth is closed.
        """
        if not self.mouth_open:
            return 0.0
        now = time.time() if current_time is None else current_time
        return float(now - self.open_time)

    def get_yawn_count(self, window_seconds: Optional[float] = None,
                       current_time: Optional[float] = None) -> int:
        """
        Number of confirmed yawns, optionally within a trailing window.

        Args:
            window_seconds: Only count yawns confirmed within this many
                seconds of ``current_time``; ``None`` counts all since
                ``reset()``.
            current_time: Reference "now". Defaults to ``time.time()``.

        Returns:
            The yawn count.
        """
        if window_seconds is None:
            return len(self.yawn_timestamps)
        now = time.time() if current_time is None else current_time
        cutoff = now - window_seconds
        return sum(1 for t in self.yawn_timestamps if t >= cutoff)

    def get_status(self, current_time: Optional[float] = None) -> Dict[str, object]:
        """
        Snapshot for the overlay.

        Args:
            current_time: Reference "now". Defaults to ``time.time()``.

        Returns:
            ``{"state": "closed"|"open"|"yawning", "open_s": float,
            "count": int}`` — ``open_s`` is the current opening's duration.
        """
        if self.is_yawning:
            state = "yawning"
        elif self.mouth_open:
            state = "open"
        else:
            state = "closed"
        return {
            "state": state,
            "open_s": self.get_open_duration(current_time),
            "count": len(self.yawn_timestamps),
        }

    def reset(self) -> None:
        """Clear all state and history (e.g. when the driver changes)."""
        self.mouth_open = False
        self.open_time = 0.0
        self.is_yawning = False
        self.yawn_timestamps.clear()
        self.yawn_durations.clear()
