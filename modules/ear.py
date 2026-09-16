"""
Module 3 — Eye Aspect Ratio (EAR).

The EAR is a scale-invariant measure of how open an eye is, computed purely
from the six dlib landmarks that outline each eye (Soukupová & Čech, 2016)::

    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

``p1`` and ``p4`` are the horizontal corners of the eye, while ``p2/p6`` and
``p3/p5`` are the two vertical pairs on the upper and lower eyelids. When the
eye is open the vertical distances are large relative to the horizontal one
(EAR ≈ 0.25–0.35); when it closes they collapse towards zero (EAR ≈ 0.05–0.15).
Because the ratio is dimensionless it is unaffected by the driver's distance
from the camera.

Downstream, the blink detector (Module 4) and PERCLOS (Module 5) both consume
the per-frame EAR produced here, and the calibration module (Module 7) uses it
to learn each driver's personal open-eye baseline.
"""

from typing import Tuple

import numpy as np
from scipy.spatial.distance import euclidean

# ---------------------------------------------------------------------------
# Eye landmark index ranges (half-open, matching ``modules.landmark``).
# They are duplicated here rather than imported so this module stays free of
# the dlib/cv2 dependency that ``modules.landmark`` pulls in — that keeps EAR
# unit-testable on a machine without dlib. Keep in sync with landmark.py.
# ---------------------------------------------------------------------------
LEFT_EYE: Tuple[int, int] = (42, 48)
RIGHT_EYE: Tuple[int, int] = (36, 42)

# Number of landmark points that outline a single eye.
EYE_POINTS: int = 6


class EARCalculator:
    """
    Compute the Eye Aspect Ratio for one eye or the average over both eyes.

    Typical usage::

        ear_calc = EARCalculator()
        ear = ear_calc.compute_average_ear(landmarks)      # landmarks: (68, 2)
        ear_norm = ear_calc.normalize(ear, baselines["ear_baseline"])
        closed = ear_calc.is_eye_closed(ear, baselines["ear_threshold"])
    """

    def __init__(self) -> None:
        """Initialise the calculator. No configuration is required."""
        # Stateless by design — every call is a pure function of its inputs so
        # the same instance can be shared safely across frames and drivers.
        pass

    def compute_ear(self, eye_points: np.ndarray) -> float:
        """
        Compute the EAR for a single eye.

        Args:
            eye_points: ``(6, 2)`` array of ``(x, y)`` coordinates ordered as
                dlib emits them: ``p1`` = outer corner, ``p2``/``p3`` = upper
                lid, ``p4`` = inner corner, ``p5``/``p6`` = lower lid.

        Returns:
            The EAR as a float. Returns ``0.0`` when the horizontal distance is
            zero (degenerate landmarks), so callers never see a division error.

        Raises:
            ValueError: if ``eye_points`` is not of shape ``(6, 2)``.
        """
        pts = np.asarray(eye_points, dtype=np.float64)
        if pts.shape != (EYE_POINTS, 2):
            raise ValueError(
                f"eye_points must have shape ({EYE_POINTS}, 2), got {pts.shape}"
            )

        # Vertical eyelid distances: p2-p6 and p3-p5 (0-based: 1-5 and 2-4).
        vertical_a = euclidean(pts[1], pts[5])
        vertical_b = euclidean(pts[2], pts[4])

        # Horizontal eye width: p1-p4 (0-based: 0-3).
        horizontal = euclidean(pts[0], pts[3])

        # A zero-width eye can only come from a broken detection (e.g. all
        # points collapsed to one pixel); treat it as fully closed rather than
        # crashing the pipeline.
        if horizontal == 0.0:
            return 0.0

        return float((vertical_a + vertical_b) / (2.0 * horizontal))

    def compute_average_ear(self, landmarks: np.ndarray) -> float:
        """
        Compute the mean EAR across the left and right eyes.

        Averaging both eyes suppresses noise from a single mis-placed landmark
        and makes the measure robust to slight head yaw, where one eye is seen
        more obliquely than the other.

        Args:
            landmarks: ``(68, 2)`` array of dlib facial landmarks.

        Returns:
            ``(EAR_left + EAR_right) / 2`` as a float.

        Raises:
            ValueError: if ``landmarks`` does not contain at least 48 rows
                (the highest eye index + 1).
        """
        lm = np.asarray(landmarks, dtype=np.float64)
        if lm.ndim != 2 or lm.shape[0] < LEFT_EYE[1] or lm.shape[1] != 2:
            raise ValueError(
                f"landmarks must be a (68, 2) array, got shape {lm.shape}"
            )

        left_eye = lm[LEFT_EYE[0]:LEFT_EYE[1]]
        right_eye = lm[RIGHT_EYE[0]:RIGHT_EYE[1]]

        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)

        return float((left_ear + right_ear) / 2.0)

    def normalize(self, ear: float, baseline: float) -> float:
        """
        Express an EAR value relative to the driver's calibrated baseline.

        A normalised value of ``1.0`` means "eyes as open as during
        calibration"; values well below ``1.0`` indicate drooping or closure.

        Args:
            ear: Raw EAR for the current frame.
            baseline: The driver's mean alert-state EAR from calibration.

        Returns:
            ``ear / baseline``, or ``0.0`` if ``baseline`` is zero.
        """
        # Guard against a zero baseline (e.g. calibration never ran); the FRS
        # module clamps 1/EAR_norm so returning 0.0 here is safe downstream.
        if baseline == 0.0:
            return 0.0
        return float(ear / baseline)

    def is_eye_closed(self, ear: float, threshold: float) -> bool:
        """
        Decide whether the eyes are closed in the current frame.

        Args:
            ear: Raw EAR for the current frame.
            threshold: Closure threshold, normally ``ear_baseline * 0.75`` from
                the calibration module.

        Returns:
            ``True`` if ``ear`` is strictly below ``threshold``.
        """
        return ear < threshold
