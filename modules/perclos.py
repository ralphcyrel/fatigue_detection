"""
Module 5 — PERCLOS (PERcentage of eye CLOSure).

PERCLOS is the proportion of time, over a sliding window, that the eyes are
closed. It is one of the most validated drowsiness indicators in the driving
literature (Wierwille et al., 1994) and is the strongest single term in the
FRS (Module 6)::

    PERCLOS = (frames with EAR < threshold) / (frames in window) * 100

The window is wall-clock seconds (``config.PERCLOS_WINDOW_SECONDS``), not a
frame count. Until 2026-09-28 it was ``window_seconds * config.CAMERA_FPS``
frames, which assumed the loop ran at the camera's 30 fps; the Pi runs at
~20 fps, so the "60 s" window actually covered ~90 s.

Warm-up: a window that has only seen a few seconds of frames swings wildly -
one 2-frame blink 1.5 s into a session read as 6.7 % PERCLOS and scored DANGER
on its own. When the driver's baseline is known, the unobserved part of the
window is assumed to be at that baseline rate, up to :data:`WARMUP_PRIOR_S`
seconds' worth; the assumption fades out as the window fills, so a full
window is pure measurement (and matches how calibration measured the
baseline). Replayed over all recorded assessments this kept every verdict
and delayed DANGER onset by 0.10-0.28 s.
"""

from collections import deque
from typing import Deque, Optional, Tuple

from config import config

# Upper bound on the normalised PERCLOS. A driver whose calibration PERCLOS is
# ~5 % who then closes their eyes for the full window would otherwise produce
# a ratio of 20, swamping every other term in the FRS.
MAX_NORMALIZED_PERCLOS: float = 5.0

# Seconds of the not-yet-observed window counted at the driver's baseline
# rate (see module docstring). The weight is min(this, window - observed), so
# it is this much for the first (window - this) seconds, then falls to zero
# as the window fills. At 10 s, a driver at 10 % true PERCLOS (baseline
# 1.86 %) reaches a PERCLOS term of 0.65 after ~8 s, and at 20 % after ~2.5 s;
# the microsleep and head-pose overrides are unaffected.
WARMUP_PRIOR_S: float = 10.0


class PERCLOSCalculator:
    """
    Maintain a time-bounded buffer of eye-closed flags and report PERCLOS.

    Typical usage (once per frame)::

        perclos_calc = PERCLOSCalculator()          # 60 s window
        perclos = perclos_calc.update(ear, threshold, time.monotonic(),
                                      baseline=baselines["perclos_baseline"])
        perclos_norm = perclos_calc.normalize(perclos, baselines["perclos_baseline"])

    Attributes:
        window_seconds: Length of the rolling window in seconds.
        prior_s: Seconds of warm-up prior (:data:`WARMUP_PRIOR_S`).
        samples: Deque of ``(timestamp, closed)`` within the window.
    """

    def __init__(
        self,
        window_seconds: float = config.PERCLOS_WINDOW_SECONDS,
        prior_s: float = WARMUP_PRIOR_S,
    ) -> None:
        """
        Create a calculator with an empty buffer.

        Args:
            window_seconds: Rolling window length in seconds.
            prior_s: Warm-up prior in seconds; ``0`` disables it.
        """
        self.window_seconds: float = float(window_seconds)
        self.prior_s: float = float(prior_s)
        self.samples: Deque[Tuple[float, bool]] = deque()
        # Timestamp of the first sample since reset: how much of the window
        # has been observed, which the first sample still in the deque stops
        # telling us once the window starts sliding.
        self._first_t: Optional[float] = None
        self._last: float = 0.0

    def update(
        self,
        ear: float,
        threshold: float,
        now: float,
        baseline: Optional[float] = None,
    ) -> float:
        """
        Record the current frame and return the updated PERCLOS.

        Args:
            ear: Raw EAR for the current frame.
            threshold: The driver's closure threshold.
            now: Timestamp in seconds (``time.monotonic()``).
            baseline: The driver's calibrated PERCLOS (percent) for the
                warm-up prior. ``None`` (enrollment, where it is being
                measured) gives the plain frame ratio from the first frame.

        Returns:
            PERCLOS over the current window, in the range ``0.0``–``100.0``.
        """
        if self._first_t is None:
            self._first_t = now
        self.samples.append((now, ear < threshold))
        while now - self.samples[0][0] > self.window_seconds:
            self.samples.popleft()

        measured = sum(c for _, c in self.samples) / len(self.samples) * 100.0
        observed = min(now - self._first_t, self.window_seconds)
        prior = min(self.prior_s, self.window_seconds - observed)
        if baseline is None or prior <= 0.0:
            self._last = float(measured)
        else:
            self._last = float((measured * observed + baseline * prior) / (observed + prior))
        return self._last

    def get_perclos(self) -> float:
        """The value returned by the last :meth:`update` (``0.0`` before any)."""
        return self._last

    def normalize(self, perclos: float, baseline_perclos: float) -> float:
        """
        Express PERCLOS relative to the driver's calibrated baseline.

        Args:
            perclos: Current PERCLOS percentage.
            baseline_perclos: Alert-state PERCLOS from calibration.

        Returns:
            ``perclos / baseline_perclos`` capped at ``5.0``, or ``0.0`` if
            the baseline is zero.
        """
        if baseline_perclos == 0.0:
            return 0.0
        ratio = perclos / baseline_perclos
        # Cap so a single extreme reading cannot dominate the composite FRS.
        return float(min(ratio, MAX_NORMALIZED_PERCLOS))

    def reset(self) -> None:
        """Discard all buffered frames (call when a new driver is detected)."""
        self.samples.clear()
        self._first_t = None
        self._last = 0.0
