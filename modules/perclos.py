"""
Module 5 — PERCLOS (PERcentage of eye CLOSure).

PERCLOS is the proportion of time, over a sliding window, that the eyes are
closed. It is one of the most validated drowsiness indicators in the driving
literature (Wierwille et al., 1994) and is the strongest single term in the
FRS (Module 6)::

    PERCLOS = (frames with EAR < threshold) / (frames in window) * 100

Here the window is expressed in frames (``window_seconds * fps``) rather than
wall-clock seconds so the rolling buffer stays a fixed size regardless of
timing jitter. On the Pi 4B at 30 fps a 60 s window is 1 800 booleans, which
``collections.deque`` handles trivially.

The window defaults come from ``config.PERCLOS_WINDOW_SECONDS`` and
``config.CAMERA_FPS``.
"""

from collections import deque
from typing import Deque

from config import config

# Upper bound on the normalised PERCLOS. A driver whose calibration PERCLOS is
# ~5 % who then closes their eyes for the full window would otherwise produce
# a ratio of 20, swamping every other term in the FRS.
MAX_NORMALIZED_PERCLOS: float = 5.0


class PERCLOSCalculator:
    """
    Maintain a rolling buffer of eye-closed flags and report PERCLOS.

    Typical usage (once per frame)::

        perclos_calc = PERCLOSCalculator()          # 60 s @ 30 fps
        perclos = perclos_calc.update(ear, threshold)
        perclos_norm = perclos_calc.normalize(perclos, baselines["perclos_baseline"])

    Attributes:
        window_seconds: Length of the rolling window in seconds.
        fps: Expected frame rate used to size the buffer.
        max_frames: ``window_seconds * fps`` — the buffer capacity.
        frames: Deque of booleans, ``True`` = eyes closed on that frame.
    """

    def __init__(
        self,
        window_seconds: int = config.PERCLOS_WINDOW_SECONDS,
        fps: int = config.CAMERA_FPS,
    ) -> None:
        """
        Create a calculator with an empty buffer.

        Args:
            window_seconds: Rolling window length in seconds.
            fps: Frame rate the pipeline is expected to run at. Together with
                ``window_seconds`` this fixes the buffer size in frames.
        """
        self.window_seconds: int = window_seconds
        self.fps: int = fps
        self.max_frames: int = max(1, window_seconds * fps)

        # ``maxlen`` makes the deque evict the oldest frame automatically once
        # it is full, so ``update()`` never has to trim manually.
        self.frames: Deque[bool] = deque(maxlen=self.max_frames)

    def update(self, ear: float, threshold: float) -> float:
        """
        Record the current frame and return the updated PERCLOS.

        Args:
            ear: Raw EAR for the current frame.
            threshold: The driver's calibrated closure threshold.

        Returns:
            PERCLOS over the current window, in the range ``0.0``–``100.0``.
        """
        self.frames.append(ear < threshold)
        return self.get_perclos()

    def get_perclos(self) -> float:
        """
        Return the PERCLOS for the frames currently in the buffer.

        Until the buffer fills, the percentage is taken over however many
        frames have been seen so far, so the value is meaningful (if noisier)
        from the very first frames of a session.

        Returns:
            Percentage of closed-eye frames, ``0.0`` if the buffer is empty.
        """
        total = len(self.frames)
        if total == 0:
            return 0.0
        closed = sum(self.frames)  # bools sum as 0/1
        return float(closed / total * 100.0)

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
        self.frames.clear()
