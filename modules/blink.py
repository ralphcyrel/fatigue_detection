"""
Module 4 — Blink detection.

`BlinkDetector` turns the per-frame EAR stream from Module 3 into discrete
blink events. A blink is the interval between the EAR dropping below the
driver's closure threshold and rising back above it. From those events two
fatigue indicators are derived:

* **Blink duration** — drowsy drivers blink more slowly; a healthy blink lasts
  roughly 100–400 ms, while fatigued blinks stretch well beyond that.
* **Blink frequency** — the number of blinks within a rolling window
  (60 s by default). Frequency tends to rise with fatigue as the driver
  fights to keep their eyes open, then falls off during microsleeps.

Both indicators are normalised against the driver's calibrated baseline so
the FRS (Module 6) can compare drivers with naturally different blink habits.

The detector is deliberately simple (a two-state machine) and stateful; call
``update()`` once per frame and ``reset()`` whenever a new driver sits down.
"""

import time
from typing import Dict, List, Optional

import numpy as np

# Blinks shorter than this are treated as landmark jitter, not real blinks.
# A genuine blink takes ≥ ~100 ms; 50 ms gives headroom for frame timing.
MIN_BLINK_DURATION_MS: float = 50.0

# How many of the most recent blinks feed ``get_average_duration()``. A short
# window makes the metric responsive to the driver's *current* state.
AVERAGE_WINDOW: int = 10


class BlinkDetector:
    """
    Detect blink events from a stream of EAR values and summarise them.

    Typical usage (once per frame)::

        detector = BlinkDetector(frequency_window=60)
        event = detector.update(ear, threshold)
        if event is not None:
            print(f"blink lasted {event['duration_ms']:.0f} ms")
        freq = detector.get_blink_frequency()       # blinks in last 60 s
        avg_ms = detector.get_average_duration()    # mean of last 10 blinks

    Attributes:
        frequency_window: Rolling window (seconds) for frequency counting.
        eyes_closed: ``True`` while the eyes are currently below threshold.
        close_time: ``time.time()`` at which the current closure began.
        blink_durations: Durations (ms) of every valid blink this session.
        blink_timestamps: Completion timestamps of every valid blink.
    """

    def __init__(self, frequency_window: int = 60) -> None:
        """
        Create a detector with empty state.

        Args:
            frequency_window: Length in seconds of the rolling window used by
                ``get_blink_frequency()``. 60 s makes the result directly
                interpretable as "blinks per minute".
        """
        self.frequency_window: int = frequency_window

        self.eyes_closed: bool = False
        self.close_time: float = 0.0
        self.blink_durations: List[float] = []
        self.blink_timestamps: List[float] = []

    def update(
        self,
        ear: float,
        threshold: float,
        timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Feed one frame's EAR into the state machine.

        Transitions:

        * open → closed when ``ear < threshold``: remember the close time.
        * closed → open when ``ear >= threshold``: measure the closure and, if
          it is long enough to be a real blink, record it and return an event.

        Args:
            ear: Raw EAR for the current frame.
            threshold: The driver's calibrated closure threshold.
            timestamp: Frame time in seconds (``time.time()`` style). Defaults
                to the wall clock; pass explicitly for deterministic tests or
                when replaying recorded video.

        Returns:
            ``{"duration_ms": float, "timestamp": float}`` when a valid blink
            just completed, otherwise ``None``.
        """
        now = time.time() if timestamp is None else timestamp
        currently_closed = ear < threshold

        if currently_closed and not self.eyes_closed:
            # Eyes just shut — start timing the closure.
            self.eyes_closed = True
            self.close_time = now
            return None

        if not currently_closed and self.eyes_closed:
            # Eyes just reopened — the closure is over.
            self.eyes_closed = False
            duration_ms = (now - self.close_time) * 1000.0

            # Reject sub-50 ms closures: those are almost always a single
            # jittery frame rather than an actual eyelid movement.
            if duration_ms < MIN_BLINK_DURATION_MS:
                return None

            self.blink_durations.append(duration_ms)
            self.blink_timestamps.append(now)
            return {"duration_ms": duration_ms, "timestamp": now}

        # No transition (still open, or still closed) — nothing to report.
        return None

    def get_blink_frequency(self, current_time: Optional[float] = None) -> float:
        """
        Count blinks that completed within the last ``frequency_window`` s.

        Args:
            current_time: Reference "now" in seconds. Defaults to wall clock.

        Returns:
            Number of blinks in the window, as a float (blinks per window).
        """
        now = time.time() if current_time is None else current_time
        window_start = now - self.frequency_window

        # Timestamps are appended in order, so a linear scan from the end
        # would be faster, but the list is small enough that clarity wins.
        count = sum(1 for ts in self.blink_timestamps if ts >= window_start)
        return float(count)

    def get_average_duration(self) -> float:
        """
        Mean duration of the most recent ``AVERAGE_WINDOW`` (10) blinks.

        Returns:
            Mean duration in milliseconds, or ``0.0`` if no blinks yet.
        """
        if not self.blink_durations:
            return 0.0
        recent = self.blink_durations[-AVERAGE_WINDOW:]
        return float(np.mean(recent))

    def normalize_duration(
        self, duration_ms: float, baseline_duration_ms: float
    ) -> float:
        """
        Express a blink duration relative to the driver's baseline.

        Args:
            duration_ms: Measured (or averaged) blink duration in ms.
            baseline_duration_ms: Calibrated alert-state blink duration in ms.

        Returns:
            ``duration_ms / baseline_duration_ms``, or ``0.0`` if the
            baseline is zero.
        """
        if baseline_duration_ms == 0.0:
            return 0.0
        return float(duration_ms / baseline_duration_ms)

    def normalize_frequency(
        self, frequency: float, baseline_frequency: float
    ) -> float:
        """
        Express a blink frequency relative to the driver's baseline.

        Args:
            frequency: Blinks per window as returned by
                ``get_blink_frequency()``.
            baseline_frequency: Calibrated alert-state blink frequency.

        Returns:
            ``frequency / baseline_frequency``, or ``0.0`` if the baseline
            is zero.
        """
        if baseline_frequency == 0.0:
            return 0.0
        return float(frequency / baseline_frequency)

    def reset(self) -> None:
        """Clear all blink history and closure state for a new driver."""
        self.eyes_closed = False
        self.close_time = 0.0
        self.blink_durations.clear()
        self.blink_timestamps.clear()
