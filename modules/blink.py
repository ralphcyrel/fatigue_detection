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

Closures of :data:`MICROSLEEP_MIN_DURATION_S` or more are **not** blinks and
never reach either indicator; :class:`MicrosleepDetector` in this module
handles them as their own class of event. Both detectors are fed the same EAR
stream and apply the same ``ear < threshold`` test, so they always agree on
whether the eyes are shut.

The detector is deliberately simple (a two-state machine) and stateful; call
``update()`` once per frame and ``reset()`` whenever a new driver sits down.
"""

import time
from typing import Dict, List, Optional

import numpy as np

# Blinks shorter than this are treated as landmark jitter, not real blinks.
# A genuine blink takes ≥ ~100 ms; 50 ms gives headroom for frame timing.
MIN_BLINK_DURATION_MS: float = 50.0

# Closures at or beyond this are **microsleeps**, not blinks, and are handled
# by :class:`MicrosleepDetector` instead of entering ``blink_durations``.
#
# The split is three-way, not two-way:
#
#   < 50 ms        landmark jitter          discarded
#   50 ms - 1.0 s  blink, incl. slow ones   -> BlinkDetector
#   >= 1.0 s       microsleep               -> MicrosleepDetector
#
# The 500 ms - 1 s band deliberately stays with the blinks. Slow drowsy
# blinking is exactly what the FRS blink-duration term exists to measure, and
# adopting the literature's 500 ms microsleep floor here would strip that term
# of most of its useful range while firing the alarm on ordinary drowsy
# blinking. 1.0 s is where much of the driver-drowsiness implementation
# literature puts the line, for that same false-positive reason.
#
# Absolute rather than a multiple of the driver's baseline, unlike every other
# metric in the system: a microsleep is sleep intrusion, not "a long blink for
# this person" - a driver with a 400 ms baseline is not microsleeping any less
# at 1.2 s than one with a 180 ms baseline. ``MIN_YAWN_DURATION_S`` is
# absolute for the same reason. The *event* gate is absolute; the FRS
# magnitudes around it stay normalised.
MICROSLEEP_MIN_DURATION_S: float = 1.0
MICROSLEEP_MIN_DURATION_MS: float = MICROSLEEP_MIN_DURATION_S * 1000.0

# Trailing window (seconds) over which ``get_average_duration()`` averages.
# Matches ``config.PERCLOS_WINDOW_SECONDS`` and the default blink-frequency
# window on purpose: all three eye metrics should describe the same 60 s of
# driving. A count-based window cannot promise that - ten blinks span 40 s at
# 15 blinks/min but several minutes from a driver who has stopped blinking,
# and a stale mean is worst exactly when the driver is most impaired.
BLINK_MEAN_WINDOW_S: float = 60.0

# Floor on how many blinks the mean is taken over. A pure time window returns
# a one- or two-blink mean whenever blinking is sparse, and then a single
# mistimed blink swings the FRS duration term through its whole range. Below
# this count ``get_average_duration()`` falls back to the most recent blinks
# regardless of age: the estimate is knowingly stale, which is the honest
# trade against it being pure noise.
MIN_BLINKS_FOR_MEAN: int = 3

# Retained for callers that still want a count-based mean, and as the cap on
# how many blinks the fallback reaches back for.
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
        avg_ms = detector.get_average_duration(now)  # mean over last 60 s

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
            just completed, otherwise ``None``. A closure outside
            ``[MIN_BLINK_DURATION_MS, MICROSLEEP_MIN_DURATION_MS)`` is not a
            blink and returns ``None``: too short is landmark jitter, too long
            is a microsleep (see :class:`MicrosleepDetector`).
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

            # Reject closures of a second or more: those are microsleeps, a
            # different kind of event entirely, and letting one into
            # ``blink_durations`` both inflates the FRS blink-duration term
            # without bound and poisons the rolling mean for the next
            # ``AVERAGE_WINDOW`` blinks. :class:`MicrosleepDetector`, fed the
            # same EAR stream, reports these instead.
            if duration_ms >= MICROSLEEP_MIN_DURATION_MS:
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

    def get_average_duration(self, current_time: Optional[float] = None) -> float:
        """
        Mean duration of the blinks in the last :data:`BLINK_MEAN_WINDOW_S`.

        Time-bounded rather than count-bounded so the metric describes the
        same span of driving as PERCLOS and blink frequency, and so it
        actually recovers: a count-based mean over the last ten blinks never
        ages out for a driver who has stopped blinking, which is itself a
        fatigue signal.

        If fewer than :data:`MIN_BLINKS_FOR_MEAN` blinks fall inside the
        window, the most recent blinks are used regardless of age (up to
        :data:`AVERAGE_WINDOW`). That keeps the estimate stable when blinking
        is sparse, at the cost of it being knowingly stale - preferable to a
        one-blink mean that swings the FRS duration term through its full
        range on a single sample.

        Args:
            current_time: Reference "now" in seconds. Defaults to the wall
                clock; pass explicitly for deterministic tests or replay.

        Returns:
            Mean duration in milliseconds, or ``0.0`` if no blinks yet.
        """
        if not self.blink_durations:
            return 0.0
        now = time.time() if current_time is None else current_time
        cutoff = now - BLINK_MEAN_WINDOW_S
        # durations and timestamps are appended together in update(), so they
        # stay index-aligned and can be zipped directly.
        recent = [d for d, ts in zip(self.blink_durations, self.blink_timestamps)
                  if ts >= cutoff]
        if len(recent) < MIN_BLINKS_FOR_MEAN:
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


class MicrosleepDetector:
    """
    Duration-gated microsleep state machine over per-frame EAR values.

    Mirrors :class:`modules.mar.YawnDetector` rather than
    :class:`BlinkDetector`, and the difference is the whole point.
    ``BlinkDetector`` only records a closure when the eyes *reopen*, so during
    a 6 s microsleep it contributes nothing at all while the driver's eyes are
    shut, then reports an enormous duration once the danger has passed. The
    timing is exactly inverted. This detector instead confirms the event at
    the moment the closure crosses ``min_duration`` **with the eyes still
    closed**, which is when an alert is worth raising.

    Typical usage (once per frame, alongside ``BlinkDetector``)::

        microsleeps = MicrosleepDetector()
        event = microsleeps.update(ear, thresholds["ear_threshold"], now)
        if event:                        # a microsleep was just *confirmed*
            logger.warning("microsleep after %.1fs", event["duration_ms"] / 1000)
        if microsleeps.is_microsleeping:  # still closed
            level = "DANGER"

    Attributes:
        min_duration: Seconds the eyes must stay closed before the closure
            counts as a microsleep.
        eyes_closed: Whether the EAR was below threshold on the last frame.
        close_time: Timestamp at which the current closure started.
        is_microsleeping: ``True`` from the moment the closure is confirmed
            as a microsleep until the eyes reopen.
        min_ear: Lowest EAR seen since the current closure began; ``nan``
            before the first closure.
        microsleep_timestamps: Confirmation time of every microsleep since
            ``reset()``.
        microsleep_durations: Total closure duration (ms) of every completed
            microsleep.
    """

    def __init__(self, min_duration: float = MICROSLEEP_MIN_DURATION_S) -> None:
        """
        Create a detector with the eyes open.

        Args:
            min_duration: Seconds of continuous closure required to confirm a
                microsleep.
        """
        self.min_duration: float = min_duration
        self.eyes_closed: bool = False
        self.close_time: float = 0.0
        self.is_microsleeping: bool = False
        self.min_ear: float = float("nan")
        self.microsleep_timestamps: List[float] = []
        self.microsleep_durations: List[float] = []

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
        * closed, not yet microsleeping, held ≥ ``min_duration``: confirm the
          microsleep and return an event (once per closure).
        * closed → open when ``ear >= threshold``: end the closure; if it was
          a confirmed microsleep, record its total duration.

        A single frame at or above the threshold ends the closure, matching
        ``BlinkDetector``'s test exactly so the two can never disagree about
        whether the eyes are shut.

        Args:
            ear: Raw EAR for the current frame.
            threshold: The driver's calibrated closure threshold.
            timestamp: Frame time in seconds (``time.time()`` style). Defaults
                to the wall clock; pass explicitly for deterministic tests or
                when replaying recorded video.

        Returns:
            ``{"timestamp": float, "duration_ms": float, "min_ear": float}``
            on the frame a microsleep is *confirmed* (``duration_ms`` is the
            closure so far, i.e. ≈ ``min_duration`` × 1000), otherwise
            ``None``. ``min_ear`` is the lowest EAR observed since the closure
            began - always below ``threshold``, unlike the confirming frame's
            own EAR in the boundary case described below.

            Confirmation normally happens with the eyes still shut. The one
            exception is a closure that crosses ``min_duration`` between two
            frames and reopens before any frame could observe it past the
            line; that is confirmed on the reopening frame instead, so no
            closure can be rejected by *both* detectors. Together the two
            detectors partition every closure ≥
            :data:`MIN_BLINK_DURATION_MS` into exactly one bucket.
        """
        now = time.time() if timestamp is None else timestamp
        currently_closed = ear < threshold

        # Deepest (lowest) EAR seen since this closure began. Reported on the
        # event so the log can show how far the eye actually shut, rather than
        # whatever the confirming frame happened to read - which, in the
        # boundary case below, is the *reopening* frame and therefore sits
        # above the threshold.
        if currently_closed:
            self.min_ear = ear if not self.eyes_closed else min(self.min_ear, ear)

        if currently_closed and not self.eyes_closed:
            # Eyes just shut — start timing the closure.
            self.eyes_closed = True
            self.close_time = now
            return None

        if currently_closed and self.eyes_closed and not self.is_microsleeping:
            held = now - self.close_time
            if held >= self.min_duration:
                self.is_microsleeping = True
                self.microsleep_timestamps.append(now)
                return {"timestamp": now, "duration_ms": held * 1000.0,
                        "min_ear": self.min_ear}
            return None

        if not currently_closed and self.eyes_closed:
            # Eyes reopened — the closure is over.
            self.eyes_closed = False
            duration_ms = (now - self.close_time) * 1000.0
            if self.is_microsleeping:
                self.is_microsleeping = False
                self.microsleep_durations.append(duration_ms)
                return None
            if duration_ms >= self.min_duration * 1000.0:
                # Boundary case: the closure crossed ``min_duration`` in the
                # gap *between* two frames, so no frame ever observed it while
                # the eyes were still shut. ``BlinkDetector`` measures the same
                # closure from the reopen timestamp and rejects it as too long,
                # so without this the event would fall through both detectors
                # and vanish. Confirm it now instead, one frame late.
                self.microsleep_timestamps.append(now)
                self.microsleep_durations.append(duration_ms)
                # ``min_ear`` is the minimum over the *closed* frames, so it
                # excludes this reopening frame's (above-threshold) value.
                return {"timestamp": now, "duration_ms": duration_ms,
                        "min_ear": self.min_ear}
            return None

        # No transition (still open, or a confirmed microsleep still running).
        return None

    def get_closed_duration(self, current_time: Optional[float] = None) -> float:
        """
        Seconds the eyes have been continuously closed.

        Args:
            current_time: Reference "now". Defaults to ``time.time()``.

        Returns:
            ``0.0`` if the eyes are open.
        """
        if not self.eyes_closed:
            return 0.0
        now = time.time() if current_time is None else current_time
        return float(now - self.close_time)

    def get_microsleep_count(
        self,
        window_seconds: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> int:
        """
        Number of confirmed microsleeps, optionally within a trailing window.

        Args:
            window_seconds: Only count microsleeps confirmed within this many
                seconds of ``current_time``; ``None`` counts all since
                ``reset()``.
            current_time: Reference "now". Defaults to ``time.time()``.

        Returns:
            The microsleep count.
        """
        if window_seconds is None:
            return len(self.microsleep_timestamps)
        now = time.time() if current_time is None else current_time
        cutoff = now - window_seconds
        return sum(1 for t in self.microsleep_timestamps if t >= cutoff)

    def get_status(self, current_time: Optional[float] = None) -> Dict[str, object]:
        """
        Snapshot for the overlay.

        Args:
            current_time: Reference "now". Defaults to ``time.time()``.

        Returns:
            ``{"state": "open"|"closed"|"microsleep", "closed_s": float,
            "count": int}`` — ``closed_s`` is the current closure's duration.
        """
        if self.is_microsleeping:
            state = "microsleep"
        elif self.eyes_closed:
            state = "closed"
        else:
            state = "open"
        return {
            "state": state,
            "closed_s": self.get_closed_duration(current_time),
            "count": len(self.microsleep_timestamps),
        }

    def note_no_face(
        self, current_time: Optional[float] = None
    ) -> Optional[Dict[str, object]]:
        """
        Abandon any closure in progress (no landmarks this frame).

        Without this the closure timer would keep running across a gap in
        which no EAR was observed at all, and the next frame with a face
        could confirm a "microsleep" that was really a lost face. An active
        microsleep is dropped too, matching how
        ``MetricsPipeline.note_no_face`` clears the head-pose override.

        Args:
            current_time: Reference "now". Defaults to ``time.time()``.

        Returns:
            ``None`` if the eyes were already open and nothing was in
            flight, otherwise a description of what was abandoned::

                {"was_microsleeping": bool, "closed_s": float}

            ``was_microsleeping`` distinguishes the case worth attention -
            the face was lost while the driver was *already* confirmed
            microsleeping - from an ordinary blink interrupted by a dropped
            frame. The caller decides how loudly to report it.
        """
        if not self.eyes_closed:
            return None
        abandoned: Dict[str, object] = {
            "was_microsleeping": self.is_microsleeping,
            "closed_s": self.get_closed_duration(current_time),
        }
        self.eyes_closed = False
        self.close_time = 0.0
        self.is_microsleeping = False
        self.min_ear = float("nan")
        return abandoned

    def reset(self) -> None:
        """Clear all microsleep history and closure state for a new driver."""
        self.eyes_closed = False
        self.close_time = 0.0
        self.is_microsleeping = False
        self.min_ear = float("nan")
        self.microsleep_timestamps.clear()
        self.microsleep_durations.clear()
