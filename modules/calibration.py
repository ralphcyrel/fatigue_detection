"""
Module 7 — Per-driver alert-state calibration.

Eye geometry and blink habits vary a lot between people: one driver's normal
open-eye EAR may be another's half-closed. Fixed thresholds therefore produce
false alarms for some drivers and miss fatigue in others. `CalibrationManager`
solves this by observing the driver for ``config.CALIBRATION_DURATION``
seconds at the start of a session — while they are assumed to be alert — and
deriving personal baselines:

* ``ear_baseline``             mean open-eye EAR
* ``ear_threshold``            75 % of ``ear_baseline`` → "eye closed" cut-off
* ``perclos_baseline``         mean PERCLOS (floored at 0.5 %)
* ``blink_duration_baseline``  mean blink duration (floored at 50 ms)
* ``blink_frequency_baseline`` blinks completed / duration, as blinks per
                               minute (floored at 1 blink/min)
* ``mar_baseline``             **median** closed-mouth MAR (floored at 0.2)
* ``yawn_threshold``           ``mar_baseline * 2.0`` → "candidate yawn" cut-off

The floors exist only to keep a degenerate calibration (no blinks detected
at all) from producing a zero baseline that the normalised metrics would
divide by. They sit at the smallest value the pipeline can physically
measure, so a genuine alert driver is never floored: a recorded blink is
at least ``MIN_BLINK_DURATION_MS`` long, one blink in the window is 1 per
minute, and 0.5 % PERCLOS is fewer closed frames than three short blinks.
Whenever a floor does engage, ``compute_baselines()`` logs a warning naming
the metric - blink detection evidently did not work during calibration,
and the resulting profile should be treated as suspect.

The blink-frequency baseline is derived from the number of blinks that
completed during calibration rather than from ``BlinkDetector``'s rolling
count. That counter starts empty at the same instant as calibration, so
sampling it every frame yields a ramp whose mean is only a fraction of the
true rate. The per-frame samples are still kept in ``blink_frequencies``
for diagnostics.

The MAR baseline uses the median rather than the mean on purpose: a driver
who talks during the 60 s calibration produces a right-skewed MAR sample
(mostly closed, with open-mouth spikes), and a mean would inflate their
baseline and hence their yawn threshold. The median is the resting
closed-mouth value regardless. The MAR keys are only present in
``compute_baselines()`` when MAR samples were supplied to ``update()``.

Progress is measured in wall-clock time rather than frame count so a slow
frame rate merely reduces the number of samples instead of stretching the
calibration period. Every time-dependent method accepts an optional
``timestamp`` so the module can be driven deterministically in tests or
video replays.
"""

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from config import config
from modules.blink import MIN_BLINK_DURATION_MS
from modules.mar import YAWN_MAR_RATIO

logger = logging.getLogger(__name__)

# Fraction of the baseline EAR below which the eye is considered closed.
# 0.75 sits comfortably between typical open (≈0.30) and closed (≈0.10) EARs.
EAR_THRESHOLD_RATIO: float = 0.75

# Lower bounds applied to the computed baselines (see module docstring).
# Each is the smallest value the pipeline can measure, so it engages only on
# a degenerate run, never on a quiet-but-genuine one.
MIN_PERCLOS_BASELINE: float = 0.5                          # percent
MIN_BLINK_DURATION_BASELINE: float = MIN_BLINK_DURATION_MS  # ms; shortest accepted blink
MIN_BLINK_FREQUENCY_BASELINE: float = 1.0                  # blinks per minute
MIN_MAR_BASELINE: float = 0.2               # outer-lip MAR; ~0.4-0.6 is typical

# Blink frequency is expressed per this many seconds regardless of duration.
BLINK_FREQUENCY_WINDOW_S: float = 60.0


class CalibrationManager:
    """
    Collect alert-state eye metrics for a fixed period and derive baselines.

    Typical usage::

        calib = CalibrationManager()
        calib.start()
        while calib.is_calibrating:
            ...  # run landmark → EAR → blink → PERCLOS for one frame
            status = calib.update(ear, blink_event_ms, blink_freq, perclos, mar=mar)
            show_progress(status["progress"])
        baselines = calib.compute_baselines()

    Attributes:
        duration: Calibration length in seconds.
        fps: Expected frame rate (used only to report expected sample count).
        is_calibrating: ``True`` between ``start()`` and completion.
        start_time: ``time.time()`` at which ``start()`` was called.
        ear_values: EAR sample per frame.
        blink_durations: Duration (ms) of each blink completed during calibration.
            Its length is the blink count behind ``blink_frequency_baseline``.
        blink_frequencies: Rolling blink-frequency sample per frame
            (diagnostics only; not used for any baseline).
        perclos_values: PERCLOS sample per frame.
        mar_values: MAR sample per frame (only if the caller supplies it).
    """

    def __init__(
        self,
        duration: int = config.CALIBRATION_DURATION,
        fps: int = config.CAMERA_FPS,
    ) -> None:
        """
        Create an idle manager. Call ``start()`` to begin collecting.

        Args:
            duration: How many seconds to observe the driver.
            fps: Expected frame rate; ``duration * fps`` is the nominal number
                of samples that will be collected.
        """
        self.duration: int = duration
        self.fps: int = fps
        self.expected_samples: int = duration * fps

        self.is_calibrating: bool = False
        self.start_time: float = 0.0
        self._completed: bool = False

        self.ear_values: List[float] = []
        self.blink_durations: List[float] = []
        self.blink_frequencies: List[float] = []
        self.perclos_values: List[float] = []
        self.mar_values: List[float] = []

    def start(self, timestamp: Optional[float] = None) -> None:
        """
        Reset all buffers and begin a new calibration period.

        Args:
            timestamp: Start time in seconds. Defaults to ``time.time()``.
        """
        self.ear_values.clear()
        self.blink_durations.clear()
        self.blink_frequencies.clear()
        self.perclos_values.clear()
        self.mar_values.clear()

        self.start_time = time.time() if timestamp is None else timestamp
        self.is_calibrating = True
        self._completed = False

    def update(
        self,
        ear: float,
        blink_duration_ms: Optional[float],
        blink_frequency: float,
        perclos: float,
        timestamp: Optional[float] = None,
        mar: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Record one frame's metrics and report calibration progress.

        Args:
            ear: Raw EAR for this frame.
            blink_duration_ms: Duration of a blink that completed on this
                frame, or ``None`` if no blink completed (the usual case).
            blink_frequency: Current rolling blink frequency (kept for
                diagnostics; the baseline is derived from the blink count).
            perclos: Current PERCLOS percentage.
            timestamp: Frame time in seconds. Defaults to ``time.time()``.
            mar: Raw MAR for this frame, or ``None`` if the caller does not
                track the mouth (then no MAR baseline is produced).

        Returns:
            ``{"progress": float, "seconds_remaining": int, "is_complete": bool}``
            where ``progress`` runs from ``0.0`` to ``1.0``.

        Raises:
            RuntimeError: if called before ``start()``.
        """
        if not self.is_calibrating and not self._completed:
            raise RuntimeError("CalibrationManager.update() called before start()")

        now = time.time() if timestamp is None else timestamp

        # Keep collecting only while the window is open; frames that arrive
        # after completion are ignored so late samples don't skew baselines.
        if self.is_calibrating:
            self.ear_values.append(float(ear))
            self.blink_frequencies.append(float(blink_frequency))
            self.perclos_values.append(float(perclos))
            # Blinks are sparse events, so only append when one actually
            # completed this frame.
            if blink_duration_ms is not None:
                self.blink_durations.append(float(blink_duration_ms))
            if mar is not None:
                self.mar_values.append(float(mar))

        progress = self.get_progress(now)
        elapsed = now - self.start_time
        seconds_remaining = max(0, int(np.ceil(self.duration - elapsed)))

        if progress >= 1.0 and self.is_calibrating:
            self.is_calibrating = False
            self._completed = True

        return {
            "progress": progress,
            "seconds_remaining": seconds_remaining,
            "is_complete": self._completed,
        }

    def compute_baselines(self) -> Dict[str, float]:
        """
        Derive the driver's personal baselines from the collected samples.

        Returns:
            ``{"ear_baseline", "ear_threshold", "perclos_baseline",
            "blink_duration_baseline", "blink_frequency_baseline"}`` plus
            ``"mar_baseline"`` and ``"yawn_threshold"`` when MAR samples
            were collected. ``blink_frequency_baseline`` is in blinks per
            minute. Zero blinks is not an error, but it is logged as a
            warning and the affected baselines come out at their floors.

        Raises:
            RuntimeError: if calibration has not completed, or if no EAR
                samples were collected (e.g. the face was never detected).
        """
        if not self._completed:
            raise RuntimeError(
                "compute_baselines() called before calibration completed"
            )
        if not self.ear_values:
            raise RuntimeError(
                "No EAR samples collected during calibration — was a face visible?"
            )

        ear_baseline = float(np.mean(self.ear_values))
        ear_threshold = ear_baseline * EAR_THRESHOLD_RATIO

        n_blinks = len(self.blink_durations)
        if n_blinks == 0:
            logger.warning(
                "Calibration saw no blinks in %d s over %d frames - blink detection "
                "did not work (threshold / frame rate?); blink and PERCLOS baselines "
                "will be floored and this profile should not be trusted",
                self.duration, len(self.ear_values),
            )

        # Blinks per minute from the count, not from the detector's rolling
        # counter (see module docstring for why the latter is biased).
        raw_blink_frequency = (
            n_blinks / self.duration * BLINK_FREQUENCY_WINDOW_S if self.duration > 0 else 0.0
        )

        # Floors only guard against a zero divisor from a degenerate run.
        perclos_baseline = self._floored(
            "perclos_baseline",
            float(np.mean(self.perclos_values)) if self.perclos_values else 0.0,
            MIN_PERCLOS_BASELINE,
        )
        blink_duration_baseline = self._floored(
            "blink_duration_baseline",
            float(np.mean(self.blink_durations)) if self.blink_durations else 0.0,
            MIN_BLINK_DURATION_BASELINE,
        )
        blink_frequency_baseline = self._floored(
            "blink_frequency_baseline", raw_blink_frequency, MIN_BLINK_FREQUENCY_BASELINE
        )

        baselines = {
            "ear_baseline": ear_baseline,
            "ear_threshold": ear_threshold,
            "perclos_baseline": perclos_baseline,
            "blink_duration_baseline": blink_duration_baseline,
            "blink_frequency_baseline": blink_frequency_baseline,
        }

        if self.mar_values:
            # Median, not mean - see the module docstring.
            mar_baseline = self._floored(
                "mar_baseline", float(np.median(self.mar_values)), MIN_MAR_BASELINE
            )
            baselines["mar_baseline"] = mar_baseline
            baselines["yawn_threshold"] = mar_baseline * YAWN_MAR_RATIO

        return baselines

    @staticmethod
    def _floored(name: str, measured: float, floor: float) -> float:
        """
        Apply a lower bound to a baseline, warning when it engages.

        A floor engaging means the measurement was degenerate (see module
        docstring), so it must never be silent: the persisted profile would
        otherwise look like a real calibration.
        """
        if measured >= floor:
            return measured
        logger.warning(
            "Calibration %s floored: measured %.4g < %.4g - using the floor",
            name, measured, floor,
        )
        return floor

    def get_progress(self, timestamp: Optional[float] = None) -> float:
        """
        Fraction of the calibration period that has elapsed.

        Args:
            timestamp: Reference "now" in seconds. Defaults to ``time.time()``.

        Returns:
            ``0.0`` before ``start()``; otherwise ``elapsed / duration``
            clamped to ``1.0``.
        """
        if not self.is_calibrating and not self._completed:
            return 0.0
        if self.duration <= 0:
            return 1.0
        now = time.time() if timestamp is None else timestamp
        elapsed = now - self.start_time
        return float(min(max(elapsed / self.duration, 0.0), 1.0))

    def is_complete(self, timestamp: Optional[float] = None) -> bool:
        """
        Whether the calibration duration has been reached.

        Args:
            timestamp: Reference "now" in seconds. Defaults to ``time.time()``.

        Returns:
            ``True`` once ``duration`` seconds have elapsed since ``start()``.
        """
        if self._completed:
            return True
        return self.is_calibrating and self.get_progress(timestamp) >= 1.0
