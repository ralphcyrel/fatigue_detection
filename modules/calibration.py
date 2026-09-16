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
* ``perclos_baseline``         mean PERCLOS (floored at 5.0 %)
* ``blink_duration_baseline``  mean blink duration (floored at 150 ms)
* ``blink_frequency_baseline`` mean blink frequency (floored at 5 blinks/min)
* ``mar_baseline``             **median** closed-mouth MAR (floored at 0.2)
* ``yawn_threshold``           ``mar_baseline * 2.0`` → "candidate yawn" cut-off

The floors stop a very still, low-blink calibration from producing tiny
baselines that would inflate every normalised metric (and hence the FRS)
during normal driving.

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

import time
from typing import Dict, List, Optional

import numpy as np

from config import config
from modules.mar import YAWN_MAR_RATIO

# Fraction of the baseline EAR below which the eye is considered closed.
# 0.75 sits comfortably between typical open (≈0.30) and closed (≈0.10) EARs.
EAR_THRESHOLD_RATIO: float = 0.75

# Lower bounds applied to the computed baselines (see module docstring).
MIN_PERCLOS_BASELINE: float = 5.0          # percent
MIN_BLINK_DURATION_BASELINE: float = 150.0  # milliseconds
MIN_BLINK_FREQUENCY_BASELINE: float = 5.0   # blinks per window (minute)
MIN_MAR_BASELINE: float = 0.2               # outer-lip MAR; ~0.4-0.6 is typical


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
        blink_frequencies: Rolling blink-frequency sample per frame.
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
            blink_frequency: Current rolling blink frequency.
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
            were collected.

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

        # Each remaining baseline is floored so that an unusually quiet
        # calibration cannot make the normalised metrics explode later.
        perclos_baseline = max(
            float(np.mean(self.perclos_values)) if self.perclos_values else 0.0,
            MIN_PERCLOS_BASELINE,
        )
        blink_duration_baseline = max(
            float(np.mean(self.blink_durations)) if self.blink_durations else 0.0,
            MIN_BLINK_DURATION_BASELINE,
        )
        blink_frequency_baseline = max(
            float(np.mean(self.blink_frequencies)) if self.blink_frequencies else 0.0,
            MIN_BLINK_FREQUENCY_BASELINE,
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
            mar_baseline = max(float(np.median(self.mar_values)), MIN_MAR_BASELINE)
            baselines["mar_baseline"] = mar_baseline
            baselines["yawn_threshold"] = mar_baseline * YAWN_MAR_RATIO

        return baselines

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
