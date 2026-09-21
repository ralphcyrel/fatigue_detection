"""
Driver Fatigue Detection System - entry point.

Three phases, selected by the vehicle's ignition state (Module 11)::

    --enroll        Enrollment (operator-supervised, at hiring): face encoding +
                    60 s alert-state calibration -> Laravel backend.

    ignition OFF    Pre-drive assessment: recognise the driver, fetch their
                    thresholds, run a 30 s assessment. PASS -> starter relay
                    released. FAIL / not recognised / no baseline -> starter
                    stays inhibited and an operator override request is raised.
                    This is the ONLY phase in which the relay engages.

    ignition ON     Continuous monitoring: same metrics, LEDs / buzzer /
                    backend notification only. The relay is never touched -
                    AlertManager refuses to lock in this phase (Module 8).

Per-frame metrics are computed by one shared pipeline (Module 12) in both
phases::

    camera -> LandmarkExtractor -> MetricsPipeline (EAR / MAR / blink / PERCLOS
              / yawn / head pose / FRS) -> phase-specific decision -> AlertManager

Runs on the Raspberry Pi (Picamera2 + GPIO) and, with automatic fallbacks,
on a development machine (cv2.VideoCapture + mocked GPIO / ignition).
"""

import argparse
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

import cv2
import numpy as np

from config import config
from modules.alert import AlertManager
from modules.api import APIClient
from modules.assessment import PredriveAssessment
from modules.blink import BlinkDetector
from modules.calibration import EAR_THRESHOLD_RATIO, CalibrationManager
from modules.ear import EARCalculator
from modules.face_recognition_module import DriverRecognizer
from modules.head_pose import HeadPoseEstimator
from modules.ignition import ForcedIgnition, IgnitionSensor
from modules.landmark import LandmarkExtractor
from modules.mar import MARCalculator
from modules.perclos import PERCLOSCalculator
from modules.phase import LockReason, Phase
from modules.pipeline import HEAD_POSE_RELEASE_HOLD_S, FrameMetrics, HeadPoseDebounce, MetricsPipeline

logger = logging.getLogger("fatigue")

# ---------------------------------------------------------------------------
# Tunables local to the main loop
# ---------------------------------------------------------------------------

# Face recognition is ~10x more expensive than landmark extraction, so in the
# monitoring phase it is only re-run every N frames; the last result is
# cached in between.
IDENTIFY_INTERVAL: int = 30

# While the driver stays in DANGER (monitoring), log at most one event per
# this many seconds - otherwise a 30 fps loop would POST 30 events per second.
DANGER_EVENT_INTERVAL: float = 5.0

# Frames of face captured for the enrollment encoding.
ENROLL_FRAMES: int = 30

# Enrollment: wall-clock seconds of the driver's own EAR observed before
# calibration starts, to derive a personal closure threshold (see
# run_enrollment). Time-scoped so a slow loop settles for the same period
# as a fast one; SEED_MIN_SAMPLES guards a face that appears late.
SEED_WINDOW_S: float = 1.0
SEED_MIN_SAMPLES: int = 5

# Log the measured loop rate every N frames.
FPS_LOG_INTERVAL: int = 30

# Used in the MONITORING phase only, when the backend has no thresholds for
# a driver (or is offline, or the driver is unrecognised) so monitoring can
# still run - degraded monitoring is better than none. Pre-drive never uses
# these: a missing baseline there is a lock condition.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "ear_baseline": 0.30,
    "ear_threshold": 0.225,
    "perclos_baseline": 5.0,
    "blink_duration_baseline": 150.0,
    "blink_frequency_baseline": 5.0,
    # Outer-lip MAR of a resting closed mouth is ~0.4-0.6; the yawn threshold
    # is 2x that (modules/mar.py). Drivers enrolled before yawn detection
    # existed have neither key in the backend and get these generic values.
    "mar_baseline": 0.45,
    "yawn_threshold": 0.90,
}

# Threshold keys that only exist for drivers enrolled with yawn detection.
MAR_THRESHOLD_KEYS = ("mar_baseline", "yawn_threshold")

# Text colours (BGR) per alert level for the overlay.
LEVEL_BGR: Dict[str, tuple] = {
    "ALERT": (0, 200, 0),
    "WARNING": (0, 220, 255),
    "DANGER": (0, 0, 255),
}
WHITE = (255, 255, 255)
GREY = (160, 160, 160)

WINDOW_NAME = "Driver Fatigue Detection"

# Preview-window keys.
KEY_QUIT = ord("q")
KEY_IGNITION = ord("i")   # mock mode only: toggle ignition
KEY_RETRY = ord("r")      # pre-drive: re-run the assessment

# Human-readable lock reasons for the overlay.
LOCK_REASON_TEXT: Dict[LockReason, str] = {
    LockReason.FATIGUE_DETECTED: "FATIGUE DETECTED",
    LockReason.DRIVER_NOT_RECOGNIZED: "DRIVER NOT RECOGNIZED",
    LockReason.NO_BASELINE: "NO BASELINE ON FILE",
}

# ---------------------------------------------------------------------------
# Module-level handles so cleanup() can reach them from any exit path
# ---------------------------------------------------------------------------
_camera: Optional["Camera"] = None
_alert_manager: Optional[AlertManager] = None
_head_pose: Optional[HeadPoseEstimator] = None
_ignition: Any = None
_display_available: bool = True


class QuitRequested(Exception):
    """Raised when the user presses ``q`` in the preview window."""


class PhaseChanged(Exception):
    """Raised inside a phase loop when the ignition state no longer matches it."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Log INFO+ to the console and to a rotating ``logs/fatigue.log``."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # 5 MB x 3 backups keeps the SD card from filling up on a long-running Pi.
    file_handler = RotatingFileHandler(
        config.LOGS_DIR / "fatigue.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Camera abstraction
# ---------------------------------------------------------------------------

class Camera:
    """
    Uniform frame source over Picamera2 (Pi) or cv2.VideoCapture (elsewhere).

    Both back-ends deliver BGR ``uint8`` frames at ``config.CAMERA_WIDTH`` x
    ``config.CAMERA_HEIGHT`` so the rest of the pipeline is back-end agnostic.
    """

    def __init__(self) -> None:
        """Open Picamera2 if importable, otherwise fall back to OpenCV."""
        self.backend: str = "cv2"
        self._picam: Any = None
        self._cap: Optional[cv2.VideoCapture] = None

        try:
            from picamera2 import Picamera2  # type: ignore

            self._picam = Picamera2(config.CAMERA_INDEX)
            cfg = self._picam.create_video_configuration(
                main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"},
                controls={"FrameRate": config.CAMERA_FPS},
            )
            self._picam.configure(cfg)
            self._picam.start()
            self.backend = "picamera2"
            logger.info("Camera: Picamera2 %dx%d @ %d fps",
                        config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS)
            return
        except ImportError:
            logger.info("Picamera2 not available - falling back to cv2.VideoCapture")
        except Exception as exc:  # Picamera2 present but camera failed to open
            logger.warning("Picamera2 failed (%s) - falling back to cv2.VideoCapture", exc)
            self._picam = None

        self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {config.CAMERA_INDEX}")
        logger.info("Camera: cv2.VideoCapture(%d)", config.CAMERA_INDEX)

    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or ``None`` if capture failed."""
        if self._picam is not None:
            # Counter-intuitively, Picamera2's "RGB888" format is laid out in
            # memory as B,G,R - i.e. exactly what OpenCV expects - so no
            # channel swap is needed (see the Picamera2 manual, ch. 4.2).
            return self._picam.capture_array("main")
        ok, frame = self._cap.read() if self._cap is not None else (False, None)
        return frame if ok else None

    def release(self) -> None:
        """Stop the camera and free the device."""
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception as exc:
                logger.warning("Picamera2 release failed: %s", exc)
            self._picam = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def capture_frame() -> Optional[np.ndarray]:
    """Grab one frame from the global camera (Picamera2 or cv2)."""
    if _camera is None:
        return None
    return _camera.read()


def next_frame() -> np.ndarray:
    """Block until a frame is captured (retrying on transient failures)."""
    while True:
        frame = capture_frame()
        if frame is not None:
            return frame
        logger.warning("Frame capture failed - retrying")
        time.sleep(0.05)


class LoopRate:
    """Measured loop rate (frames / wall-clock seconds), logged every N frames."""

    def __init__(self, interval: int = FPS_LOG_INTERVAL) -> None:
        self.interval = interval
        self.fps: Optional[float] = None
        self._count = 0
        self._window_start = time.monotonic()

    def tick(self) -> Optional[float]:
        """Count one loop iteration; returns the last measured fps (or ``None``)."""
        self._count += 1
        if self._count % self.interval == 0:
            now = time.monotonic()
            self.fps = self.interval / max(now - self._window_start, 1e-6)
            self._window_start = now
            logger.info("Loop rate: %.1f fps (%d frames in %.2fs)",
                        self.fps, self.interval, self.interval / self.fps)
        return self.fps


# ---------------------------------------------------------------------------
# Display helpers (all tolerate a headless Pi with no monitor)
# ---------------------------------------------------------------------------

def show_frame(frame: np.ndarray) -> int:
    """
    Show ``frame`` in the preview window if a display is available.

    On a headless Pi (or with ``opencv-python-headless`` installed)
    ``cv2.imshow`` raises; the first failure disables the preview for the
    rest of the run so the loop keeps going, and says so loudly - once - in
    the log and on stdout.

    Returns:
        The key code pressed (``cv2.waitKey`` & 0xFF), or ``-1`` if none /
        headless.
    """
    global _display_available
    if not _display_available:
        return -1
    try:
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1)
        return key & 0xFF if key >= 0 else -1
    except cv2.error as exc:
        _display_available = False
        detail = exc.err if getattr(exc, "err", None) else str(exc).strip().splitlines()[-1]
        logger.warning(
            "PREVIEW WINDOW DISABLED for the rest of this run - cv2.imshow failed: %s. "
            "Detection continues headless. If you expected a window: 'not implemented' "
            "means the installed OpenCV has no GUI backend (opencv-python-headless) - "
            "run `pip uninstall -y opencv-python-headless && pip install opencv-python`; "
            "otherwise check DISPLAY / that you are on the Pi's desktop, not SSH.",
            detail,
        )
        print(
            "\n" + "=" * 72 + "\n"
            "  !! NO PREVIEW WINDOW - running headless for the rest of this run !!\n"
            f"  cv2.imshow failed: {detail}\n"
            "  Fix (GUI-less OpenCV):  pip uninstall -y opencv-python-headless && pip install opencv-python\n"
            "  Fix (no display):       run from the Pi desktop terminal, or set DISPLAY=:0\n"
            + "=" * 72 + "\n",
            flush=True,
        )
        return -1


def present(frame: np.ndarray) -> int:
    """
    Show ``frame`` and act on the global keys.

    ``q`` raises :class:`QuitRequested`; ``i`` toggles the mock ignition.
    Other keys are returned to the caller (e.g. ``r`` to retry).
    """
    key = show_frame(frame)
    if key == KEY_QUIT:
        raise QuitRequested()
    if key == KEY_IGNITION and _ignition is not None:
        if getattr(_ignition, "mock", False):
            new_state = not _ignition.read()
            logger.info("Preview key 'i': mock ignition -> %s", "ON" if new_state else "OFF")
            _ignition.set_mock_state(new_state)
        else:
            logger.info("Preview key 'i' ignored: ignition is read from hardware")
    return key


def draw_banner(frame: np.ndarray, text: str, color: tuple = WHITE) -> None:
    """Bottom strip with the phase / status text, drawn in place."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    driver: Optional[Dict[str, Any]],
    m: FrameMetrics,
    fps: Optional[float] = None,
) -> None:
    """
    Annotate ``frame`` in place with driver name, FRS, level, EAR, PERCLOS,
    MAR / yawn state, relay state, measured loop rate and head pose.

    Args:
        frame: BGR frame to draw on.
        driver: Dict from ``DriverRecognizer.identify()`` (may be ``None``).
        m: Pipeline output for this frame.
        fps: Last measured loop rate, or ``None`` before the first sample.
    """
    level = m.level
    color = LEVEL_BGR.get(level, WHITE)
    name = driver["name"] if driver else "Unknown"
    font = cv2.FONT_HERSHEY_SIMPLEX
    w = frame.shape[1]

    # Dark strip behind the text keeps it legible over a bright face.
    cv2.rectangle(frame, (0, 0), (w, 135), (0, 0, 0), -1)

    cv2.putText(frame, f"Driver: {name}", (10, 25), font, 0.6, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}" if fps is not None else "FPS: --",
                (w - 150, 25), font, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FRS: {m.frs:.3f}", (10, 50), font, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(frame, level, (w - 150, 50), font, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"EAR: {m.ear:.3f}", (10, 75), font, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"PERCLOS: {m.perclos:.1f}%", (10, 100), font, 0.55, WHITE, 1, cv2.LINE_AA)

    # Mouth column next to the eye metrics: raw MAR, then the yawn state
    # machine so a talking mouth ("open 0.4s", resets) can be told apart
    # from a confirmed yawn ("YAWNING", red) at a glance.
    cv2.putText(frame, f"MAR: {m.mar:.3f}", (200, 75), font, 0.55, WHITE, 1, cv2.LINE_AA)
    yawn = m.yawn_status
    state = str(yawn.get("state", "closed"))
    if state == "yawning":
        yawn_text, yawn_color = f"Yawn: YAWNING {yawn['open_s']:.1f}s", (0, 0, 255)
    elif state == "open":
        yawn_text, yawn_color = f"Yawn: open {yawn['open_s']:.1f}s", (0, 220, 255)
    else:
        yawn_text, yawn_color = "Yawn: --", (0, 200, 0)
    cv2.putText(frame, f"{yawn_text}  (n={yawn.get('count', 0)})", (200, 100), font, 0.55,
                yawn_color, 1, cv2.LINE_AA)

    relay = _alert_manager.get_relay_state() if _alert_manager else "n/a"
    cv2.putText(frame, f"Relay: {relay}", (w - 150, 100), font, 0.5,
                (0, 0, 255) if relay == "LOCKED" else (0, 200, 0), 1, cv2.LINE_AA)

    # Head pose row: angles on the left, status on the right. "ALERT" from
    # get_status_text() means a normal head, which reads better as "OK" here.
    pose = m.pose
    if pose is not None:
        angles = f"Pitch: {pose['pitch']:+.1f}  Yaw: {pose['yaw']:+.1f}  Roll: {pose['roll']:+.1f}"
        status = _head_pose.get_status_text(pose) if _head_pose else "n/a"
        if status == "ALERT":
            status = "OK"
        status_color = (0, 0, 255) if pose["alert"] else (0, 200, 0)
    else:
        angles, status, status_color = "Pitch: --  Yaw: --  Roll: --", "N/A", GREY
    cv2.putText(frame, angles, (10, 125), font, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Head: {status}", (w - 200, 125), font, 0.55,
                status_color, 2, cv2.LINE_AA)


def draw_pose_debug(
    frame: np.ndarray,
    debounce: HeadPoseDebounce,
    now: float,
    fps: Optional[float] = None,
) -> None:
    """
    ``--debug-pose`` strip under the main overlay: how long the pose has been
    alerting vs. the DANGER threshold, whether the override is active, and
    (while active) how long the pose has been normal vs. the release hold.

    Args:
        frame: BGR frame to draw on.
        debounce: The pipeline's :class:`HeadPoseDebounce`.
        now: ``time.monotonic()`` for this frame.
        fps: Last measured loop rate (shown so the rate is visible on the
            NO FACE / UNKNOWN DRIVER frames, which have no main overlay).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    y0 = 135
    cv2.rectangle(frame, (0, y0), (frame.shape[1], y0 + 48), (30, 30, 30), -1)

    alert_s = debounce.alert_elapsed(now)
    normal_s = debounce.normal_elapsed(now)
    if debounce.active:
        state, color = "OVERRIDE: DANGER", (0, 0, 255)
    elif alert_s > 0:
        state, color = "OVERRIDE: arming", (0, 220, 255)
    else:
        state, color = "OVERRIDE: off", (0, 200, 0)

    # Progress bar for whichever timer is currently counting.
    if debounce.active:
        frac = min(normal_s / debounce.release_hold, 1.0) if debounce.release_hold > 0 else 1.0
        label = f"normal {normal_s:.2f}s / release at {debounce.release_hold:.2f}s"
        bar_color = (0, 200, 0)
    else:
        frac = min(alert_s / debounce.danger_hold, 1.0)
        label = f"alerting {alert_s:.2f}s / DANGER at {debounce.danger_hold:.2f}s"
        bar_color = (0, 0, 255)

    cv2.putText(frame, f"POSE DEBUG  {state}", (10, y0 + 18), font, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}" if fps is not None else "FPS: --",
                (frame.shape[1] - 150, y0 + 18), font, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (10, y0 + 40), font, 0.45, WHITE, 1, cv2.LINE_AA)
    bx0, bx1, by = frame.shape[1] - 150, frame.shape[1] - 10, y0 + 30
    cv2.rectangle(frame, (bx0, by), (bx1, by + 12), (90, 90, 90), 1)
    if frac > 0:
        cv2.rectangle(frame, (bx0, by), (bx0 + int((bx1 - bx0) * frac), by + 12), bar_color, -1)


def display_text(frame: np.ndarray, text: str, color: tuple = (0, 0, 255), dy: int = 0) -> None:
    """
    Draw ``text`` centred on ``frame`` in place (e.g. ``"UNKNOWN DRIVER"``).

    Args:
        frame: BGR frame to draw on.
        text: Message to show.
        color: BGR colour.
        dy: Vertical offset from centre (for a second line).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 1.0, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max((frame.shape[1] - tw) // 2, 0)
    y = (frame.shape[0] + th) // 2 + dy
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def cleanup() -> None:
    """Release the camera, reset GPIO and close any OpenCV windows."""
    global _camera
    logger.info("Shutting down...")
    if _camera is not None:
        _camera.release()
        _camera = None
    if _alert_manager is not None:
        _alert_manager.cleanup()
    if _ignition is not None:
        _ignition.cleanup()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # headless
    logger.info("Cleanup complete")


# ---------------------------------------------------------------------------
# Enrollment mode
# ---------------------------------------------------------------------------

def run_enrollment(api_client: APIClient, extractor: LandmarkExtractor) -> int:
    """
    Enrol a new driver: face encoding + 60 s alert-state calibration.

    Steps:
    1. Ask for the driver's DB id on the terminal.
    2. Capture ``ENROLL_FRAMES`` frames containing a face and average their
       128-d encodings (averaging is more robust than a single frame).
    3. Settle for ``SEED_WINDOW_S`` to derive a personal closure threshold
       from the driver's own median EAR, then run ``CalibrationManager``
       for ``config.CALIBRATION_DURATION`` s, feeding it EAR / blink /
       PERCLOS / MAR from the live pipeline.
    4. Write the per-frame series to ``config.CALIBRATIONS_DIR`` and POST
       encoding + baselines to the backend.

    Args:
        api_client: Connected API client.
        extractor: Landmark extractor for the calibration phase.

    Returns:
        Process exit code (0 = success).
    """
    import face_recognition  # heavy import; only needed here

    raw = input("Enter driver_id to enrol: ").strip()
    try:
        driver_id = int(raw)
    except ValueError:
        logger.error("Invalid driver_id %r - must be an integer", raw)
        return 2

    # ---- 1. Face encoding ------------------------------------------------
    logger.info("Look at the camera. Capturing %d frames for the face encoding...", ENROLL_FRAMES)
    encodings = []
    attempts = 0
    while len(encodings) < ENROLL_FRAMES and attempts < ENROLL_FRAMES * 10:
        attempts += 1
        frame = capture_frame()
        if frame is None:
            continue
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        locations = face_recognition.face_locations(rgb, model="hog")
        if not locations:
            display_text(frame, "NO FACE DETECTED")
            present(frame)
            continue
        # Use only the largest face in case someone is visible in the background.
        largest = max(locations, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
        enc = face_recognition.face_encodings(rgb, [largest])
        if enc:
            encodings.append(enc[0])
        display_text(frame, f"ENCODING {len(encodings)}/{ENROLL_FRAMES}")
        present(frame)

    if len(encodings) < ENROLL_FRAMES // 2:
        logger.error("Only %d usable face frames captured - aborting enrollment", len(encodings))
        return 3
    face_encoding = np.mean(np.stack(encodings), axis=0)
    logger.info("Face encoding computed from %d frames", len(encodings))

    # ---- 2. Calibration ---------------------------------------------------
    ear_calc = EARCalculator()
    mar_calc = MARCalculator()
    blink_detector = BlinkDetector(frequency_window=60)
    perclos_calc = PERCLOSCalculator()
    calib = CalibrationManager()

    # No personal threshold exists yet. Rather than seeding one from
    # DEFAULT_THRESHOLDS (0.225 is 75 % of a 0.30 EAR but 88 % of a 0.255 one -
    # inside landmark noise, so it manufactures false closures), observe the
    # driver for SEED_WINDOW_S of wall-clock time *before* calibration starts
    # and derive the threshold from their own median EAR. Nothing is fed to
    # the blink detector, PERCLOS or the calibration until then, so the
    # calibration window is pure detection time under a personal threshold.
    ear_history: list = []
    seed_start = time.time()
    logger.info("Settling for %.0f s to derive a personal closure threshold...", SEED_WINDOW_S)
    while time.time() - seed_start < SEED_WINDOW_S or len(ear_history) < SEED_MIN_SAMPLES:
        frame = capture_frame()
        if frame is None:
            continue
        landmarks, _ = extractor.extract(frame)
        if landmarks is None:
            display_text(frame, "FACE LOST - please look at the camera")
            present(frame)
            continue
        ear_history.append(ear_calc.compute_average_ear(landmarks))
        display_text(frame, "SETTLING - look at the road")
        present(frame)
    threshold = float(np.median(ear_history)) * EAR_THRESHOLD_RATIO
    logger.info("Seed threshold %.4f from %d frames (median EAR %.4f)",
                threshold, len(ear_history), float(np.median(ear_history)))

    logger.info("Starting %d s calibration - stay alert, look at the road and keep "
                "your mouth relaxed (talking inflates the MAR baseline).",
                config.CALIBRATION_DURATION)
    calib.start()
    while calib.is_calibrating:
        frame = capture_frame()
        if frame is None:
            continue
        landmarks, _ = extractor.extract(frame)
        if landmarks is None:
            display_text(frame, "FACE LOST - please look at the camera")
            present(frame)
            continue

        ear = ear_calc.compute_average_ear(landmarks)
        mar = mar_calc.compute_mar(landmarks)
        # Keep tightening to the running median as more of the driver is seen.
        ear_history.append(ear)
        threshold = float(np.median(ear_history)) * EAR_THRESHOLD_RATIO

        now = time.time()
        eye_closed = ear < threshold  # the test both detectors below apply
        event = blink_detector.update(ear, threshold, now)
        perclos = perclos_calc.update(ear, threshold)
        status = calib.update(
            ear,
            event["duration_ms"] if event else None,
            blink_detector.get_blink_frequency(now),
            perclos,
            timestamp=now,
            mar=mar,
            eye_closed=eye_closed,
            threshold=threshold,
        )

        display_text(frame, f"CALIBRATING {status['progress'] * 100:.0f}%  "
                            f"({status['seconds_remaining']}s left)")
        present(frame)

    baselines = calib.compute_baselines()
    logger.info("Calibration baselines: %s", baselines)
    # Pre-floor measurements and counts, so a baseline that landed on a
    # MIN_* floor (modules/calibration.py) can be traced to its cause.
    logger.info("Calibration raw: %s", calib.raw_summary())
    # Per-frame series + summary on disk, written before the backend call so
    # a rejected enrollment still leaves the evidence behind.
    try:
        calib.write_files(config.DEVICE_ID, driver_id, baselines)
    except OSError as exc:
        logger.error("Could not write calibration data: %s", exc)

    # ---- 3. Persist ---------------------------------------------------------
    ok = api_client.save_driver_enrollment(driver_id, face_encoding.tolist(), baselines)
    if not ok:
        logger.error("Backend rejected enrollment for driver %s", driver_id)
        return 5

    msg = f"Enrollment complete for driver {driver_id}"
    logger.info(msg)
    print(msg)
    return 0


# ---------------------------------------------------------------------------
# Shared helpers for the two operating phases
# ---------------------------------------------------------------------------

def check_phase(ignition: Any, expected: Phase) -> None:
    """Raise :class:`PhaseChanged` if the ignition no longer implies ``expected``."""
    if ignition.phase() is not expected:
        raise PhaseChanged()


def load_thresholds(
    api_client: APIClient, driver_id: int, allow_defaults: bool
) -> Optional[Dict[str, float]]:
    """
    Fetch the calibrated baselines for this device and fill any omitted field.

    The backend keys calibration by device (``config.DEVICE_ID``) and
    resolves the assigned driver itself; ``driver_id`` is the driver the
    camera recognised and is only used for logging / a mismatch warning.

    Args:
        api_client: Backend client.
        driver_id: Recognised driver.
        allow_defaults: ``True`` (monitoring) substitutes
            :data:`DEFAULT_THRESHOLDS` when no usable record exists;
            ``False`` (pre-drive) returns ``None`` instead, because a
            missing baseline is a lock condition there.

    Returns:
        A complete thresholds dict, or ``None`` (pre-drive only).
    """
    thresholds = api_client.get_device_calibration(
        config.DEVICE_ID, expected_driver_id=driver_id
    )
    if not thresholds or "ear_threshold" not in thresholds:
        if not allow_defaults:
            logger.warning("No baseline on file for driver %s (backend returned %r)",
                           driver_id, thresholds)
            return None
        logger.warning("No thresholds for driver %s - using defaults; "
                       "run `main.py --enroll` for this driver", driver_id)
        return dict(DEFAULT_THRESHOLDS)

    # Drivers enrolled before yawn detection have no MAR baseline; say so
    # explicitly because the generic default silently makes yawn detection
    # non-personal.
    missing_mar = [k for k in MAR_THRESHOLD_KEYS if k not in thresholds]
    if missing_mar:
        logger.warning("Driver %s has no MAR baseline (%s missing) - yawn detection will "
                       "use generic defaults; re-enrol with `main.py --enroll` to calibrate it",
                       driver_id, ", ".join(missing_mar))
    # Fill any field the backend omitted so the pipeline never KeyErrors on
    # a partial record.
    for key, val in DEFAULT_THRESHOLDS.items():
        thresholds.setdefault(key, val)
    return thresholds


def identify_driver_bounded(
    recognizer: DriverRecognizer,
    ignition: Any,
    rate: LoopRate,
    timeout_s: float = config.PREDRIVE_RECOGNITION_TIMEOUT_S,
    matches_required: int = config.PREDRIVE_RECOGNITION_MATCHES,
) -> Optional[Dict[str, Any]]:
    """
    Pre-drive recognition: identify on every frame for up to ``timeout_s``
    and accept once the same ``driver_id`` has matched ``matches_required``
    times (rejects a single false match; gives the camera time to settle).

    Returns:
        The accepted match, or ``None`` on timeout.

    Raises:
        PhaseChanged: if the ignition turns ON meanwhile.
    """
    deadline = time.monotonic() + timeout_s
    counts: Dict[int, int] = {}
    best: Optional[Dict[str, Any]] = None
    logger.info("Pre-drive: identifying driver (up to %.0fs, %d consistent matches required)",
                timeout_s, matches_required)
    while time.monotonic() < deadline:
        check_phase(ignition, Phase.PREDRIVE)
        frame = next_frame()
        rate.tick()
        match = recognizer.identify(frame)
        if match is not None:
            did = int(match["driver_id"])
            counts[did] = counts.get(did, 0) + 1
            best = match
            if counts[did] >= matches_required:
                logger.info("Driver identified: %s (id=%s, confidence %.2f, %d matches)",
                            match["name"], did, match["confidence"], counts[did])
                return match
        seen = counts.get(int(best["driver_id"]), 0) if best else 0
        display_text(frame, f"IDENTIFYING {seen}/{matches_required}", WHITE)
        display_text(frame, f"{max(0.0, deadline - time.monotonic()):.0f}s left", GREY, dy=40)
        draw_banner(frame, "PRE-DRIVE  |  starter LOCKED  |  identifying driver", (0, 220, 255))
        present(frame)
    logger.warning("Pre-drive: driver not recognised within %.0fs (matches seen: %s)",
                   timeout_s, counts or "none")
    return None


def wait_for_ignition(ignition: Any, rate: LoopRate, message: str, color: tuple) -> bool:
    """
    Show ``message`` until the ignition turns ON (return ``True``) or the
    user presses ``r`` to re-run the assessment (return ``False``).
    """
    while True:
        if ignition.phase() is Phase.MONITORING:
            return True
        frame = next_frame()
        rate.tick()
        display_text(frame, message, color)
        display_text(frame, "waiting for ignition ON  (r = re-assess)", GREY, dy=40)
        relay = _alert_manager.get_relay_state() if _alert_manager else "n/a"
        draw_banner(frame, f"PRE-DRIVE  |  starter {relay}", color)
        if present(frame) == KEY_RETRY:
            logger.info("Pre-drive: re-assessment requested from the preview window")
            return False


def await_override(
    api_client: APIClient,
    alert_manager: AlertManager,
    ignition: Any,
    rate: LoopRate,
    driver_id: Optional[int],
    reason: LockReason,
    assessment_id: Optional[int],
) -> None:
    """
    Starter stays inhibited: raise one override request and poll the
    operator's decision until approved (-> release starter, wait for
    ignition), the user presses ``r`` (-> re-assess) or the phase changes.

    All three :class:`LockReason` values resolve through this one path.
    """
    reason_text = LOCK_REASON_TEXT[reason]
    logger.warning("Pre-drive: starter stays LOCKED - %s (driver=%s)", reason.value, driver_id)
    alert_manager.lock_relay()  # already locked; explicit for clarity + log

    request_id = api_client.request_override(config.DEVICE_ID, driver_id, reason, assessment_id)
    status = "pending"
    last_poll = time.monotonic()
    offline_logged = request_id is None
    if offline_logged:
        logger.error("Override request could not be raised (backend offline?) - will retry "
                     "every %.0fs; starter stays locked", config.OVERRIDE_POLL_SECONDS)

    while True:
        check_phase(ignition, Phase.PREDRIVE)
        now = time.monotonic()
        if now - last_poll >= config.OVERRIDE_POLL_SECONDS:
            last_poll = now
            if request_id is None:
                request_id = api_client.request_override(
                    config.DEVICE_ID, driver_id, reason, assessment_id)
            elif status == "pending":
                polled = api_client.check_override_request(request_id)
                if polled is not None and polled != status:
                    status = polled
                    logger.warning("Override request %s -> %s", request_id, status)

        if status == "approved":
            logger.warning("Operator override APPROVED (request %s) - releasing starter", request_id)
            alert_manager.unlock_relay()
            wait_for_ignition(ignition, rate, "OVERRIDE APPROVED - start vehicle", (0, 200, 0))
            return

        frame = next_frame()
        rate.tick()
        display_text(frame, f"STARTER LOCKED: {reason_text}")
        if request_id is None:
            second, color = "backend unreachable - retrying override request", (0, 220, 255)
        elif status == "denied":
            second, color = f"override DENIED (request {request_id})  -  r = re-assess", (0, 0, 255)
        else:
            second, color = f"awaiting operator override (request {request_id})", (0, 220, 255)
        display_text(frame, second, color, dy=40)
        draw_banner(frame, "PRE-DRIVE  |  starter LOCKED", (0, 0, 255))
        if present(frame) == KEY_RETRY:
            logger.info("Pre-drive: re-assessment requested from the preview window")
            return


# ---------------------------------------------------------------------------
# Phase 1: pre-drive assessment (ignition OFF)
# ---------------------------------------------------------------------------

def run_predrive_assessment(
    api_client: APIClient,
    extractor: LandmarkExtractor,
    recognizer: DriverRecognizer,
    alert_manager: AlertManager,
    head_pose: HeadPoseEstimator,
    ignition: Any,
    rate: LoopRate,
    debug_pose: bool = False,
    pose_release_hold: float = HEAD_POSE_RELEASE_HOLD_S,
) -> None:
    """
    One pre-drive cycle. Returns when the ignition turns ON (after a pass or
    an approved override), or when the user asks to re-run (``r``).

    1. Relay inhibited (``set_phase(PREDRIVE)``).
    2. Recognise the driver (bounded)          -> else DRIVER_NOT_RECOGNIZED.
    3. Fetch thresholds, no defaults           -> else NO_BASELINE.
    4. 30 s assessment through the shared pipeline; every frame recorded.
    5. Persist CSV/JSON + POST /assessments.
    6. PASS -> release starter, wait for ignition.
       FAIL -> FATIGUE_DETECTED.
    7. Any lock reason -> :func:`await_override`.
    """
    alert_manager.set_phase(Phase.PREDRIVE)
    # Every pre-drive cycle starts inhibited, including a re-run requested
    # with 'r' after a pass (set_phase is a no-op when already in PREDRIVE).
    alert_manager.lock_relay()
    alert_manager.set_alert_level("ALERT")
    logger.info("=== PRE-DRIVE phase (ignition OFF) - starter %s ===",
                alert_manager.get_relay_state())

    driver: Optional[Dict[str, Any]] = None
    driver_id: Optional[int] = None
    lock_reason: Optional[LockReason] = None
    assessment_id: Optional[int] = None

    try:
        # 2. Recognise
        driver = identify_driver_bounded(recognizer, ignition, rate)
        if driver is None:
            lock_reason = LockReason.DRIVER_NOT_RECOGNIZED
        else:
            driver_id = int(driver["driver_id"])
            # 3. Thresholds - no defaults at pre-drive
            thresholds = load_thresholds(api_client, driver_id, allow_defaults=False)
            if thresholds is None:
                lock_reason = LockReason.NO_BASELINE

        if lock_reason is None:
            # 4. Assessment
            pipeline = MetricsPipeline(
                head_pose,
                blink_window_s=config.PREDRIVE_ASSESSMENT_SECONDS,
                pose_release_hold=pose_release_hold,
            )
            assessment = PredriveAssessment()
            assessment.start(time.time())
            logger.info("Pre-drive: %d s assessment started for driver %s (%s)",
                        config.PREDRIVE_ASSESSMENT_SECONDS, driver_id, driver["name"])

            while not assessment.is_complete(time.time()):
                check_phase(ignition, Phase.PREDRIVE)
                frame = next_frame()
                fps = rate.tick()
                now, now_mono = time.time(), time.monotonic()
                remaining = assessment.seconds_remaining(now)
                banner = (f"PRE-DRIVE  |  assessing {remaining:4.1f}s left  |  "
                          f"starter {alert_manager.get_relay_state()}")

                landmarks, _rect = extractor.extract(frame)
                if landmarks is None:
                    pipeline.note_no_face()
                    assessment.add_no_face(now)
                    alert_manager.set_alert_level("ALERT")
                    display_text(frame, "NO FACE")
                    draw_banner(frame, banner, (0, 220, 255))
                    if debug_pose:
                        draw_pose_debug(frame, pipeline.pose_debounce, now_mono, fps)
                    present(frame)
                    continue

                m = pipeline.process(landmarks, thresholds, now, now_mono)
                assessment.add(m, now)
                # LEDs / buzzer give the driver feedback; the relay is decided
                # once, below, on the aggregate - never per frame.
                alert_manager.set_alert_level(m.level)
                draw_overlay(frame, driver, m, fps)
                draw_banner(frame, banner, (0, 220, 255))
                if debug_pose:
                    draw_pose_debug(frame, pipeline.pose_debounce, now_mono, fps)
                present(frame)

            # 5. Persist
            result = assessment.result()
            logger.info("Pre-drive result: passed=%s void=%s worst_%.0fs_mean=%.3f mean=%.3f "
                        "median=%.3f max=%.3f n=%d no_face=%d pose_override_frames=%d yawns=%d",
                        result["passed"], result["void"], result["worst_window_s"],
                        result["worst_window_mean"], result["mean_frs"], result["median_frs"],
                        result["max_frs"], result["n_samples"], result["n_no_face"],
                        result["pose_override_frames"], result["yawns"])
            try:
                assessment.write_files(config.DEVICE_ID, driver_id)
            except OSError as exc:
                logger.error("Could not write assessment files: %s", exc)

            if result["void"]:
                # Not enough face to judge - treat like an unrecognised driver.
                logger.warning("Pre-drive: assessment VOID (face missing %.0f%% of frames)",
                               result["no_face_fraction"] * 100)
                lock_reason = LockReason.DRIVER_NOT_RECOGNIZED
            elif not result["passed"]:
                lock_reason = LockReason.FATIGUE_DETECTED

            assessment_id = api_client.post_assessment(
                config.DEVICE_ID, driver_id, result, lock_reason)

            # 6. Verdict
            if lock_reason is None:
                logger.info("Pre-drive PASSED (worst %.0fs mean %.3f < %.2f) - releasing starter",
                            result["worst_window_s"], result["worst_window_mean"],
                            result["pass_threshold"])
                alert_manager.unlock_relay()
                alert_manager.set_alert_level("ALERT")
                wait_for_ignition(ignition, rate, "ASSESSMENT PASSED - start vehicle", (0, 200, 0))
                return

        # 7. Lock path (all three reasons)
        alert_manager.set_alert_level("ALERT")
        await_override(api_client, alert_manager, ignition, rate, driver_id,
                       lock_reason, assessment_id)

    except PhaseChanged:
        logger.info("Pre-drive: ignition turned ON - leaving pre-drive (starter %s)",
                    alert_manager.get_relay_state())


# ---------------------------------------------------------------------------
# Phase 2: continuous monitoring (ignition ON)
# ---------------------------------------------------------------------------

def run_monitoring(
    api_client: APIClient,
    extractor: LandmarkExtractor,
    recognizer: DriverRecognizer,
    alert_manager: AlertManager,
    head_pose: HeadPoseEstimator,
    ignition: Any,
    rate: LoopRate,
    debug_pose: bool = False,
    pose_release_hold: float = HEAD_POSE_RELEASE_HOLD_S,
) -> None:
    """
    Monitor continuously while the ignition is ON. Returns when it turns OFF.

    Alerts (LEDs, buzzer) and backend DANGER notifications only. The relay is
    never driven here; ``AlertManager`` refuses ``lock_relay()`` in this
    phase, so nothing in this loop can inhibit the starter even by mistake.
    An unrecognised driver, or one without a baseline, is monitored with
    :data:`DEFAULT_THRESHOLDS` - degraded monitoring is better than none.
    """
    alert_manager.set_phase(Phase.MONITORING)
    alert_manager.set_alert_level("ALERT")
    logger.info("=== MONITORING phase (ignition ON) - starter %s, lock refused ===",
                alert_manager.get_relay_state())

    pipeline = MetricsPipeline(head_pose, pose_release_hold=pose_release_hold)
    driver: Optional[Dict[str, Any]] = None
    current_driver_id: Optional[int] = None
    thresholds: Dict[str, float] = dict(DEFAULT_THRESHOLDS)
    defaults_logged = False
    last_danger_push = 0.0
    frame_count = 0

    while ignition.phase() is Phase.MONITORING:
        frame_count += 1
        frame = next_frame()
        fps = rate.tick()
        now, now_mono = time.time(), time.monotonic()
        if alert_manager.relay_locked:
            # Key turned ON before an assessment passed: the fuse tap is live
            # but the starter is still inhibited. Nothing here can release it.
            banner, banner_color = ("MONITORING  |  starter LOCKED - turn ignition OFF "
                                    "for pre-drive assessment"), (0, 0, 255)
        else:
            banner, banner_color = "MONITORING  |  relay never engages", (0, 200, 0)

        # Landmarks - no face is treated as the safe state
        landmarks, _rect = extractor.extract(frame)
        if landmarks is None:
            pipeline.note_no_face()
            alert_manager.set_alert_level("ALERT")
            display_text(frame, "NO FACE")
            draw_banner(frame, banner, banner_color)
            if debug_pose:
                draw_pose_debug(frame, pipeline.pose_debounce, now_mono, fps)
            present(frame)
            continue

        # Identify - every IDENTIFY_INTERVAL frames, or every frame until
        # someone has been recognised at all.
        if current_driver_id is None or frame_count % IDENTIFY_INTERVAL == 0:
            match = recognizer.identify(frame)
            if match is None:
                if current_driver_id is None and not defaults_logged:
                    logger.warning("Monitoring: driver not recognised - monitoring with "
                                   "DEFAULT_THRESHOLDS until a face is recognised")
                    defaults_logged = True
                # else: a known driver's periodic re-identify failed; keep
                # the cached driver and thresholds.
            elif int(match["driver_id"]) != current_driver_id:
                driver = match
                current_driver_id = int(match["driver_id"])
                logger.info("Driver identified: %s (id=%s, confidence %.2f)",
                            match["name"], current_driver_id, match["confidence"])
                thresholds = load_thresholds(api_client, current_driver_id, allow_defaults=True)
                # Fresh per-driver history so the previous driver's blinks
                # and yawns don't leak into this driver's metrics.
                pipeline.reset()
                last_danger_push = 0.0
            else:
                driver = match

        m = pipeline.process(landmarks, thresholds, now, now_mono)

        # Physical alerts - LEDs and buzzer only. m.level is DANGER while the
        # head-pose override is active regardless of the (lower) FRS level.
        alert_manager.set_alert_level(m.level)

        # Operator notification: DANGER to the backend, rate-limited.
        if m.level == "DANGER" and now - last_danger_push >= DANGER_EVENT_INTERVAL:
            api_client.push_fatigue_event(
                current_driver_id, m.effective_result(), m.ear, m.perclos,
                relay_triggered=False, phase=Phase.MONITORING,
            )
            last_danger_push = now

        draw_overlay(frame, driver, m, fps)
        draw_banner(frame, banner, banner_color)
        if debug_pose:
            draw_pose_debug(frame, pipeline.pose_debounce, now_mono, fps)
        present(frame)

    logger.info("Monitoring: ignition turned OFF - returning to pre-drive")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Driver fatigue detection system")
    parser.add_argument(
        "--enroll", action="store_true",
        help="enrol a new driver (face encoding + 60 s calibration) and exit",
    )
    parser.add_argument(
        "--mock-gpio", action="store_true",
        help="force AlertManager and IgnitionSensor into mock mode even on a Pi "
             "(press 'i' in the preview window to toggle the mock ignition)",
    )
    parser.add_argument(
        "--force-phase", choices=[p.value for p in Phase], default=None,
        help="ignore the ignition input and stay in this phase (testing without "
             "ignition hardware); in predrive, 'r' re-runs the assessment",
    )
    parser.add_argument(
        "--debug-pose", action="store_true",
        help="overlay the head-pose debounce timers / override state on the preview",
    )
    parser.add_argument(
        "--pose-release-hold", type=float, default=HEAD_POSE_RELEASE_HOLD_S, metavar="SECONDS",
        help="seconds the head pose must be normal before a head-pose DANGER "
             f"is released (default {HEAD_POSE_RELEASE_HOLD_S})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    """
    Start-up sequence, then enrollment or the phase supervisor loop.

    ``cleanup()`` is guaranteed to run via ``try/finally`` so the camera is
    always released and GPIO reset (the starter is left inhibited -
    fail-secure).

    Returns:
        Process exit code.
    """
    global _camera, _alert_manager, _head_pose, _ignition

    args = parse_args(argv)
    setup_logging()
    logger.info("=== Driver Fatigue Detection System starting ===")
    exit_code = 0

    try:
        # 1. Config is imported at module load; log the key values.
        logger.info("Config: device %s, camera %dx%d@%dfps, API %s, calibration %ds, "
                    "pre-drive %ds (worst %.0fs window)",
                    config.DEVICE_ID, config.CAMERA_WIDTH, config.CAMERA_HEIGHT,
                    config.CAMERA_FPS, config.API_BASE_URL, config.CALIBRATION_DURATION,
                    config.PREDRIVE_ASSESSMENT_SECONDS, config.PREDRIVE_WORST_WINDOW_SECONDS)

        # 2. Backend
        api_client = APIClient()
        if not api_client.ping():
            logger.warning("Laravel backend is offline - continuing; pre-drive will lock "
                           "(no baseline) and monitoring will use defaults")

        # 3. Landmarks
        if not config.LANDMARK_MODEL.exists():
            logger.error("Landmark model not found at %s (see README.md)", config.LANDMARK_MODEL)
            return 1
        extractor = LandmarkExtractor(str(config.LANDMARK_MODEL), config.SCALE_FACTOR)

        # 4. Driver recognition
        recognizer = DriverRecognizer(api_client)
        loaded = recognizer.load_encodings()
        if loaded == 0 and not args.enroll:
            logger.warning("No driver encodings loaded - every driver will be UNKNOWN. "
                           "Run `python main.py --enroll` first.")

        # 5. Ignition (selects the phase) - forced, mocked, or real GPIO
        if args.force_phase:
            _ignition = ForcedIgnition(Phase(args.force_phase))
        else:
            _ignition = IgnitionSensor(mock=args.mock_gpio)
        initial_phase = _ignition.phase()

        # 6. Alerts (auto-detects GPIO). Relay starts inhibited in pre-drive.
        _alert_manager = AlertManager(mock=args.mock_gpio, phase=initial_phase)

        # 6b. Head pose (camera intrinsics derived from the frame size)
        _head_pose = HeadPoseEstimator(
            frame_width=config.CAMERA_WIDTH, frame_height=config.CAMERA_HEIGHT
        )

        # 7. Camera
        _camera = Camera()

        if args.enroll:
            exit_code = run_enrollment(api_client, extractor)
        else:
            rate = LoopRate()
            logger.info("Supervisor started in %s - press 'q' in the window or Ctrl-C to stop",
                        initial_phase.value)
            # 8. Phase supervisor: each phase function returns when the
            #    ignition state changes (or, in pre-drive, on 'r' to re-run).
            while True:
                if _ignition.phase() is Phase.PREDRIVE:
                    run_predrive_assessment(
                        api_client, extractor, recognizer, _alert_manager, _head_pose,
                        _ignition, rate, debug_pose=args.debug_pose,
                        pose_release_hold=args.pose_release_hold,
                    )
                else:
                    run_monitoring(
                        api_client, extractor, recognizer, _alert_manager, _head_pose,
                        _ignition, rate, debug_pose=args.debug_pose,
                        pose_release_hold=args.pose_release_hold,
                    )

    except QuitRequested:
        logger.info("Quit requested from the preview window")
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl-C)")
    except Exception:
        logger.exception("Fatal error in main loop")
        exit_code = 1
    finally:
        cleanup()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
