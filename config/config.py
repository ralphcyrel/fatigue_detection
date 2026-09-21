"""
Central configuration for the driver fatigue detection system.

Every tunable constant lives here so that the individual modules stay free of
magic numbers and the whole system can be re-tuned from a single place.
Values are grouped by concern: paths, camera, landmark detection, Laravel API,
GPIO pins, FRS weights and timing windows.

Target platform: Raspberry Pi 4B (8 GB) + NOIR Camera Module 3,
Raspberry Pi OS Bookworm, Python 3.9+.
"""

import os
from pathlib import Path
from typing import Dict, Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# BASE_DIR is the project root (the folder containing main.py). It is derived
# from this file's location so the project works no matter where it is cloned
# or which directory the process is launched from.
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

# Runtime data (landmark model, per-driver calibration files, etc.).
DATA_DIR: Final[Path] = BASE_DIR / "data"

# Rotating log files written by the alert / API modules.
LOGS_DIR: Final[Path] = BASE_DIR / "logs"

# Per-frame CSV + JSON summary of every pre-drive assessment (results data).
ASSESSMENTS_DIR: Final[Path] = LOGS_DIR / "assessments"

# Per-frame CSV + JSON summary of every enrollment calibration, so a stored
# baseline can be traced back to the raw EAR/MAR series that produced it.
CALIBRATIONS_DIR: Final[Path] = LOGS_DIR / "calibrations"

# Any additional trained models (e.g. custom face encoders) go here.
MODELS_DIR: Final[Path] = BASE_DIR / "models"

# dlib 68-point facial landmark predictor. Must be downloaded manually
# (see README.md) because the file is ~100 MB and not bundled with the repo.
LANDMARK_MODEL: Final[Path] = DATA_DIR / "shape_predictor_68_face_landmarks.dat"

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

# Index passed to cv2.VideoCapture / Picamera2. 0 is the first CSI/USB camera.
CAMERA_INDEX: Final[int] = 0

# 640x480 @ 30 fps is a deliberate trade-off: high enough resolution for
# reliable eye landmarks at driver distance, low enough for the Pi 4B to keep
# up with HOG face detection in real time.
CAMERA_WIDTH: Final[int] = 640
CAMERA_HEIGHT: Final[int] = 480
CAMERA_FPS: Final[int] = 30

# ---------------------------------------------------------------------------
# Landmark detection
# ---------------------------------------------------------------------------

# Face *detection* (the expensive HOG pass) is run on a frame downscaled by
# this factor. Landmark *prediction* is still done on the full-resolution
# frame so accuracy is not sacrificed. 0.5 roughly quarters detection cost.
SCALE_FACTOR: Final[float] = 0.5

# ---------------------------------------------------------------------------
# Laravel API
# ---------------------------------------------------------------------------

# Base URL of the Laravel backend that stores drivers, face encodings and
# fatigue events. Override via environment variable in deployment so the
# value does not need to be committed.
API_BASE_URL: Final[str] = os.environ.get(
    "FATIGUE_API_BASE_URL", "http://localhost:8000/api"
)

# Bearer token used for authenticating against the API. Read from the
# environment for the same reason as above — never hard-code secrets.
API_TOKEN: Final[str] = os.environ.get("FATIGUE_API_TOKEN", "")

# Seconds before an HTTP request is abandoned. Kept short so a dead network
# link never stalls the detection loop for long.
API_TIMEOUT: Final[int] = 5

# Identity of this Pi / vehicle unit. Assessments and operator override
# requests are keyed by it, because at pre-drive there may be no recognised
# driver to key them by. Set per vehicle in the environment.
DEVICE_ID: Final[str] = os.environ.get("FATIGUE_DEVICE_ID", "pi-01")

# Version of this Pi software, reported in every heartbeat so the operator
# portal can see which build each unit is running. Bump on release.
FIRMWARE_VERSION: Final[str] = "1.0.0"

# How often (seconds, wall clock) the main loop posts a heartbeat
# (POST /devices/{id}/heartbeat) so the portal can show the unit as online.
# Ticked from the detection loop itself - not a background timer - so a
# stalled loop stops heartbeating and the portal sees the unit go offline.
HEARTBEAT_INTERVAL_SECONDS: Final[float] = 30.0

# ---------------------------------------------------------------------------
# GPIO pins (BCM numbering)
# ---------------------------------------------------------------------------

# Starter-inhibit relay. It sits in the starter solenoid signal, NOT the
# ignition or fuel circuit, so it can only prevent the engine from being
# started - it can never stop a running engine. It is driven ONLY during
# the pre-drive phase (modules/alert.py refuses to lock in monitoring).
# Wired normally-OPEN for fail-secure behaviour: with the Pi unpowered, the
# pin LOW, or the process crashed before a pass, the starter is inhibited.
RELAY_PIN: Final[int] = 17

# Ignition sense input (fuse tap on an ignition-switched circuit). The 12 V
# tap MUST go through an optocoupler or a divider clamped to 3.3 V before it
# reaches this pin. Reads HIGH when the ignition is ON (IGNITION_ACTIVE_HIGH)
# and selects the operating phase: OFF -> pre-drive, ON -> monitoring.
IGNITION_PIN: Final[int] = 25
IGNITION_ACTIVE_HIGH: Final[bool] = True
# A raw reading must hold for this long before the reported state flips, so
# contact bounce / alternator ripple on the tap cannot flap the phase.
IGNITION_DEBOUNCE_S: Final[float] = 0.3

# Piezo buzzer for audible alerts.
BUZZER_PIN: Final[int] = 27

# Traffic-light style status LEDs: green = alert, yellow = drowsy,
# red = fatigued.
LED_GREEN: Final[int] = 22
LED_YELLOW: Final[int] = 23
LED_RED: Final[int] = 24

# ---------------------------------------------------------------------------
# Fatigue Risk Score (FRS) weights
# ---------------------------------------------------------------------------

# Each indicator contributes to the composite FRS in proportion to its
# weight. EAR and PERCLOS dominate because they are the most direct measures
# of eye closure; blink duration and frequency are supporting signals.
#
# The four *eye* weights form a partition of 1.0, so "every eye metric at
# 2x baseline" scores exactly 1.0. The yawn term is deliberately ADDITIVE on
# top of that partition rather than carved out of it: shrinking the eye
# weights to make room would have scaled every eye-only FRS (and therefore
# every tuned WARNING / DANGER threshold) by the same factor. Yawning can
# only add risk; it is fed to the FRS solely while a confirmed yawn is in
# progress (see modules/mar.py) and its excess is capped at 2.0 in
# modules/frs.py, so a full yawn contributes at most 0.15 * 2.0 = +0.30.
FRS_WEIGHTS: Final[Dict[str, float]] = {
    "ear": 0.35,
    "blink_duration": 0.20,
    "blink_frequency": 0.15,
    "perclos": 0.30,
    "yawn": 0.15,
}

# Guard against accidental edits: the eye-metric weights must still form a
# proper partition of 1.0 (the yawn weight is additive and excluded).
_EYE_WEIGHT_SUM: Final[float] = sum(
    w for k, w in FRS_WEIGHTS.items() if k != "yawn"
)
assert abs(_EYE_WEIGHT_SUM - 1.0) < 1e-9, (
    f"FRS_WEIGHTS eye metrics (all but 'yawn') must sum to 1.0, got {_EYE_WEIGHT_SUM}"
)

# ---------------------------------------------------------------------------
# Timing windows
# ---------------------------------------------------------------------------

# PERCLOS = percentage of time the eyes are closed over a sliding window.
# 60 s is the conventional window used in the drowsiness literature.
PERCLOS_WINDOW_SECONDS: Final[int] = 60

# How long (seconds) the calibration module observes the driver at the start
# of a session to learn their personal baseline EAR / blink statistics.
CALIBRATION_DURATION: Final[int] = 60

# ---------------------------------------------------------------------------
# Pre-drive assessment
# ---------------------------------------------------------------------------

# Length of the scored assessment window at ignition-OFF.
PREDRIVE_ASSESSMENT_SECONDS: Final[int] = 30

# The verdict is taken on the WORST rolling window of this length (mean FRS
# over the window, maximised over the assessment), not on the overall mean:
# a 5 s microsleep in 25 s of alertness is exactly what a pre-drive check
# exists to catch, and a plain mean or median would dilute it. A window
# rather than a single-frame max keeps one landmark glitch from failing a
# driver. The pass threshold itself is modules.frs.WARNING_THRESHOLD.
PREDRIVE_WORST_WINDOW_SECONDS: Final[float] = 5.0

# Face recognition at pre-drive: try for this long and accept once the same
# driver_id has matched this many times. Timeout -> starter stays inhibited
# with LockReason.DRIVER_NOT_RECOGNIZED.
PREDRIVE_RECOGNITION_TIMEOUT_S: Final[float] = 10.0
PREDRIVE_RECOGNITION_MATCHES: Final[int] = 3

# If the face is missing for more than this fraction of the assessment
# window the assessment is void (treated as not recognised).
PREDRIVE_MAX_NO_FACE_FRACTION: Final[float] = 0.5

# How often (seconds, wall clock) to poll the backend for the operator's
# decision on a pending override request.
OVERRIDE_POLL_SECONDS: Final[float] = 5.0
