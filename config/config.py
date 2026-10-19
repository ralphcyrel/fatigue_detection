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

# Per-frame FRS debug CSV (modules/frs_debug_log.py): every processed frame
# in both phases with raw metrics, normalised inputs, weighted terms, FRS,
# level and the reason for the level, to logs/debug_frs_<UTC stamp>.csv.
# Diagnostic only; off by default. FATIGUE_DEBUG_FRS_CSV=1 turns it on
# without editing this file.
DEBUG_FRS_CSV: Final[bool] = os.environ.get("FATIGUE_DEBUG_FRS_CSV", "0") == "1"

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
# In-vehicle display
# ---------------------------------------------------------------------------

# The 5" HDMI touchscreen. During pre-drive and monitoring it shows the data
# screen (modules/session_display.py) - never the camera image.
DISPLAY_WIDTH: Final[int] = 800
DISPLAY_HEIGHT: Final[int] = 480

# The data screen is redrawn at most this often; between redraws the loop
# skips imshow / waitKey entirely. Keys (q / i / r) are read on redraws.
DATA_SCREEN_HZ: Final[float] = 10.0

# Diagnostic MJPEG stream (main.py --debug-stream; modules/debug_stream.py).
# LAN only, never recorded. Frames are sent at most this often.
DEBUG_STREAM_PORT: Final[int] = 8080
DEBUG_STREAM_FPS: Final[float] = 10.0

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
# 5 V optocoupler module, LOW-level trigger, starter on the normally-OPEN
# contact: pin LOW = coil energised = start allowed; pin HIGH (the safe
# default) = coil released = starter inhibited. With the Pi unpowered the
# coil is released and the starter is inhibited. Levels live in
# modules/alert.py (_RELAY_ALLOW / _RELAY_INHIBIT).
RELAY_PIN: Final[int] = 17

# The relay control above is the Pi's ONLY connection to the starter circuit
# (2026-10-19). SMPS +5 V feeds the key switch's BAT terminal, START feeds
# relay COM and relay NO feeds the motor, so the motor runs only while the
# key is held at START *and* the relay is energised. Nothing reads the key.
# TODO(deferred hardware, next semester): ignition-position sense (was
# GPIO 25, IGNITION_PIN / IGNITION_ACTIVE_HIGH / IGNITION_DEBOUNCE_S) and a
# relay-contact feedback divider (GPIO 5). Both inputs are unwired; no code
# reads either pin. The start confirmation (START_CONFIRM_MODE) is where the
# ignition sense will plug in.

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
    "microsleep": 0.25,
}

# Weights that sit *on top of* the eye-metric partition rather than inside it
# (see modules.frs). Both were added after the eye weights and bands were
# tuned; carving them out would have rescaled every existing eye-only score.
_ADDITIVE_WEIGHTS: Final[tuple] = ("yawn", "microsleep")

# Guard against accidental edits: the eye-metric weights must still form a
# proper partition of 1.0 (the additive weights are excluded).
_EYE_WEIGHT_SUM: Final[float] = sum(
    w for k, w in FRS_WEIGHTS.items() if k not in _ADDITIVE_WEIGHTS
)
assert abs(_EYE_WEIGHT_SUM - 1.0) < 1e-9, (
    f"FRS_WEIGHTS eye metrics (all but {_ADDITIVE_WEIGHTS}) must sum to 1.0, "
    f"got {_EYE_WEIGHT_SUM}"
)

# ---------------------------------------------------------------------------
# Alert-level escalation (2026-10-15; modules/pipeline.LevelEscalator)
# ---------------------------------------------------------------------------

# ALERT -> WARNING once FRS >= 0.40 has held for this long. A head-pose alert
# (already held HEAD_POSE_ALERT_HOLD_S) enters WARNING without further delay.
LEVEL_WARNING_ENGAGE_S: Final[float] = 1.0
# WARNING -> DANGER only after this long continuously in WARNING with
# FRS >= 0.65 or a head-pose alert. There is no direct ALERT -> DANGER.
LEVEL_DANGER_ENGAGE_S: Final[float] = 3.0
# A level is lowered only after its condition has been false for this long.
LEVEL_RELEASE_HOLD_S: Final[float] = 2.0

# A confirmed microsleep forces DANGER directly (False), or only feeds the
# escalation above like a head-pose alert, i.e. it must spend
# LEVEL_DANGER_ENGAGE_S in WARNING first (True). Either way a microsleep
# still fails the pre-drive assessment outright.
MICROSLEEP_REQUIRES_WARNING: Final[bool] = False

# A gap longer than this between two processed frames (a loop stall, e.g.
# face recognition) is unknown time: blink and microsleep closures in
# progress are abandoned rather than extended across it.
MAX_FRAME_GAP_S: Final[float] = 0.5

# Yawn detection (modules/mar.YawnDetector), multiples of the driver's
# closed-mouth MAR baseline. An opening starts at MAR >= YAWN_MAR_RATIO x
# baseline, continues while MAR stays >= YAWN_MAR_RELEASE_RATIO x baseline,
# and is a yawn once it has lasted mar.MIN_YAWN_DURATION_S (1.5 s). The
# release ratio (2026-10-15) is hysteresis: a natural yawn wobbles around
# 2.0x, and with a single threshold run 081406 held 1.8-2.2x for 4.2 s yet
# never stayed >= 2.0x longer than 0.32 s. Invalid frames and gaps over
# MAX_FRAME_GAP_S end an opening, as for eye closures.
YAWN_MAR_RATIO: Final[float] = 2.0
YAWN_MAR_RELEASE_RATIO: Final[float] = 1.6

# Monitoring no-face policy (modules/pipeline.NoFacePolicy). Up to this long
# a lost face holds the level (and latches DANGER); past it the reported
# state is FAULT, whatever the level was - prolonged loss is a fault, not
# fatigue - until a valid face returns, when escalation restarts from the
# current score (2026-10-15; was 10 s from ALERT / WARNING and never from
# DANGER, which held DANGER through any gap).
NO_FACE_FAULT_S: Final[float] = 2.0

# ---------------------------------------------------------------------------
# Scoring floors and warm-up (2026-10-15)
# ---------------------------------------------------------------------------

# Floors on the calibrated baselines the FRS normalises against, applied at
# scoring time (modules/pipeline) to every source - stored calibrations,
# defaults and self-seeded runs - while the stored record keeps the measured
# value. The recorded calibrations measured 70-118 ms / 3-9 per min /
# 0.5-1.9 % at ~20 fps (dlib under-tracks short closures), against which one
# ordinary blink scored DANGER. These are not the degenerate-run floors in
# modules/calibration.py (MIN_*), which only prevent a zero divisor.
PERCLOS_BASELINE_FLOOR: Final[float] = 2.0             # percent
# Blink duration: 100 ms (2026-10-15; was 150 ms, which sat above the
# 85-135 ms alert blinks replayed, so the floor rather than the driver's
# baseline normalised the term).
BLINK_DURATION_BASELINE_FLOOR_MS: Final[float] = 100.0  # ms
BLINK_FREQUENCY_BASELINE_FLOOR: Final[float] = 6.0     # blinks per minute

# PERCLOS is scored against an absolute anchor, not the driver's baseline
# alone (2026-10-15): t = max(floored baseline, PERCLOS_REFERENCE) and
# excess = min(4.0, max(0, (PERCLOS - t) / t)). Against a 2 % baseline the
# pure ratio read ordinary blinking (~9 % at 20 fps) as an excess of 3.5;
# anchored, 9 % scores 0, 20 % scores 1.0 and 40 % scores 3.0. The warm-up
# prior (modules.perclos) still uses the floored baseline: it estimates this
# driver's PERCLOS, it does not score it.
PERCLOS_REFERENCE: Final[float] = 10.0                 # percent

# Blink frequency is anchored the same way (2026-10-15): t = max(floored
# baseline, BLINK_FREQ_REFERENCE), scaled to the time observed,
# min(elapsed, window) / 60, and excess = min(4.0, max(0, (f - t) / t)).
# Against the 6 /min floor a normal ~18 /min read as an excess of 2.0, and
# 30 /min or more sat at the cap (0.60 weighted, over the pass threshold on
# its own); anchored, 30 /min scores 0.5 and 40 /min 1.0.
BLINK_FREQ_REFERENCE: Final[float] = 20.0              # blinks per minute

# The blink-duration term is zero until at least this many blinks fall in
# its window (modules.blink.BLINK_MEAN_WINDOW_S): a one- or two-blink "mean"
# is a single sample.
BLINK_DURATION_MIN_BLINKS: Final[int] = 3
# The blink-frequency term is zero until this much time has been observed;
# after that the baseline is scaled to min(elapsed, window), not the window.
BLINK_FREQUENCY_MIN_ELAPSED_S: Final[float] = 10.0

# ---------------------------------------------------------------------------
# Head pose (2026-10-15: relative to the driver's own neutral pose)
# ---------------------------------------------------------------------------

# Deviation from the driver's neutral pose (degrees) that flags a pose: a
# nod is a pitch DROP of at least this much (pitch < 0 is head down), yaw /
# roll an absolute deviation beyond it. The neutral pose is the median over
# enrollment's calibration window, kept on this unit in POSE_NEUTRAL_FILE;
# without one, the median of the pre-drive settling period, or in monitoring
# of the first HEAD_POSE_NEUTRAL_SEED_S of face time (no pose alerts until
# then). The recorded neutral pitch on the bench mounting is +21..+27 deg,
# which the old absolute -15 deg limit could not account for.
HEAD_POSE_NOD_DROP_DEG: Final[float] = 15.0
HEAD_POSE_YAW_DEVIATION_DEG: Final[float] = 30.0
HEAD_POSE_ROLL_DEVIATION_DEG: Final[float] = 20.0
# A flagged pose must persist this long to count as a head-pose alert.
HEAD_POSE_ALERT_HOLD_S: Final[float] = 1.5
HEAD_POSE_NEUTRAL_SEED_S: Final[float] = 3.0

# Per-driver neutral head pose, keyed by driver id. Local to this unit on
# purpose: it depends on the camera mounting, and the backend API has no
# field for it.
POSE_NEUTRAL_FILE: Final[Path] = DATA_DIR / "pose_neutral.json"

# Landmark validity: a frame whose eye landmarks fall outside the image, or
# whose eye is narrower than this (pixels, corner to corner), is invalid and
# handled like a frame with no face - never scored as closed eyes.
LANDMARK_MIN_EYE_WIDTH_PX: Final[float] = 8.0

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

# MINIMUM length of the pre-drive assessment, and the length of its rolling
# window (2026-10-19). The assessment is not a fixed 30 s run: it keeps going
# for as long as the driver sits there before starting. Once it has run this
# long its first 30 s are judged exactly as the old fixed run was (a fail is
# a hard failure); from then on the most recent 30 s are re-judged on every
# frame and decide READY_TO_START. Reaching 30 s only makes the driver
# eligible to start - it never advances the phase.
PREDRIVE_ASSESSMENT_SECONDS: Final[int] = 30

# What confirms the start and moves READY_TO_START into continuous
# monitoring (2026-10-19). There is no timeout: without a confirmation the
# unit stays in READY_TO_START indefinitely.
#   "touch" - the BEGIN MONITORING button on the touchscreen launcher's
#             strip (sent to main.py as SIGUSR1; key 'b' in the window).
#   "gpio"  - the ignition-position sense (deferred hardware, next
#             semester). Not implemented: main.py refuses to start with it.
START_CONFIRM_MODE: Final[str] = "touch"

# While eligible (READY_TO_START, or re-assessing after it), the recognised
# driver is re-identified this often (seconds, measured from the end of the
# previous attempt) so a change of driver drops READY. Each attempt stalls
# the loop for ~1 s on the Pi; closures in progress are abandoned across it
# as for any stall (MAX_FRAME_GAP_S). Not run during the first 30 s, which
# stay exactly as the fixed assessment saw them. After a face loss longer
# than NO_FACE_FAULT_S the driver is re-identified on every face frame until
# confirmed. A different enrolled driver matched PREDRIVE_RECOGNITION_MATCHES
# times restarts the pre-drive for them.
PREDRIVE_REIDENTIFY_S: Final[float] = 10.0

# The verdict is taken on the WORST rolling window of this length (mean FRS
# over the window, maximised over the first PREDRIVE_ASSESSMENT_SECONDS, and
# then over the most recent PREDRIVE_ASSESSMENT_SECONDS), not on the overall mean:
# a 5 s microsleep in 25 s of alertness is exactly what a pre-drive check
# exists to catch, and a plain mean or median would dilute it. A window
# rather than a single-frame max keeps one landmark glitch from failing a
# driver. The pass threshold itself is modules.frs.WARNING_THRESHOLD, which the
# recorded assessments do not determine - see the note there.
PREDRIVE_WORST_WINDOW_SECONDS: Final[float] = 5.0

# Face recognition at pre-drive: try for this long and accept once the same
# driver_id has matched this many times. Timeout -> starter stays inhibited
# with LockReason.DRIVER_NOT_RECOGNIZED.
PREDRIVE_RECOGNITION_TIMEOUT_S: Final[float] = 10.0
PREDRIVE_RECOGNITION_MATCHES: Final[int] = 3

# Settling (2026-10-15): after recognition, the face and landmarks must be
# valid for this long continuously before anything is scored; a lost face or
# invalid landmarks restart it. The head pose seen here is the fallback
# neutral pose for a driver with none stored (see POSE_NEUTRAL_FILE). If it
# never completes within the timeout the run is an invalid assessment.
PREDRIVE_SETTLE_SECONDS: Final[float] = 3.0
PREDRIVE_SETTLE_TIMEOUT_S: Final[float] = 20.0

# The assessment is valid only if at least this fraction of its frames had a
# face with valid landmarks. Below it the result is FAULT (invalid
# assessment): the starter stays inhibited and the lock is raised as
# driver_not_recognized (the backend has no separate reason); the real cause
# is logged locally as INVALID_ASSESSMENT with the valid-frame percentage.
# Replaces PREDRIVE_MAX_NO_FACE_FRACTION (0.5).
PREDRIVE_MIN_VALID_FRACTION: Final[float] = 0.80

# How often (seconds, wall clock) to poll the backend for the operator's
# decision on a pending override request.
OVERRIDE_POLL_SECONDS: Final[float] = 5.0
