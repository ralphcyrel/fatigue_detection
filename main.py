"""
Driver Fatigue Detection System - entry point.

Two phases and enrollment. There is no ignition input (2026-10-19): the Pi's
only connection to the starter circuit is the relay on GPIO 17, and the motor
runs only while the key is held at START *and* the relay is energised::

    --enroll        Enrollment (operator-supervised, at hiring): face encoding +
                    60 s alert-state calibration -> Laravel backend.
                    ``--driver-id N`` skips the terminal prompt (used by
                    ``launcher.py``, which has no keyboard).

    pre-drive       Recognise the driver, fetch their thresholds, settle 3 s,
                    then a ROLLING assessment with a 30 s minimum. At 30 s
                    the first 30 s are judged as before; a fail (or not
                    recognised / no baseline) keeps the starter inhibited and
                    raises an operator override request (LOCKED). On a pass
                    the most recent 30 s are re-judged every frame:
                    READY_TO_START (starter released, green LED) while they
                    pass, back to assessing (starter inhibited) when they
                    stop passing, the face is lost > NO_FACE_FAULT_S or the
                    driver changes. Monitoring begins only on the start
                    confirmation (``config.START_CONFIRM_MODE``: the BEGIN
                    MONITORING button), never on a timeout. This is the ONLY
                    phase in which the relay engages.

    monitoring      Continuous monitoring: same metrics, LEDs / buzzer /
                    backend notification only. The relay is never touched -
                    AlertManager refuses to lock in this phase (Module 8).

    END SESSION     (launcher strip, SIGUSR2, key 'e') from any state: starter
                    inhibited, alerts off, session closed, IDLE until START
                    NEXT DRIVER - the process keeps running. Key-OFF cannot be
                    sensed, so this is how a session ends.

    The default and --sequence run pre-drive -> monitoring (the touchscreen
    launcher's default); --force-phase P tests one phase alone (a confirmed
    start then returns to IDLE instead of entering monitoring).

Per-frame metrics are computed by one shared pipeline (Module 12) in both
phases::

    camera -> LandmarkExtractor -> MetricsPipeline (EAR / MAR / blink / PERCLOS
              / yawn / head pose / FRS) -> phase-specific decision -> AlertManager

In both phases the loop also posts a heartbeat every
``config.HEARTBEAT_INTERVAL_SECONDS`` (``POST /devices/{id}/heartbeat``) carrying
the relay state as actually driven (``normal`` / ``interrupted``), so the
operator portal can show the unit as online and spot one that has not applied
a command. With ``--debug-stream`` the same tick also re-sends where the video
stream can be opened (``PUT /devices/{id}/stream``) for the portal's live view;
``cleanup()`` withdraws it.

With ``--state-file PATH`` (the launcher passes it) the current session state
(``modules.phase.SessionState``) is written to PATH as JSON on every change,
so the launcher can show BEGIN MONITORING only in READY_TO_START.

Runs on the Raspberry Pi (Picamera2 + GPIO) and, with automatic fallbacks,
on a development machine (cv2.VideoCapture + mocked GPIO).
"""

import argparse
import itertools
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import config
from modules.alert import AlertManager
from modules.api import (CALIBRATION_SOURCE_DEVICE, REPAIR_EAR_PAIR_SELF_SEEDED,
                         REPAIR_EAR_THRESHOLD_RECOMPUTED, REPAIR_PERCLOS_BASELINE_FLOORED,
                         SELF_SEEDED_PROVENANCE, APIClient, Calibration, repair_defaulted)
from modules.assessment import PredriveAssessment
from modules.blink import BlinkDetector, MicrosleepDetector
from modules.calibration import (CLOSED_CAPTURE_ATTEMPTS, EAR_THRESHOLD_RATIO,
                                 MIN_PERCLOS_BASELINE, SELF_SEED_S, CalibrationManager,
                                 ClosedEyeCapture, ear_calibration_problem, finite_or_none)
from modules.debug_stream import DebugStream, lan_address, stream_token
from modules.ear import EARCalculator
from modules.face_recognition_module import DriverRecognizer
from modules.frs_debug_log import FrsDebugLog
from modules.head_pose import HeadPoseEstimator
from modules.ignition import ForcedIgnition
from modules.landmark import LandmarkExtractor
from modules.mar import MARCalculator
from modules.perclos import PERCLOSCalculator
from modules.phase import LockReason, Phase, SessionState
from modules.pipeline import (HEAD_POSE_RELEASE_HOLD_S, FrameMetrics, HeadPoseDebounce,
                              MetricsPipeline, SettlingWatch, landmark_problem)
from modules.pose_neutral import load_neutral, median_pose, save_neutral
from modules.recognition_worker import InlineRecognition, KnownFaces, RecognitionWorker
from modules.session_display import (FACE_NONE, FACE_OK, FACE_SEARCHING,
                                     FACE_UNRECOGNISED, GREEN, RED, EnrollState, ScreenState,
                                     render as render_data_screen, render_enroll)
from modules.start_confirm import (BEGIN, END_SESSION, OperatorRequests, TouchStartConfirm,
                                   make_start_confirm)
from modules.text_render import put_text, text_size

logger = logging.getLogger("fatigue")

# ---------------------------------------------------------------------------
# Tunables local to the main loop
# ---------------------------------------------------------------------------

# Monitoring, driver not (yet) recognised: recognition is retried on the
# first face frame at least this long after the previous attempt ENDED.
# Each attempt blocks the loop for ~1 s on the Pi (dlib holds the GIL, so it
# cannot be threaded); retrying every frame held an unrecognised driver at
# ~1 fps, too blind for blink / microsleep detection. At 5 s the loop sees
# ~5 s of every ~6 s. Nothing is accepted between attempts - the driver
# simply stays unrecognised (no calibration, self-seeded baseline).
UNRECOGNISED_RETRY_S: float = 5.0

# While the driver stays in DANGER (monitoring), log at most one event per
# this many seconds - otherwise a 30 fps loop would POST 30 events per second.
DANGER_EVENT_INTERVAL: float = 5.0

# An open monitoring fault is re-POSTed (status "open", updated gap_s) this
# often, so the portal can tell a fault that is still live from one whose
# device went silent.
MONITORING_FAULT_REFRESH_S: float = 5.0

# NoFaceVerdict.event -> the fault_type it opens on POST /monitoring-faults.
FAULT_TYPE_FOR_EVENT: Dict[str, str] = {
    "fault_entered": "no_face",
    "latch_lost": "danger_latched",
}
# Opened when the EAR self-seed gives up (FrameMetrics.seed_failed).
FAULT_TYPE_NO_EAR_BASELINE: str = "no_ear_baseline"
# Opened when monitoring recognises someone other than the driver who passed
# pre-drive (2026-10-19). The backend has no type for it, and adding one is an
# API change, so it goes out as the closest existing one: no_ear_baseline -
# the face is seen, but this driver is scored on a self-seeded baseline, not
# an assessed calibration. Same body (gap_s = seconds open, last_known null),
# resolved "ignition_off" when monitoring ends. Distinguished locally by the
# DRIVER MISMATCH log line and the run summary's driver_mismatch.
FAULT_TYPE_DRIVER_MISMATCH: str = FAULT_TYPE_NO_EAR_BASELINE


@dataclass
class OpenFault:
    """A monitoring fault that has been opened on the backend and not resolved."""

    fault_uuid: str
    fault_type: str
    driver_id: Optional[int]
    entry_level: str
    started_at: datetime                  # UTC, when the face was lost
    last_known: Optional[Dict[str, Any]]  # FrameMetrics.last_known() before the gap
    gap_s: float                          # latest no-face duration seen
    last_push_mono: float                 # time.monotonic() of the last POST
    # time.monotonic() the fault began: what its duration is measured from
    # (started_at is the wall clock, for the backend).
    opened_mono: float = 0.0

    def push(self, api_client: APIClient, status: str, now_mono: float,
             resolved_at: Optional[datetime] = None,
             resolution: Optional[str] = None) -> None:
        """POST this fault's current state (fire-and-forget)."""
        api_client.push_monitoring_fault(
            self.fault_uuid, status, self.fault_type, self.driver_id, self.entry_level,
            self.started_at, self.gap_s, self.last_known, resolved_at, resolution,
        )
        self.last_push_mono = now_mono

    def refresh_elapsed(self, api_client: APIClient, now_mono: float) -> None:
        """
        Keep a fault that is not a no-face gap (``no_ear_baseline``) live:
        ``gap_s`` becomes the seconds since it opened, re-POSTed every
        ``MONITORING_FAULT_REFRESH_S``.
        """
        self.gap_s = now_mono - self.opened_mono
        if now_mono - self.last_push_mono >= MONITORING_FAULT_REFRESH_S:
            self.push(api_client, "open", now_mono)

# Pre-drive -> monitoring hand-off: after the start confirmation a green
# message is shown on live frames for this long, before monitoring starts.
# Kept under 2 s.
RELEASE_INDICATOR_S: float = 1.5

# Frames of face captured for the enrollment encoding.
ENROLL_FRAMES: int = 30

# Enrollment: wall-clock seconds of the driver's own EAR observed before
# calibration starts, to derive a personal closure threshold (see
# run_enrollment). Time-scoped so a slow loop settles for the same period
# as a fast one; SEED_MIN_SAMPLES guards a face that appears late.
SEED_WINDOW_S: float = 1.0
SEED_MIN_SAMPLES: int = 5

# Enrollment closed-eye capture (modules.calibration.ClosedEyeCapture): an
# on-screen countdown this long before each attempt, and this long with the
# eyes open again before the 60 s window starts, so the reopening (and any
# eye-rubbing) stays out of the alert-state baselines.
CLOSED_PROMPT_LEAD_S: float = 3.0
REOPEN_SETTLE_S: float = 2.0

# Enrollment step 1, positioning (before the face capture, so the operator
# can fine-tune with the debug stream while the panel shows text guidance).
# The face box must stay inside these limits for ENROLL_POSITION_HOLD_S.
# Starting values, not yet measured on the rig: dlib's HOG finds faces down
# to ~80 px at the 0.5 detection scale, i.e. ~0.25 of a 640 px frame, so the
# floor sits just above that. Each positioning step logs the face size and
# centre it accepted, for tuning.
ENROLL_FACE_MIN_W: float = 0.27     # face box width / frame width
ENROLL_FACE_MAX_W: float = 0.65
ENROLL_CENTRE_TOL_X: float = 0.15   # |centre - 0.5| as a fraction of the frame
ENROLL_CENTRE_TOL_Y: float = 0.18
ENROLL_POSITION_HOLD_S: float = 3.0
ENROLL_POSITION_TIMEOUT_S: float = 90.0
# The panel's outline is drawn as a MIRROR seen by the driver (they move to
# their left, the box moves left), and "move left / right" is worded for the
# driver. The camera faces the driver, so the left of an unflipped camera
# image is the driver's RIGHT. Set True only if the camera is configured to
# flip the image horizontally (Picamera2 does not by default).
ENROLL_CAMERA_HFLIP: bool = False

# run_enrollment exit codes beyond the original 2 / 3 / 5.
EXIT_NO_EYELID_CONTRAST: int = 6    # closed-eye check failed; nothing POSTed
EXIT_CLOSED_EAR_DROPPED: int = 7    # POSTed, but the backend did not keep ear_closed_baseline

# Log the measured loop rate every N frames.
FPS_LOG_INTERVAL: int = 30

# Population values for the non-EAR baselines. Used in MONITORING when there
# is no usable calibration (driver unrecognised, no record, backend offline),
# and to fill fields missing from a partial backend record. Pre-drive never
# runs on them: a missing baseline there is a lock condition.
#
# There is deliberately no EAR baseline or threshold here. The old population
# pair (0.30 / 0.225) put the closure threshold at 88 % of a 0.255 EAR -
# inside landmark noise - and on 2026-09-23 drove 130 level changes in 71 s
# for an unrecognised driver. Without a calibration the EAR pair is seeded
# from the driver's own EAR (modules.calibration.EarSelfSeed), and until then
# only the overrides run.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "perclos_baseline": 5.0,
    "blink_duration_baseline": 150.0,
    "blink_frequency_baseline": 5.0,
    # Outer-lip MAR of a resting closed mouth is ~0.4-0.6; the yawn threshold
    # is 2x that (modules/mar.py). Drivers enrolled before yawn detection
    # existed have neither key in the backend and get these generic values.
    "mar_baseline": 0.45,
    "yawn_threshold": 0.90,
}

# check_threshold_sanity: how far ear_threshold / ear_baseline may stray from
# EAR_THRESHOLD_RATIO before the pair counts as inconsistent.
EAR_RATIO_TOLERANCE: float = 0.10

# Threshold keys that only exist for drivers enrolled with yawn detection.
MAR_THRESHOLD_KEYS = ("mar_baseline", "yawn_threshold")

# Text colours (BGR) per alert level for the overlay.
LEVEL_BGR: Dict[str, tuple] = {
    "ALERT": (0, 200, 0),
    "WARNING": (0, 220, 255),
    "DANGER": (0, 0, 255),
    "FAULT": (255, 0, 255),   # magenta: not a fatigue level
}
WHITE = (255, 255, 255)
GREY = (160, 160, 160)

WINDOW_NAME = "Driver Fatigue Detection"

# Preview-window keys. ('i', the mock-ignition toggle, went with the ignition
# input on 2026-10-19.)
KEY_QUIT = ord("q")
KEY_RETRY = ord("r")      # pre-drive: re-run the assessment
KEY_BEGIN = ord("b")      # = the launcher's BEGIN MONITORING / START NEXT DRIVER
KEY_END_SESSION = ord("e")  # = the launcher's END SESSION (no confirmation step)

# Human-readable lock reasons for the overlay.
LOCK_REASON_TEXT: Dict[LockReason, str] = {
    LockReason.FATIGUE_DETECTED: "FATIGUE DETECTED",
    LockReason.MICROSLEEP_DETECTED: "MICROSLEEP DETECTED",
    LockReason.DRIVER_NOT_RECOGNIZED: "DRIVER NOT RECOGNIZED",
    LockReason.NO_BASELINE: "NO BASELINE ON FILE",
    LockReason.FOREIGN_DEVICE_BASELINE: "BASELINE FROM ANOTHER UNIT - RE-ENROL",
}

# ---------------------------------------------------------------------------
# Module-level handles so cleanup() can reach them from any exit path
# ---------------------------------------------------------------------------
_camera: Optional["Camera"] = None
_alert_manager: Optional[AlertManager] = None
_head_pose: Optional[HeadPoseEstimator] = None
_ignition: Any = None
_heartbeat: Optional["Heartbeat"] = None
# Display window on. False under --no-preview, or once cv2.imshow has failed
# (headless).
_display_available: bool = True
# What the window shows. "data": the data screen, no camera image - the
# default for pre-drive and monitoring (group decision 2026-10-05). "enroll":
# the enrollment screen (live mirror view so the driver can position
# themselves, face box, guidance, countdown - 2026-10-07: the one phase the
# panel shows the camera). "video": the annotated camera preview
# (--show-video, a diagnostic only). "none": --no-preview.
_display_mode: str = "data"
_fullscreen: bool = False
_window_ready: bool = False
# Bottom px of the data screen left empty for the launcher's STOP strip.
_reserve_bottom: int = 0
_screen = ScreenState()
_enroll = EnrollState()
_last_screen_draw: float = float("-inf")
_rate: Optional["LoopRate"] = None
# Diagnostic MJPEG stream (--debug-stream), else None.
_stream: Optional[DebugStream] = None
# Latched per frame in capture_frame(): whether this frame's draw_* calls
# annotate the camera image (video mode, or the stream wants this frame) and
# whether present() hands it to the stream. Otherwise the helpers only fill
# _screen and the frame is never drawn on.
_annotate_frame: bool = False
_stream_frame: bool = False
# LoopProfiler under --profile-loop (monitoring only), else a no-op.
_profiler: Any = None
# Per-frame FRS debug CSV (config.DEBUG_FRS_CSV); a no-op when the flag is off.
_frs_debug = FrsDebugLog()
# Outcome of run_enrollment for --result-file (the launcher's result screen).
_enroll_report: Dict[str, Any] = {}
# Operator requests (BEGIN / END_SESSION) from the launcher's signals and the
# window's keys, and the start confirmation built on them. main() replaces
# _start_confirm per config.START_CONFIRM_MODE.
_requests = OperatorRequests()
_start_confirm: Any = TouchStartConfirm(_requests)
# Where the session is (None = not a pre-drive / monitoring session, e.g.
# enrollment), and the --state-file it is published to.
_session_state: Optional[SessionState] = None
_state_note: str = ""
_state_file: Optional[str] = None
# Face recognition off the frame loop (modules/recognition_worker.py): the
# one worker process, used by pre-drive re-identification and monitoring.
# None -> identify() inline (tests, enrollment, or it could not be started).
# _recognition_factory rebuilds the recogniser in the worker; start / stop
# only through start_recognition() / stop_recognition().
_recognition: Optional[RecognitionWorker] = None
_recognition_factory: Any = None
# Tags recognition requests with their owner (one pre-drive DriverCheck, one
# monitoring run), so an answer that arrives after its owner is gone - a
# re-run, a change of driver, the hand-off to monitoring - is never taken by
# the next one.
_recognition_owner = itertools.count(1)
# The calibration the current pre-drive cycle loaded and used (driver_id ->
# Calibration), and the driver it released the starter for. Monitoring
# entered from that release reuses it instead of fetching it again
# (2026-10-19). Cleared at every pre-drive cycle and END SESSION.
_calibration_cache: Dict[int, Calibration] = {}
_released_driver: Optional[Dict[str, Any]] = None


class QuitRequested(Exception):
    """Raised when the user presses ``q`` in the preview window."""


class PhaseChanged(Exception):
    """Raised inside a phase loop when the ignition state no longer matches it."""


class SessionEnded(Exception):
    """Raised by :func:`present` on END SESSION; the supervisor returns to IDLE."""


# ---------------------------------------------------------------------------
# Session state and operator requests (2026-10-19)
# ---------------------------------------------------------------------------

def _stamp(t: float) -> str:
    """``time.time()`` as local ``YYYY-mm-dd HH:MM:SS.mmm`` for the log."""
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def set_session_state(state: SessionState, note: str = "") -> None:
    """Enter ``state`` (logged once per change) and publish it to ``--state-file``."""
    global _session_state, _state_note
    if state is _session_state and note == _state_note:
        return
    previous = _session_state
    _session_state, _state_note = state, note
    if state is not previous:
        logger.info("Session state %s -> %s%s", previous.value if previous else "-",
                    state.value, f" ({note})" if note else "")
    _write_state_file()


def _write_state_file() -> None:
    """Atomically replace ``--state-file`` with the current state (best effort)."""
    if not _state_file or _session_state is None:
        return
    body = {"state": _session_state.value, "note": _state_note, "driver": _screen.driver_name,
            "pid": os.getpid(), "since": time.time()}
    tmp = f"{_state_file}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(body, fh)
        os.replace(tmp, _state_file)
    except OSError as exc:
        logger.warning("Could not write the state file %s: %s", _state_file, exc)


def request_operator(kind: str, source: str) -> None:
    """Record an operator request (signal-handler safe: it only stores)."""
    _requests.request(kind, _session_state.value if _session_state else "-", source)


def _install_request_signals() -> List[Tuple[int, Any]]:
    """
    SIGUSR1 -> BEGIN, SIGUSR2 -> END SESSION (the launcher's strip buttons).
    POSIX only; returns ``[(signum, previous handler)]`` for restoring.
    """
    installed = []
    for name, kind in (("SIGUSR1", BEGIN), ("SIGUSR2", END_SESSION)):
        signum = getattr(signal, name, None)
        if signum is None:
            continue                     # Windows (development): keys b / e only
        try:
            previous = signal.signal(signum, lambda _s, _f, _k=kind, _n=name:
                                     request_operator(_k, _n))
        except ValueError:               # not the main thread (tests)
            continue
        installed.append((signum, previous))
    return installed


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
    """
    Grab one frame from the global camera (Picamera2 or cv2).

    Every loop takes its frame through here, so this is also where a new
    frame's display state starts: the data screen's per-frame fields are
    cleared and the annotate / stream decision is latched for the frame.
    """
    global _annotate_frame, _stream_frame
    if _camera is None:
        return None
    frame = _camera.read()
    _screen.begin_frame()
    _stream_frame = _stream is not None and _stream.wants_frame()
    _annotate_frame = _stream_frame or (_display_mode == "video" and _display_available)
    return frame


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


class _NullSection:
    """Context manager that does nothing (profiling off / outside a frame)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return False


_NULL_SECTION = _NullSection()


class LoopProfiler:
    """
    ``--profile-loop``: where the monitoring loop's wall time goes, per frame.

    ``frame()`` is called once at the top of every loop iteration and closes
    the previous frame, so a frame's total is the full iteration time
    including every ``continue`` path, and the frame totals of a window add
    up to its wall time. Work inside a frame is attributed with
    ``with profiler.section(name):``. Sections may nest (``present()`` times
    the heartbeat inside the display section); a parent is charged only its
    own time, never its children's. Whatever no section covers is reported
    as ``other``.

    Every ``window_s`` of wall time a summary is logged and the window's
    frames are appended to a CSV (``logs/loop_profile_<UTC>.csv``, one row
    per frame, milliseconds).

    Only main-thread time is measured. The heartbeat / event / fault POSTs
    and the buzzer run on their own threads; their cost can only appear as
    slower main-thread sections (GIL contention), never under ``api``.
    """

    CATEGORIES = ("capture", "landmarks", "recognition", "pipeline", "gpio", "api", "display")

    def __init__(self, window_s: float = 60.0, label: str = "monitoring") -> None:
        self.window_s = window_s
        self.label = label
        self._csv_path = config.LOGS_DIR / (
            f"loop_profile_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv")
        self._csv_header_written = False
        self._rows: list = []                 # finished frames of the current window
        self._frame: Optional[Dict[str, float]] = None
        self._frame_start = 0.0
        self._window_start: Optional[float] = None
        self._stack: list = []                # [name, start, child_time] of open sections

    # ---- per-frame hooks ---------------------------------------------------

    def frame(self) -> None:
        """Close the previous frame (if any) and open a new one."""
        now = time.perf_counter()
        self._close_frame(now)
        if self._window_start is None:
            self._window_start = now
        elif now - self._window_start >= self.window_s:
            self.report(now)
            self._window_start = now
        self._frame = dict.fromkeys(self.CATEGORIES, 0.0)
        self._frame_start = now

    def section(self, name: str) -> Any:
        """Time a block under ``name`` (no-op outside a frame)."""
        if self._frame is None:
            return _NULL_SECTION
        return _Section(self, name)

    def stop(self) -> None:
        """Close the last frame and report the partial window (loop exit)."""
        now = time.perf_counter()
        self._close_frame(now)
        if self._rows:
            self.report(now)
        self._window_start = None

    # ---- internals ---------------------------------------------------------

    def _enter(self, name: str) -> None:
        self._stack.append([name, time.perf_counter(), 0.0])

    def _exit(self) -> None:
        name, start, child = self._stack.pop()
        elapsed = time.perf_counter() - start
        if self._frame is not None:
            self._frame[name] = self._frame.get(name, 0.0) + elapsed - child
        if self._stack:
            self._stack[-1][2] += elapsed

    def _close_frame(self, now: float) -> None:
        if self._frame is None:
            return
        row = self._frame
        row["total"] = now - self._frame_start
        row["other"] = max(0.0, row["total"] - sum(row[c] for c in self.CATEGORIES))
        self._rows.append(row)
        self._frame = None

    def report(self, now: Optional[float] = None) -> None:
        """Log the summary of the finished frames and flush them to the CSV."""
        rows, self._rows = self._rows, []
        if not rows:
            return
        wall = sum(r["total"] for r in rows)
        totals = np.array([r["total"] for r in rows]) * 1000.0
        lines = [
            f"LOOP PROFILE ({self.label}) - {wall:.1f} s wall, {len(rows)} frames, "
            f"{len(rows) / wall:.1f} fps",
            f"  frame time ms: mean {totals.mean():.1f}  p50 {np.percentile(totals, 50):.1f}  "
            f"p95 {np.percentile(totals, 95):.1f}  max {totals.max():.1f}  |  "
            f"frames > 100 ms: {int((totals > 100).sum())}",
            f"  {'category':<12} {'total s':>8} {'share':>7} {'frames':>7} "
            f"{'ms/frame':>9} {'ms/call':>8} {'max ms':>8}",
        ]
        for cat in self.CATEGORIES + ("other",):
            vals = np.array([r[cat] for r in rows]) * 1000.0
            ran = vals[vals > 0]
            lines.append(
                f"  {cat:<12} {vals.sum() / 1000:8.2f} {vals.sum() / 10 / wall:6.1f}% "
                f"{len(ran):7d} {vals.mean():9.2f} "
                f"{(ran.mean() if len(ran) else 0.0):8.2f} {vals.max():8.1f}")
        summary = "\n".join(lines)
        logger.info("\n%s", summary)
        try:
            config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            cols = ("total",) + self.CATEGORIES + ("other",)
            with open(self._csv_path, "a", encoding="utf-8") as fh:
                if not self._csv_header_written:
                    fh.write(",".join(f"{c}_ms" for c in cols) + "\n")
                    self._csv_header_written = True
                for r in rows:
                    fh.write(",".join(f"{r[c] * 1000:.3f}" for c in cols) + "\n")
        except OSError as exc:
            logger.warning("Could not write loop profile CSV %s: %s", self._csv_path, exc)


class _Section:
    """One timed block of a :class:`LoopProfiler` frame."""

    __slots__ = ("profiler", "name")

    def __init__(self, profiler: LoopProfiler, name: str) -> None:
        self.profiler = profiler
        self.name = name

    def __enter__(self) -> None:
        self.profiler._enter(self.name)

    def __exit__(self, *exc: Any) -> bool:
        self.profiler._exit()
        return False


class _NoProfiler:
    """Stand-in when ``--profile-loop`` is off: every hook is a no-op."""

    def frame(self) -> None:
        pass

    def section(self, name: str) -> _NullSection:
        return _NULL_SECTION

    def stop(self) -> None:
        pass


_NO_PROFILER = _NoProfiler()


class Heartbeat:
    """
    Periodic "still online" post to the portal, ticked from the frame loop.

    Fires every ``config.HEARTBEAT_INTERVAL_SECONDS`` of wall-clock time,
    the first one on the first tick so the portal sees the unit as soon as
    the supervisor starts. It is driven from the loop rather than a
    background timer on purpose: a stalled detection loop then stops
    heartbeating and the portal shows the unit offline. The POST itself
    runs on a daemon thread (``APIClient.post_heartbeat``), so a slow or
    dead backend never delays a frame.

    ``confirmed_state`` is read from ``AlertManager`` at send time - the
    relay as actually driven, not as commanded (``UNLOCKED`` -> ``normal``,
    ``LOCKED`` -> ``interrupted``). Without an ``AlertManager`` the state is
    reported as ``unknown``; in practice ``main()`` creates the manager
    (which drives the relay to its fail-secure state) before this ticker,
    so a real run never sends ``unknown``.
    """

    def __init__(
        self,
        api_client: APIClient,
        alert_manager: Optional[AlertManager],
        interval: float = config.HEARTBEAT_INTERVAL_SECONDS,
        stream: Optional[DebugStream] = None,
    ) -> None:
        self.api_client = api_client
        self.alert_manager = alert_manager
        self.interval = interval
        # --debug-stream server, if running: its address rides along with
        # every beat (the portal forgets it 90 s after the last one).
        self.stream = stream
        self._stream_reported = False
        self._last_sent: Optional[float] = None

    def tick(self, now: Optional[float] = None) -> bool:
        """
        Queue a heartbeat (and stream report) if the interval has elapsed.

        Args:
            now: ``time.monotonic()`` for this frame (taken here if omitted).

        Returns:
            ``True`` if a heartbeat was queued on this tick.
        """
        now = time.monotonic() if now is None else now
        if self._last_sent is not None and now - self._last_sent < self.interval:
            return False
        self._last_sent = now
        relay = self.alert_manager.get_relay_state() if self.alert_manager else None
        self.api_client.post_heartbeat(relay)
        if self.stream is not None:
            # Address re-read each time: DHCP may move the unit mid-session.
            self.api_client.report_stream(lan_address(), self.stream.port, self.stream.token)
            self._stream_reported = True
        return True

    def withdraw_stream(self) -> None:
        """Take the stream off the portal at shutdown (only if it was listed)."""
        if self._stream_reported:
            self.api_client.clear_stream()
            self._stream_reported = False


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
    global _display_available, _window_ready
    if not _display_available:
        return -1
    try:
        if not _window_ready:
            # Fullscreen (launcher on the 800x480 panel) needs a resizable
            # window; otherwise imshow's default autosized window, as before.
            if _fullscreen:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            _window_ready = True
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
            "run `pip uninstall -y opencv-python-headless && pip install \"opencv-python<4.12\"` "
            "(see requirements.txt); "
            "otherwise check DISPLAY / that you are on the Pi's desktop, not SSH.",
            detail,
        )
        print(
            "\n" + "=" * 72 + "\n"
            "  !! NO PREVIEW WINDOW - running headless for the rest of this run !!\n"
            f"  cv2.imshow failed: {detail}\n"
            "  Fix (GUI-less OpenCV):  pip uninstall -y opencv-python-headless && "
            "pip install \"opencv-python<4.12\"\n"
            "  Fix (no display):       run from the Pi desktop terminal, or set DISPLAY=:0\n"
            + "=" * 72 + "\n",
            flush=True,
        )
        return -1


def present(frame: np.ndarray) -> int:
    """
    Show ``frame`` and act on the global keys and operator requests.

    ``q`` raises :class:`QuitRequested`. ``b`` / ``e`` make the same BEGIN /
    END SESSION requests as the launcher's buttons (SIGUSR1 / SIGUSR2). A
    pending END SESSION raises :class:`SessionEnded` here - every loop passes
    through, so a session can be ended from any state; in IDLE (or outside
    a session) it is dropped. Other keys are returned to the caller (e.g.
    ``r`` to retry).

    Every loop in both phases passes through here once per frame, so this
    is also where the :class:`Heartbeat` is ticked (a no-op until
    ``config.HEARTBEAT_INTERVAL_SECONDS`` have elapsed; never blocks).
    """
    prof = _profiler if _profiler is not None else _NO_PROFILER
    if _heartbeat is not None:
        with prof.section("api"):
            _heartbeat.tick()
    with prof.section("display"):
        key = _display(frame)
    if key == KEY_QUIT:
        raise QuitRequested()
    if key == KEY_BEGIN:
        request_operator(BEGIN, "key 'b'")
    elif key == KEY_END_SESSION:
        request_operator(END_SESSION, "key 'e'")
    if _requests.pending(END_SESSION):
        if _session_state is None or _session_state is SessionState.IDLE:
            _requests.discard(END_SESSION, "no session to end")
        else:
            raise SessionEnded()
    return key


def _display(frame: np.ndarray) -> int:
    """
    One frame's worth of display work; returns the key pressed (or -1).

    The (annotated) camera frame goes to the debug stream if it wanted this
    frame, and to the window only in video mode. In data mode the window
    gets the data screen instead, redrawn at most ``config.DATA_SCREEN_HZ``;
    frames in between skip imshow / waitKey entirely.
    """
    global _last_screen_draw
    if _stream_frame and _stream is not None:
        _stream.offer(frame)
    if _display_mode == "video":
        return show_frame(frame)
    if _display_mode not in ("data", "enroll") or not _display_available:
        return -1
    now = time.monotonic()
    if now - _last_screen_draw < 1.0 / config.DATA_SCREEN_HZ:
        return -1
    _last_screen_draw = now
    if _display_mode == "enroll":
        canvas = render_enroll(_enroll, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT,
                               _reserve_bottom)
    else:
        _screen.fps = _rate.fps if _rate is not None else None
        canvas = render_data_screen(_screen, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT,
                                    _reserve_bottom)
    return show_frame(canvas)


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> None:
    """The 68 landmark points, in place (eyes yellow) - annotated view only."""
    for idx, (x, y) in enumerate(landmarks):
        color = (0, 255, 255) if 36 <= idx < 48 else (255, 255, 255)
        cv2.circle(frame, (int(x), int(y)), 1, color, -1)


def draw_banner(frame: np.ndarray, text: str, color: tuple = WHITE) -> None:
    """Bottom strip with the phase / status text, drawn in place."""
    _screen.phase, _screen.phase_color = text, color
    if not _annotate_frame:
        return  # nobody sees the camera image this frame
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    put_text(frame, text, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_overlay(
    frame: np.ndarray,
    driver: Optional[Dict[str, Any]],
    m: FrameMetrics,
    fps: Optional[float] = None,
    pipeline: Optional[MetricsPipeline] = None,
    thresholds: Optional[Dict[str, float]] = None,
    landmarks: Optional[np.ndarray] = None,
    calibrated: bool = True,
) -> None:
    """
    Annotate ``frame`` in place with driver name, FRS, level, EAR, PERCLOS,
    MAR / yawn state, relay state, measured loop rate and head pose, plus a
    MICROSLEEP banner while that override is active - and fill the data
    screen with the same numbers next to the driver's baselines.

    Call it after every other use of ``frame`` (recognition must never see
    the drawn landmarks).

    Args:
        frame: BGR frame to draw on.
        driver: Dict from ``DriverRecognizer.identify()`` (may be ``None``).
        m: Pipeline output for this frame.
        fps: Last measured loop rate, or ``None`` before the first sample.
        pipeline: The pipeline that produced ``m`` (data screen: blink
            window, self-seeded EAR baseline).
        thresholds: The baselines ``m`` was scored against (data screen).
        landmarks: Drawn on the annotated view when given.
        calibrated: ``thresholds`` are the driver's own calibration (else
            population defaults / a self-seeded EAR baseline).
    """
    _fill_screen(driver, m, pipeline, thresholds, calibrated)
    if not _annotate_frame:
        return  # nobody sees the camera image this frame
    if landmarks is not None:
        draw_landmarks(frame, landmarks)
    level = m.level
    color = LEVEL_BGR.get(level, WHITE)
    name = driver["name"] if driver else "Unknown"
    font = cv2.FONT_HERSHEY_SIMPLEX
    w = frame.shape[1]

    # Dark strip behind the text keeps it legible over a bright face.
    cv2.rectangle(frame, (0, 0), (w, 135), (0, 0, 0), -1)

    put_text(frame, f"Driver: {name}", (10, 25), font, 0.6, WHITE, 1)
    cv2.putText(frame, f"FPS: {fps:.1f}" if fps is not None else "FPS: --",
                (w - 150, 25), font, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FRS: {m.frs:.3f}" if m.scored
                else "FRS: -- (NO EAR BASELINE)" if m.seed_failed
                else "FRS: -- (learning EAR baseline)",
                (10, 50), font, 0.6, color, 2, cv2.LINE_AA)
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

    # Microsleep banner across the middle of the frame while the override is
    # up. Deliberately loud and deliberately not in the metrics strip: this is
    # the one event the whole system exists to catch.
    if m.microsleep_override:
        closed_s = float(m.microsleep_status.get("closed_s", 0.0))
        held = f" {closed_s:.1f}s" if closed_s > 0 else " (recovering)"
        banner_y = frame.shape[0] // 2
        cv2.rectangle(frame, (0, banner_y - 34), (w, banner_y + 12), (0, 0, 0), -1)
        cv2.putText(frame, f"MICROSLEEP{held}  (n={m.microsleep_count})",
                    (12, banner_y), font, 0.95, (0, 0, 255), 2, cv2.LINE_AA)


def _fill_screen(
    driver: Optional[Dict[str, Any]],
    m: FrameMetrics,
    pipeline: Optional[MetricsPipeline],
    thresholds: Optional[Dict[str, float]],
    calibrated: bool,
) -> None:
    """Copy one scored frame onto the data screen. Reads only; changes nothing."""
    s = _screen
    s.baseline_label = "baseline" if calibrated else "defaults"
    s.level = m.level
    s.frs = m.frs if m.scored else None
    s.frs_note = ("no EAR baseline" if m.seed_failed
                  else "learning EAR baseline" if not m.scored else "")
    s.ear, s.mar, s.perclos = m.ear, m.mar, m.perclos
    if m.pose is not None:
        s.yaw, s.pitch = float(m.pose["yaw"]), float(m.pose["pitch"])
    if driver:
        s.driver_name = str(driver["name"])
    s.face = FACE_OK if driver else FACE_UNRECOGNISED
    if m.microsleep_override:
        closed_s = float(m.microsleep_status.get("closed_s", 0.0))
        s.alert_banner = f"MICROSLEEP {closed_s:.1f}s" if closed_s > 0 else "MICROSLEEP"
    elif m.overridden:
        s.alert_banner = m.override_reason.upper()
    elif m.pose_override:
        s.alert_banner = "HEAD POSE"
    if pipeline is not None:
        # Display units only: the blink count is over the pipeline's window
        # (30 s in pre-drive); the calibrated baseline is per 60 s.
        s.blink_per_min = m.blink_freq * 60.0 / pipeline.blink_window_s
    if thresholds is not None:
        s.ear_base = (finite_or_none(thresholds.get("ear_baseline"))
                      or (pipeline.ear_seed.baseline if pipeline is not None else None))
        s.mar_base = finite_or_none(thresholds.get("mar_baseline"))
        s.perclos_base = finite_or_none(thresholds.get("perclos_baseline"))
        s.blink_base_per_min = finite_or_none(thresholds.get("blink_frequency_baseline"))


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
    if not _annotate_frame:
        return  # nobody sees the camera image this frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    y0 = 135
    cv2.rectangle(frame, (0, y0), (frame.shape[1], y0 + 48), (30, 30, 30), -1)

    alert_s = debounce.alert_elapsed(now)
    normal_s = debounce.normal_elapsed(now)
    if debounce.active:
        state, color = "POSE ALERT: active", (0, 0, 255)
    elif alert_s > 0:
        state, color = "POSE ALERT: arming", (0, 220, 255)
    else:
        state, color = "POSE ALERT: off", (0, 200, 0)

    # Progress bar for whichever timer is currently counting.
    if debounce.active:
        frac = min(normal_s / debounce.release_hold, 1.0) if debounce.release_hold > 0 else 1.0
        label = f"normal {normal_s:.2f}s / release at {debounce.release_hold:.2f}s"
        bar_color = (0, 200, 0)
    else:
        frac = min(alert_s / debounce.danger_hold, 1.0)
        label = f"alerting {alert_s:.2f}s / alert at {debounce.danger_hold:.2f}s"
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
    _screen.status.append((text, color))   # data screen: in call order
    if not _annotate_frame:
        return  # nobody sees the camera image this frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 1.0, 2
    tw, th = text_size(text, font, scale, thickness)
    x = max((frame.shape[1] - tw) // 2, 0)
    y = (frame.shape[0] + th) // 2 + dy
    put_text(frame, text, (x, y), font, scale, color, thickness)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def _raise_on_sigterm(signum, frame) -> None:
    """
    SIGTERM / SIGHUP handler: unwind like Ctrl-C so main()'s finally runs
    cleanup() - starter inhibited, recognition worker stopped.
    """
    name = {getattr(signal, "SIGTERM", None): "SIGTERM",
            getattr(signal, "SIGHUP", None): "SIGHUP"}.get(signum, f"signal {signum}")
    logger.warning("%s received - shutting down (starter will be inhibited)", name)
    raise KeyboardInterrupt


def start_recognition() -> None:
    """
    Start the recognition worker unless one is running. There is only ever
    one: a failed or stopped one is closed (and waited for) first. No-op
    without a factory (enrollment, tests).
    """
    global _recognition
    if _recognition_factory is None:
        return
    if _recognition is not None and not _recognition.failed and _recognition.alive:
        return
    stop_recognition("restarting")
    try:
        worker = RecognitionWorker(_recognition_factory)
        worker.start()
        _recognition = worker
    except Exception:
        logger.exception("Recognition worker could not be started - recognition runs on the "
                         "frame loop instead (blocks ~1 s per call on the Pi)")


def stop_recognition(why: str) -> None:
    """Stop the recognition worker and wait for it to exit (idempotent)."""
    global _recognition
    worker, _recognition = _recognition, None
    if worker is not None:
        logger.info("Stopping the recognition worker (%s)", why)
        worker.close()


def recognition_channel(inline: InlineRecognition) -> Any:
    """The worker if it can take requests (starting counts), else ``inline``."""
    if _recognition is not None and not _recognition.failed:
        return _recognition
    return inline


def cleanup() -> None:
    """Release the camera, inhibit the starter, reset GPIO and close any OpenCV windows."""
    global _camera, _profiler
    logger.info("Shutting down...")
    if _profiler is not None:
        # Quit ('q' / Ctrl-C / launcher STOP) mid-window: report what was timed.
        _profiler.stop()
        _profiler = None
    if _camera is not None:
        _camera.release()
        _camera = None
    _frs_debug.close()
    stop_recognition("exit")
    if _alert_manager is not None:
        _alert_manager.cleanup()
    if _ignition is not None:
        _ignition.cleanup()
    if _stream is not None:
        if _heartbeat is not None:
            _heartbeat.withdraw_stream()
        _stream.close()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # headless
    logger.info("Cleanup complete")


# ---------------------------------------------------------------------------
# Enrollment mode
# ---------------------------------------------------------------------------

def _enroll_result(code: int, outcome: str, message: str, **detail: Any) -> int:
    """
    Record how the enrollment ended (for ``--result-file``) and return ``code``.

    ``outcome`` is what the launcher's result screen keys on: ``saved``,
    ``saved_unverified``, ``dropped`` (saved without ear_closed_baseline),
    ``refused`` (nothing sent: closed-eye check / contrast / too few face
    frames), ``rejected`` (backend refused or unreachable), ``bad_driver``.
    """
    name = _enroll_report.get("driver_name")
    _enroll_report.clear()
    _enroll_report.update(exit_code=code, outcome=outcome, message=message, **detail)
    if name and "driver_name" not in detail:
        _enroll_report["driver_name"] = name
    return code


def write_enroll_result(path: str, driver_id: Optional[int], exit_code: int,
                        interrupted: bool) -> None:
    """Write the enrollment outcome as JSON (atomically) for the launcher."""
    report = dict(_enroll_report)
    if "outcome" not in report:
        report.update(
            exit_code=exit_code,
            outcome="stopped" if interrupted else "error",
            message=("Enrollment stopped before it finished - nothing was saved."
                     if interrupted else
                     "Enrollment crashed - see logs/session.log. Nothing was saved."),
        )
    report.setdefault("driver_id", driver_id)
    report["device_id"] = config.DEVICE_ID
    report["api_base_url"] = config.API_BASE_URL
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        os.replace(tmp, path)
    except OSError as exc:
        logger.error("Could not write enrollment result to %s: %s", path, exc)

Box = Tuple[int, int, int, int]       # (left, top, right, bottom), frame pixels


def rect_box(rect: Any) -> Optional[Box]:
    """A dlib rectangle as a ``Box`` (``None`` stays ``None``)."""
    if rect is None:
        return None
    return int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())


def guide_zone() -> Tuple[float, float, float, float]:
    """Where the face centre may sit, normalised (symmetric, so mirroring is moot)."""
    return (0.5 - ENROLL_CENTRE_TOL_X, 0.5 - ENROLL_CENTRE_TOL_Y,
            0.5 + ENROLL_CENTRE_TOL_X, 0.5 + ENROLL_CENTRE_TOL_Y)


def mirror_x(x: float) -> float:
    """Normalised camera-image x -> x in the driver's mirror view."""
    return x if ENROLL_CAMERA_HFLIP else 1.0 - x


def position_guidance(box: Optional[Box], frame_w: int, frame_h: int) -> Tuple[bool, str]:
    """
    Whether the face box is in position, and what to tell the driver if not.

    Size first (distance is the coarse adjustment), then left / right, then
    up / down. Left / right are the driver's own (see ``ENROLL_CAMERA_HFLIP``).
    """
    if box is None:
        return False, "Face the camera"
    left, top, right, bottom = box
    width = (right - left) / frame_w
    cx = (left + right) / 2 / frame_w
    cy = (top + bottom) / 2 / frame_h
    if width < ENROLL_FACE_MIN_W:
        return False, "Move closer to the camera"
    if width > ENROLL_FACE_MAX_W:
        return False, "Move back from the camera"
    # In the mirror view, a box left of centre is a driver sitting too far
    # to their own left.
    mx = mirror_x(cx)
    if mx < 0.5 - ENROLL_CENTRE_TOL_X:
        return False, "Move to your right"
    if mx > 0.5 + ENROLL_CENTRE_TOL_X:
        return False, "Move to your left"
    if cy < 0.5 - ENROLL_CENTRE_TOL_Y:
        return False, "Move down (sit lower or tilt the camera up)"
    if cy > 0.5 + ENROLL_CENTRE_TOL_Y:
        return False, "Move up (sit higher or tilt the camera down)"
    return True, "Good - hold still"


def enroll_view(
    frame: np.ndarray,
    step: str,
    instruction: str,
    *,
    color: tuple = (255, 255, 255),
    detail: str = "",
    note: str = "",
    box: Optional[Box] = None,
    face: Optional[bool] = None,
    box_ok: bool = False,
    outline: bool = True,
    progress: Optional[float] = None,
    progress_label: str = "",
    countdown: Optional[float] = None,
    countdown_label: str = "",
    landmarks: Optional[np.ndarray] = None,
) -> None:
    """
    One enrollment frame: fill the panel's enrollment screen (``frame`` is
    its live mirror view), and - only when the debug stream wants this frame
    (or under --show-video) - annotate the camera frame with the same
    guidance, the face box and the landmarks.

    ``face``: True face found, False no face, None not looked for this step.
    Call after every other use of ``frame``.
    """
    global _enroll
    if _rate is not None:
        _rate.tick()
    h, w = frame.shape[:2]
    norm = None
    if box is not None:
        x0, x1 = sorted((mirror_x(box[0] / w), mirror_x(box[2] / w)))
        norm = (x0, box[1] / h, x1, box[3] / h)
    if face is None:
        face_text, face_color = "CAMERA: not checked in this step", (150, 150, 150)
    elif face:
        face_text, face_color = "FACE DETECTED", GREEN
    else:
        face_text, face_color = "NO FACE - face the camera (operator: check its aim)", RED
    _enroll = EnrollState(
        driver_name=_enroll_report.get("driver_name"), step=step, instruction=instruction,
        instruction_color=color, detail=detail, note=note, face_text=face_text,
        face_color=face_color, box=norm, box_ok=box_ok,
        guide=guide_zone() if outline else None, progress=progress,
        progress_label=progress_label, countdown=countdown, countdown_label=countdown_label,
        fps=_rate.fps if _rate is not None else None,
        # The live view, by reference: render_enroll reads it only on a
        # redraw (DATA_SCREEN_HZ), so the capture loop does no extra work.
        # Copied only when the stream is about to draw on this frame below.
        image=frame.copy() if _annotate_frame else frame, mirror=not ENROLL_CAMERA_HFLIP,
    )
    if not _annotate_frame:
        return  # nobody sees the camera image this frame
    if outline:
        gx0, gy0, gx1, gy1 = guide_zone()
        cv2.rectangle(frame, (int(gx0 * w), int(gy0 * h)), (int(gx1 * w), int(gy1 * h)),
                      (120, 120, 120), 1)
    if box is not None:
        cv2.rectangle(frame, box[:2], box[2:], (0, 200, 0) if box_ok else (0, 200, 255), 2)
    if landmarks is not None:
        draw_landmarks(frame, landmarks)
    display_text(frame, instruction, color)
    if countdown is not None:
        display_text(frame, f"{countdown:.1f}s", color, dy=40)
    elif progress is not None:
        display_text(frame, f"{progress_label}", (200, 200, 200), dy=40)
    draw_banner(frame, step, (0, 200, 255))


def run_positioning(extractor: LandmarkExtractor) -> Optional[Box]:
    """
    Enrollment step 1: guide the driver until the face box has been in
    position for ``ENROLL_POSITION_HOLD_S`` (returns the last box), or give
    up after ``ENROLL_POSITION_TIMEOUT_S`` (returns ``None``).

    Uses the landmark extractor's face detection (~30 ms a frame on the Pi),
    not face_recognition (~1 s a frame), so the debug stream stays fluid
    while the operator fine-tunes the camera and seat.
    """
    step = "Step 1 of 4 - Position"
    start = time.monotonic()
    held_since: Optional[float] = None
    last_text = ""
    while True:
        now = time.monotonic()
        if now - start > ENROLL_POSITION_TIMEOUT_S:
            logger.error("Positioning: face not held in position within %.0f s",
                         ENROLL_POSITION_TIMEOUT_S)
            return None
        frame = capture_frame()
        if frame is None:
            continue
        h, w = frame.shape[:2]
        _landmarks, rect = extractor.extract(frame)
        box = rect_box(rect)
        ok, text = position_guidance(box, w, h)
        if text != last_text:
            if box is not None:
                logger.info("Positioning: %s (face width %.2f of frame, centre %.2f / %.2f)",
                            text, (box[2] - box[0]) / w, (box[0] + box[2]) / 2 / w,
                            (box[1] + box[3]) / 2 / h)
            else:
                logger.info("Positioning: no face")
            last_text = text
        held_since = (held_since or now) if ok else None
        held = now - held_since if held_since is not None else 0.0
        if held >= ENROLL_POSITION_HOLD_S:
            logger.info("Positioning: accepted after %.1f s (face width %.2f of frame)",
                        now - start, (box[2] - box[0]) / w)
            return box
        enroll_view(
            frame, step, text, color=GREEN if ok else (0, 200, 255),
            detail="Sit as you will drive and look straight at the camera - "
                   "bring the dot on your face into the dashed box.",
            box=box, face=box is not None, box_ok=ok,
            progress=held / ENROLL_POSITION_HOLD_S,
            progress_label=f"Holding position {held:.1f} / {ENROLL_POSITION_HOLD_S:.0f} s",
        )
        present(frame)


def run_enrollment(
    api_client: APIClient, extractor: LandmarkExtractor, driver_id: Optional[int] = None
) -> int:
    """
    Enrol a new driver: face encoding + 60 s alert-state calibration.

    Steps:
    1. Ask for the driver's DB id on the terminal (skipped when ``driver_id``
       is supplied, e.g. via ``--driver-id`` from the touchscreen launcher).
    1b. Positioning (:func:`run_positioning`): text guidance and a face-box
       outline on the panel until the face holds in position.
    2. Capture ``ENROLL_FRAMES`` frames containing a face and average their
       128-d encodings (averaging is more robust than a single frame).
    3. Settle for ``SEED_WINDOW_S`` to derive a personal closure threshold
       from the driver's own median EAR.
    4. Closed-eye capture (:func:`run_closed_eye_capture`): the driver shuts
       their eyes on cue, up to ``CLOSED_CAPTURE_ATTEMPTS`` tries, and the
       drop must show the landmarks follow the eyelid. Done *before* the
       alert window so the closure cannot contaminate it.
    5. Run ``CalibrationManager`` for ``config.CALIBRATION_DURATION`` s,
       feeding it EAR / blink / PERCLOS / MAR from the live pipeline, then
       re-check the contrast against the final ``ear_baseline``.
    6. Write the per-frame series to ``config.CALIBRATIONS_DIR``, POST
       encoding + baselines to the backend, and read the record back to
       confirm ``ear_closed_baseline`` was stored.

    Args:
        api_client: Connected API client.
        extractor: Landmark extractor for the calibration phase.
        driver_id: Driver's DB id. ``None`` prompts on the terminal.

    Returns:
        Process exit code: 0 success; 2 bad driver id; 3 too few face
        frames; 5 backend rejected the enrollment;
        ``EXIT_NO_EYELID_CONTRAST`` closed-eye check failed (nothing
        POSTed); ``EXIT_CLOSED_EAR_DROPPED`` saved, but the backend did not
        keep ``ear_closed_baseline``.
    """
    import face_recognition  # heavy import; only needed here

    if driver_id is None:
        raw = input("Enter driver_id to enrol: ").strip()
        try:
            driver_id = int(raw)
        except ValueError:
            logger.error("Invalid driver_id %r - must be an integer", raw)
            return _enroll_result(2, "bad_driver", f"Invalid driver id {raw!r}.")

    # The driver must exist on the backend, and the backend must answer -
    # checked before the driver sits through ~90 s of capture that could
    # only end in a failed POST. Also logs whose record this will write.
    token_state = "set" if api_client.token else "NOT SET"
    roster = api_client.get_drivers()
    if roster is None:
        logger.error("ENROLLMENT NOT STARTED: cannot read GET /drivers from %s - backend "
                     "unreachable or refusing this unit (API token %s)",
                     api_client.base_url, token_state)
        return _enroll_result(
            5, "rejected",
            f"Backend not reachable at {api_client.base_url} (API token {token_state}). "
            "Nothing was captured.")
    record = next((d for d in roster if str(d.get("id")) == str(driver_id)), None)
    if record is None:
        logger.error("ENROLLMENT NOT STARTED: no driver with id %s on %s (%d drivers listed)",
                     driver_id, api_client.base_url, len(roster))
        return _enroll_result(2, "bad_driver",
                              f"No driver with id {driver_id} on {api_client.base_url}.")
    driver_name = str(record.get("full_name") or f"Driver {driver_id}")
    _enroll_report["driver_name"] = driver_name
    logger.info("Enrolling driver %s (%s) on %s, API token %s",
                driver_id, driver_name, api_client.base_url, token_state)

    # ---- 1. Positioning ---------------------------------------------------
    logger.info("Positioning: guiding the driver into frame (hold %.0f s, timeout %.0f s)",
                ENROLL_POSITION_HOLD_S, ENROLL_POSITION_TIMEOUT_S)
    if run_positioning(extractor) is None:
        return _enroll_result(
            3, "refused", f"The face did not stay in position for {ENROLL_POSITION_HOLD_S:.0f} s "
            f"within {ENROLL_POSITION_TIMEOUT_S:.0f} s - check camera aim and seat position "
            "(video stream). Nothing was captured.")

    # ---- 2. Face encoding ------------------------------------------------
    logger.info("Look at the camera. Capturing %d frames for the face encoding...", ENROLL_FRAMES)
    step = "Step 2 of 4 - Face capture"
    encodings = []
    attempts = 0
    while len(encodings) < ENROLL_FRAMES and attempts < ENROLL_FRAMES * 10:
        attempts += 1
        frame = capture_frame()
        if frame is None:
            continue
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        locations = face_recognition.face_locations(rgb, model="hog")
        progress = len(encodings) / ENROLL_FRAMES
        label = f"Face capture {len(encodings)} / {ENROLL_FRAMES}"
        if not locations:
            enroll_view(frame, step, "Face the camera", color=(0, 200, 255),
                        detail="The face was lost - look straight at the camera.",
                        face=False, progress=progress, progress_label=label)
            present(frame)
            continue
        # Use only the largest face in case someone is visible in the background.
        largest = max(locations, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
        enc = face_recognition.face_encodings(rgb, [largest])
        if enc:
            encodings.append(enc[0])
        top, right, bottom, left = largest
        enroll_view(frame, step, "Look at the camera and hold still",
                    detail="Capturing the face for recognition.",
                    box=(left, top, right, bottom), face=True, box_ok=True,
                    progress=len(encodings) / ENROLL_FRAMES,
                    progress_label=f"Face capture {len(encodings)} / {ENROLL_FRAMES}")
        present(frame)

    if len(encodings) < ENROLL_FRAMES // 2:
        logger.error("Only %d usable face frames captured - aborting enrollment", len(encodings))
        return _enroll_result(
            3, "refused", f"Only {len(encodings)} of {ENROLL_FRAMES} face frames captured - "
            "check camera aim and lighting. Nothing was saved.")
    face_encoding = np.mean(np.stack(encodings), axis=0)
    logger.info("Face encoding computed from %d frames", len(encodings))

    # ---- 3. Calibration ---------------------------------------------------
    ear_calc = EARCalculator()
    mar_calc = MARCalculator()
    blink_detector = BlinkDetector(frequency_window=60)
    perclos_calc = PERCLOSCalculator()
    microsleep_detector = MicrosleepDetector()
    calib = CalibrationManager()
    # Neutral head pose = median over the calibration window (2026-10-15).
    head_pose = _head_pose or HeadPoseEstimator(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    calib_poses: List[Dict[str, Any]] = []

    # No personal threshold exists yet. Rather than seeding one from
    # DEFAULT_THRESHOLDS (0.225 is 75 % of a 0.30 EAR but 88 % of a 0.255 one -
    # inside landmark noise, so it manufactures false closures), observe the
    # driver for SEED_WINDOW_S of elapsed (monotonic) time *before* calibration starts
    # and derive the threshold from their own median EAR. Nothing is fed to
    # the blink detector, PERCLOS or the calibration until then, so the
    # calibration window is pure detection time under a personal threshold.
    ear_history: list = []
    seed_start = time.monotonic()
    logger.info("Settling for %.0f s to derive a personal closure threshold...", SEED_WINDOW_S)
    while time.monotonic() - seed_start < SEED_WINDOW_S or len(ear_history) < SEED_MIN_SAMPLES:
        frame = capture_frame()
        if frame is None:
            continue
        landmarks, rect = extractor.extract(frame)
        if landmarks is None:
            enroll_view(frame, "Step 3 of 4 - Closed eyes", "Face the camera",
                        color=(0, 200, 255), face=False)
            present(frame)
            continue
        ear_history.append(ear_calc.compute_average_ear(landmarks))
        enroll_view(frame, "Step 3 of 4 - Closed eyes", "Look at the road, eyes open",
                    detail="Measuring your open eyes first.", box=rect_box(rect), face=True,
                    box_ok=True, landmarks=landmarks)
        present(frame)
    open_reference = float(np.median(ear_history))
    threshold = open_reference * EAR_THRESHOLD_RATIO
    logger.info("Seed threshold %.4f from %d frames (median EAR %.4f)",
                threshold, len(ear_history), open_reference)

    # Closed-eye capture, before the alert window (see run_closed_eye_capture).
    closed_capture, closed_attempts = run_closed_eye_capture(extractor, ear_calc, open_reference)
    if closed_capture is None:
        logger.error("ENROLLMENT ABORTED for driver %s: the closed-eye check failed %d times - "
                     "nothing was saved. If the driver did close their eyes, the landmarks are "
                     "not tracking their eyelids: check lighting and camera position, and run "
                     "tools/eye_check.py to see where the eye landmarks sit.",
                     driver_id, len(closed_attempts))
        reasons = "; ".join(f"#{a['attempt']}: {a['reason']}" for a in closed_attempts)
        return _enroll_result(
            EXIT_NO_EYELID_CONTRAST, "refused",
            f"Closed-eye check failed {len(closed_attempts)} times - nothing was saved. "
            "Either the eyes were not shut, or the landmarks do not follow the eyelids "
            "(check lighting / camera; tools/eye_check.py).", reasons=reasons)
    settle_start = time.monotonic()
    while time.monotonic() - settle_start < REOPEN_SETTLE_S:
        frame = capture_frame()
        if frame is None:
            continue
        enroll_view(frame, "Step 3 of 4 - Closed eyes", "OPEN YOUR EYES", color=GREEN,
                    detail="Closed-eye check passed. Look at the road.", outline=False,
                    countdown=REOPEN_SETTLE_S - (time.monotonic() - settle_start),
                    countdown_label="calibration starts in")
        present(frame)

    logger.info("Starting %d s calibration - stay alert, look at the road and keep "
                "your mouth relaxed (talking inflates the MAR baseline).",
                config.CALIBRATION_DURATION)
    calib.start()
    step = "Step 4 of 4 - Calibration"
    calib_detail = ("Stay alert, look at the road as when driving, keep your mouth relaxed "
                    "and do not talk.")
    status = {"progress": 0.0, "seconds_remaining": config.CALIBRATION_DURATION}
    while calib.is_calibrating:
        frame = capture_frame()
        if frame is None:
            continue
        landmarks, rect = extractor.extract(frame)
        # Invalid landmarks are skipped like a missing face: an EAR that is
        # not a measurement must not enter the baselines.
        if landmark_problem(landmarks, frame.shape) is not None:
            blink_detector.abandon()
            microsleep_detector.note_no_face(time.monotonic())
            enroll_view(frame, step, "Face the camera", color=(0, 200, 255), detail=calib_detail,
                        face=False, progress=status["progress"],
                        progress_label=f"Calibration - {status['seconds_remaining']} s left")
            present(frame)
            continue

        ear = ear_calc.compute_average_ear(landmarks)
        mar = mar_calc.compute_mar(landmarks)
        pose = head_pose.estimate(landmarks)
        if pose is not None:
            calib_poses.append(pose)
        # Keep tightening to the running median as more of the driver is seen.
        ear_history.append(ear)
        threshold = float(np.median(ear_history)) * EAR_THRESHOLD_RATIO

        now = time.monotonic()
        eye_closed = ear < threshold  # the test both detectors below apply
        event = blink_detector.update(ear, threshold, now)
        # Same EAR and threshold: closures >= 1 s are microsleeps and are
        # rejected by the blink detector, so they must be collected here or
        # they would vanish from the calibration record entirely.
        ms_event = microsleep_detector.update(ear, threshold, now)
        # No warm-up prior: the baseline is what is being measured.
        perclos = perclos_calc.update(ear, threshold, now)
        status = calib.update(
            ear,
            event["duration_ms"] if event else None,
            blink_detector.get_blink_frequency(now),
            perclos,
            timestamp=now,
            mar=mar,
            eye_closed=eye_closed,
            threshold=threshold,
            microsleep_ms=ms_event["duration_ms"] if ms_event else None,
        )

        enroll_view(frame, step, "Look at the road", detail=calib_detail, box=rect_box(rect),
                    face=True, box_ok=True, landmarks=landmarks, progress=status["progress"],
                    progress_label=f"Calibration - {status['seconds_remaining']} s left")
        present(frame)

    baselines = calib.compute_baselines()
    baselines["ear_closed_baseline"] = float(closed_capture.closed_ear)
    logger.info("Calibration baselines: %s", baselines)
    # Kept out of ``baselines``, which is POSTed as-is: the backend has no
    # field for it, so it is stored on this unit (modules/pose_neutral.py).
    pose_neutral = median_pose(calib_poses)
    logger.info("Calibration neutral head pose: %s", pose_neutral)
    # Pre-floor measurements and counts, so a baseline that landed on a
    # MIN_* floor (modules/calibration.py) can be traced to its cause.
    logger.info("Calibration raw: %s", calib.raw_summary())
    # The pre-check used the 1 s settle median; the stored contrast is
    # against the 60 s ear_baseline, and that is what load_calibration judges.
    problem = ear_calibration_problem(baselines["ear_baseline"],
                                      baselines["ear_closed_baseline"])
    # Per-frame series + summary on disk, written before the backend call so
    # a rejected enrollment still leaves the evidence behind.
    try:
        calib_csv = calib.write_files(config.DEVICE_ID, driver_id, baselines, closed_capture={
            "attempts": closed_attempts,
            "accepted": closed_capture.summary(baselines["ear_baseline"]),
            "problem": problem,
        }, pose_neutral=pose_neutral)
    except OSError as exc:
        calib_csv = None
        logger.error("Could not write calibration data: %s", exc)
    if problem is not None:
        logger.error("ENROLLMENT REFUSED for driver %s: %s. Nothing was sent to the backend; "
                     "re-run the enrollment.", driver_id, problem)
        return _enroll_result(EXIT_NO_EYELID_CONTRAST, "refused",
                              "Calibration refused - nothing was sent to the backend. "
                              "Re-run the enrollment.", reasons=str(problem))
    logger.info("Closed-eye contrast %.2f (closed %.4f / ear_baseline %.4f)",
                baselines["ear_closed_baseline"] / baselines["ear_baseline"],
                baselines["ear_closed_baseline"], baselines["ear_baseline"])

    # ---- 4. Persist ---------------------------------------------------------
    ok = api_client.save_driver_enrollment(driver_id, face_encoding.tolist(), baselines)
    if not ok:
        logger.error("Backend rejected enrollment for driver %s", driver_id)
        return _enroll_result(5, "rejected", "The backend did not save the enrollment.",
                              reasons=api_client.last_enroll_error or "")
    # Stored only once the calibration it belongs to has been saved.
    if pose_neutral is not None:
        save_neutral(driver_id, pose_neutral, calib_csv.stem if calib_csv is not None else "")
    else:
        logger.warning("No neutral head pose for driver %s (too few pose estimates) - pre-drive "
                       "will use the settling median", driver_id)
    # A Laravel FormRequest drops fields it does not validate, silently: the
    # record would come back without ear_closed_baseline and be treated as a
    # pre-2026-09-30 record (no contrast check). Read it back to be sure.
    stored = api_client.get_driver_calibration(driver_id)
    contrast = round(baselines["ear_closed_baseline"] / baselines["ear_baseline"], 2)
    numbers = {k: round(float(baselines[k]), 4) for k in
               ("ear_baseline", "ear_closed_baseline", "mar_baseline") if k in baselines}
    if stored is None:
        logger.warning("Enrollment saved, but the calibration could not be read back to confirm "
                       "ear_closed_baseline was stored")
        return _enroll_result(
            0, "saved_unverified", "Saved, but the record could not be read back to confirm "
            "the closed-eye baseline was stored.", contrast=contrast, baselines=numbers)
    if "ear_closed_baseline" not in stored.thresholds:
        logger.error("Enrollment saved, but the backend DROPPED ear_closed_baseline: this "
                     "driver's record has no contrast check and will be treated as a legacy "
                     "record. Deploy the backend change (deploy/BACKEND_CHANGES_2026-09-28.md, "
                     "item 6) and re-enrol.")
        return _enroll_result(
            EXIT_CLOSED_EAR_DROPPED, "dropped",
            "Saved, but the backend DROPPED the closed-eye baseline: this record has no "
            "contrast check. Deploy backend item 6 (deploy/BACKEND_CHANGES_2026-09-28.md) "
            "and re-enrol.", baselines=numbers)

    msg = f"Enrollment complete for driver {driver_id}"
    logger.info(msg)
    print(msg)
    return _enroll_result(0, "saved", f"Enrollment saved for {driver_name}.",
                          contrast=contrast, baselines=numbers)


def run_closed_eye_capture(
    extractor: LandmarkExtractor, ear_calc: EARCalculator, open_reference: float
) -> Tuple[Optional[ClosedEyeCapture], List[Dict[str, Any]]]:
    """
    Enrollment step: measure the driver's EAR with the eyes deliberately shut.

    Each attempt shows a ``CLOSED_PROMPT_LEAD_S`` countdown, beeps once
    (close now), captures for ``CLOSED_CAPTURE_S`` and beeps long (open) -
    the driver cannot read the screen with their eyes shut, so the operator
    should also say "open". An attempt passes when closed / open is at most
    ``EAR_CONTRAST_MAX`` against ``open_reference`` (the settle median); a
    failure is logged with its reason and retried, up to
    ``CLOSED_CAPTURE_ATTEMPTS`` in all. A driver who did not close their eyes
    and landmarks that do not follow the eyelid look the same in the
    numbers, so the reason names both and the operator decides.

    Returns:
        ``(capture, attempts)``: the passing :class:`ClosedEyeCapture` (or
        ``None`` if every attempt failed) and one summary dict per attempt.
    """
    attempts: List[Dict[str, Any]] = []

    def beep(pattern: str) -> None:
        # trigger_buzzer blocks for the beep; keep the capture loop running.
        if _alert_manager is not None:
            threading.Thread(target=_alert_manager.trigger_buzzer, args=(pattern,),
                             daemon=True).start()

    step = "Step 3 of 4 - Closed eyes"
    note = ""
    for attempt in range(1, CLOSED_CAPTURE_ATTEMPTS + 1):
        capture = ClosedEyeCapture()
        logger.info("Closed-eye capture %d/%d: tell the driver to close their eyes on the beep "
                    "and keep them shut until the long beep (%.0f s)",
                    attempt, CLOSED_CAPTURE_ATTEMPTS, capture.duration_s)
        lead_start = time.monotonic()
        while time.monotonic() - lead_start < CLOSED_PROMPT_LEAD_S:
            frame = capture_frame()
            if frame is None:
                continue
            remaining = CLOSED_PROMPT_LEAD_S - (time.monotonic() - lead_start)
            enroll_view(frame, f"{step} (attempt {attempt} of {CLOSED_CAPTURE_ATTEMPTS})",
                        "CLOSE YOUR EYES on the beep", color=(0, 200, 255),
                        detail=f"Keep them shut until the long beep (about "
                               f"{capture.duration_s:.0f} s). Operator: say \"open\" "
                               "at the long beep.",
                        note=note, outline=False, countdown=remaining,
                        countdown_label="beep in")
            present(frame)

        capture.start(time.monotonic())
        beep("short")
        while not capture.done(time.monotonic()):
            frame = capture_frame()
            if frame is None:
                continue
            landmarks, rect = extractor.extract(frame)
            if landmarks is not None:
                capture.update(ear_calc.compute_average_ear(landmarks), time.monotonic())
            enroll_view(frame, f"{step} (attempt {attempt} of {CLOSED_CAPTURE_ATTEMPTS})",
                        "EYES CLOSED - keep them shut", color=(0, 0, 255),
                        detail="" if landmarks is not None else
                        "Face lost - keep facing the camera, eyes shut.",
                        face=landmarks is not None, outline=False,
                        countdown=max(0.0, capture.duration_s - capture.elapsed(time.monotonic())),
                        countdown_label="until the long beep", box=rect_box(rect),
                        landmarks=landmarks)
            present(frame)
        beep("long")

        ok, ratio, reason = capture.evaluate(open_reference)
        record = dict(capture.summary(open_reference), attempt=attempt, ok=ok,
                      reason=reason or None)
        attempts.append(record)
        if ok:
            logger.info("Closed-eye capture %d passed: closed EAR %.4f / open %.4f = %.2f "
                        "(%d frames measured)", attempt, capture.closed_ear, open_reference,
                        ratio, len(capture.measured))
            return capture, attempts
        logger.warning("Closed-eye capture %d/%d FAILED: %s (closed EAR %s, open %.4f)",
                       attempt, CLOSED_CAPTURE_ATTEMPTS, reason,
                       "n/a" if capture.closed_ear is None else f"{capture.closed_ear:.4f}",
                       open_reference)
        note = f"Attempt {attempt} did not pass: {reason}. Close the eyes fully this time."
    return None, attempts


# ---------------------------------------------------------------------------
# Shared helpers for the two operating phases
# ---------------------------------------------------------------------------

def check_phase(ignition: Any, expected: Phase) -> None:
    """Raise :class:`PhaseChanged` if the ignition no longer implies ``expected``."""
    if ignition.phase() is not expected:
        raise PhaseChanged()


def load_calibration(
    api_client: APIClient, driver_id: int, phase: Phase
) -> Tuple[Optional[Calibration], Optional[Dict[str, Any]]]:
    """
    Fetch the recognised driver's calibration, and decide whether it is usable.

    Keyed by driver and device (``APIClient.get_driver_calibration``): the
    driver's calibration captured on this unit, else - flagged
    ``foreign_device`` - one captured on another unit. Never another
    driver's. (Pre-drive locks on ``foreign_device`` itself; see
    :func:`run_predrive_assessment`.)

    **Pre-drive and monitoring treat a broken record differently, on
    purpose.** Pre-drive can release the starter, so it is fail-secure: any
    defect - an EAR key or core baseline missing, a threshold inconsistent
    with its baseline, a zero PERCLOS baseline - returns ``None`` and the unit
    locks with ``NO_BASELINE``. Monitoring cannot release anything and a
    driver's real alert-state baseline is worth keeping, so there a broken
    record is *repaired* - but only where its EAR baseline is independently
    credible (``ear_calibration_problem``):

    * threshold missing / inconsistent -> recomputed as
      ``ear_baseline * EAR_THRESHOLD_RATIO``;
    * EAR baseline missing -> the EAR pair is self-seeded, the record's other
      baselines are kept;
    * PERCLOS baseline <= 0 -> floored at calibration's MIN_PERCLOS_BASELINE;
    * another core baseline missing -> filled from DEFAULT_THRESHOLDS.

    An EAR baseline *present but not credible* is never repaired in either
    phase: it is the suspect value, and a threshold recomputed from it would
    launder the corruption. The record is discarded (monitoring then
    self-seeds everything). Credible means (2026-09-30): with an
    ``ear_closed_baseline``, closed / open at most ``EAR_CONTRAST_MAX`` and
    the baseline inside the degenerate-value backstop; without one (records
    enrolled before the closed-eye capture), grandfathered if the baseline
    is inside ``EAR_BASELINE_UNCONTRASTED_MIN`` - ``EAR_BASELINE_MAX``. Every repair is logged at WARNING with original
    and new values and listed in ``Calibration.repairs``, which travels as
    ``calibration_repairs`` on every fatigue event. Frequent repairs point at
    a problem in the enrollment path.

    Args:
        api_client: Backend client.
        driver_id: Recognised driver.
        phase: ``Phase.PREDRIVE`` (no repairs) or ``Phase.MONITORING``.

    Returns:
        ``(calibration, provenance)``. ``calibration`` is usable - every
        pipeline key present except, when ``self_seed_ear``, the EAR pair -
        or ``None``. ``provenance`` describes the record the backend
        returned *whether or not it was used* (``None`` only if there was
        none), so a lock on a rejected record still shows the operator which
        calibration was refused and where it was captured.
    """
    fetched = api_client.get_driver_calibration(driver_id)
    if fetched is None:
        return None, None
    thresholds = dict(fetched.thresholds)
    source = f"backend, driver {driver_id} ({fetched.source}, calibration {fetched.calibration_id})"
    monitoring = Phase(phase) is Phase.MONITORING

    baseline = thresholds.get("ear_baseline")
    closed = thresholds.get("ear_closed_baseline")
    if baseline is not None:
        problem = ear_calibration_problem(baseline, closed)
        if problem is not None:
            logger.error("CALIBRATION DISCARDED (%s): %s - the baseline itself is suspect, so "
                         "the record is not repaired%s", source, problem,
                         "; self-seeding instead" if monitoring else "; locking")
            return None, fetched.provenance()
        if closed is None:
            logger.info("Calibration (%s) predates the closed-eye capture: no contrast check "
                        "possible, grandfathered on its ear_baseline %.4f", source, baseline)
        else:
            logger.info("Calibration (%s): closed/open EAR %.4f/%.4f = %.2f", source, closed,
                        baseline, closed / baseline)

    # Drivers enrolled before yawn detection have no MAR baseline; say so
    # explicitly because the generic default silently makes yawn detection
    # non-personal. Allowed in both phases (a known enrollment vintage, not
    # a broken record).
    missing_mar = [k for k in MAR_THRESHOLD_KEYS if k not in thresholds]
    if missing_mar:
        logger.warning("Driver %s has no MAR baseline (%s missing) - yawn detection will "
                       "use generic defaults; re-enrol with `main.py --enroll` to calibrate it",
                       driver_id, ", ".join(missing_mar))
        for key in missing_mar:
            thresholds[key] = DEFAULT_THRESHOLDS[key]
    missing_core = [k for k in DEFAULT_THRESHOLDS if k not in thresholds]

    if not monitoring:
        # Pre-drive: fail-secure, no repairs.
        missing = [k for k in ("ear_baseline", "ear_threshold") if k not in thresholds]
        if missing + missing_core:
            logger.error("Calibration unusable for pre-drive (%s): missing %s",
                         source, ", ".join(missing + missing_core))
            return None, fetched.provenance()
        if not check_threshold_sanity(thresholds, source):
            return None, fetched.provenance()
        usable = Calibration(thresholds, fetched.driver_id, fetched.source,
                             fetched.calibration_id, fetched.captured_on_device_id)
        return usable, usable.provenance()

    # Monitoring-only repairs (see docstring).
    repairs = []
    for key in missing_core:
        thresholds[key] = DEFAULT_THRESHOLDS[key]
        repairs.append(repair_defaulted(key))
        logger.warning("CALIBRATION REPAIRED (%s): %s missing -> default %.3f",
                       source, key, thresholds[key])
    if baseline is None:
        thresholds.pop("ear_threshold", None)
        repairs.append(REPAIR_EAR_PAIR_SELF_SEEDED)
        logger.warning("CALIBRATION REPAIRED (%s): no ear_baseline -> EAR pair self-seeded, "
                       "other baselines kept", source)
    else:
        original = thresholds.get("ear_threshold")
        if original is None or abs(original / baseline - EAR_THRESHOLD_RATIO) > EAR_RATIO_TOLERANCE:
            thresholds["ear_threshold"] = baseline * EAR_THRESHOLD_RATIO
            repairs.append(REPAIR_EAR_THRESHOLD_RECOMPUTED)
            logger.warning("CALIBRATION REPAIRED (%s): ear_threshold %s (ratio %s) -> %.4f "
                           "(%.2f x ear_baseline %.4f)", source,
                           "missing" if original is None else f"{original:.4f}",
                           "-" if original is None else f"{original / baseline:.3f}",
                           thresholds["ear_threshold"], EAR_THRESHOLD_RATIO, baseline)
    if thresholds["perclos_baseline"] <= 0.0:
        original = thresholds["perclos_baseline"]
        thresholds["perclos_baseline"] = MIN_PERCLOS_BASELINE
        repairs.append(REPAIR_PERCLOS_BASELINE_FLOORED)
        logger.warning("CALIBRATION REPAIRED (%s): perclos_baseline %.3f -> %.3f (calibration "
                       "floor)", source, original, MIN_PERCLOS_BASELINE)
    if baseline is not None:
        check_threshold_sanity(thresholds, source)   # logs the (now consistent) pair
    usable = Calibration(thresholds, fetched.driver_id, fetched.source,
                         fetched.calibration_id, fetched.captured_on_device_id, tuple(repairs))
    return usable, usable.provenance()


class CalibrationFetch:
    """
    :func:`load_calibration` (monitoring rules) on a background thread, so
    the frame loop never waits for the backend (2026-10-19). Unlike dlib, an
    HTTP wait releases the GIL, so a thread is enough. The thread gets its
    own client (``APIClient.clone``: a ``requests.Session`` is not
    thread-safe) and is a daemon: an answer that never comes is simply never
    taken.
    """

    def __init__(self, api_client: Any, driver_id: int) -> None:
        self.driver_id = int(driver_id)
        self.started = time.monotonic()
        self._result: Optional[Tuple[Optional[Calibration], Optional[Dict[str, Any]]]] = None
        self._done = threading.Event()
        clone = getattr(type(api_client), "clone", None)
        client = api_client.clone() if callable(clone) else api_client
        threading.Thread(target=self._run, args=(client,), name="calibration-fetch",
                         daemon=True).start()

    def _run(self, client: Any) -> None:
        try:
            self._result = load_calibration(client, self.driver_id, Phase.MONITORING)
        except Exception:
            logger.exception("Calibration fetch for driver %s failed", self.driver_id)
            self._result = (None, None)
        finally:
            self._done.set()

    def poll(self) -> Optional[Tuple[Optional[Calibration], Optional[Dict[str, Any]]]]:
        """``load_calibration``'s result once it is in, else ``None``. Never blocks."""
        return self._result if self._done.is_set() else None


def check_threshold_sanity(thresholds: Dict[str, float], source: str) -> bool:
    """
    Check that ``ear_threshold`` is consistent with ``ear_baseline``.

    Everything that decides "are the eyes shut" -
    PERCLOS, the blink detector and the microsleep detector - tests
    ``ear < thresholds["ear_threshold"]``, so a threshold that does not match
    the baseline it was derived from silently disables all three at once:
    PERCLOS stays at 0 %, no blink ever completes, and no closure is ever
    long enough to confirm a microsleep. The FRS then rides on the EAR term
    alone, which still responds to closure because it divides by the
    baseline rather than comparing against the threshold.

    Calibration sets ``ear_threshold = ear_baseline * EAR_THRESHOLD_RATIO``
    (0.75), so the ratio below should read ~0.75. Anything much lower means
    the two values came from different places - a stale threshold against a
    re-enrolled baseline, or a partial backend record.

    Until 2026-09-28 this only logged, and monitoring then ran on the record
    anyway; now a failing record is not used (pre-drive locks, monitoring
    self-seeds).

    Args:
        thresholds: The dict about to be handed to the pipeline.
        source: Where it came from, for the log line.

    Returns:
        ``True`` if the record is usable.
    """
    # finite_or_none: a key that is present but None / non-numeric reads as
    # 0 here (unusable) instead of raising.
    def num(key: str) -> float:
        return finite_or_none(thresholds.get(key)) or 0.0

    baseline = num("ear_baseline")
    threshold = num("ear_threshold")
    if baseline <= 0.0:
        logger.error("THRESHOLD CHECK (%s): ear_baseline is %.4f - EAR normalisation "
                     "and the closure test are both meaningless", source, baseline)
        return False
    ratio = threshold / baseline
    usable = True
    logger.info(
        "THRESHOLD CHECK (%s): ear_baseline=%.4f ear_threshold=%.4f ratio=%.3f "
        "(expected %.2f) | perclos_baseline=%.3f blink_duration_baseline=%.1fms "
        "blink_frequency_baseline=%.2f",
        source, baseline, threshold, ratio, EAR_THRESHOLD_RATIO,
        num("perclos_baseline"), num("blink_duration_baseline"),
        num("blink_frequency_baseline"),
    )
    if ratio < EAR_THRESHOLD_RATIO - EAR_RATIO_TOLERANCE:
        logger.error(
            "THRESHOLD TOO LOW (%s): ear_threshold is %.1f%% of ear_baseline, expected "
            "%.0f%%. The eye must drop to %.1f%% of its open value (below %.4f) before "
            "anything counts as closed. Real closures often only reach 40-50%% of "
            "baseline, so PERCLOS, blink detection and microsleep detection may ALL "
            "report nothing while the driver's eyes are visibly shut. Re-enrol this "
            "driver, or check the backend is not serving a stale threshold against a "
            "newer baseline.",
            source, ratio * 100, EAR_THRESHOLD_RATIO * 100, ratio * 100, threshold,
        )
        usable = False
    elif ratio > EAR_THRESHOLD_RATIO + EAR_RATIO_TOLERANCE:
        logger.error(
            "THRESHOLD TOO HIGH (%s): ear_threshold is %.1f%% of ear_baseline, expected "
            "%.0f%%. Partly-open eyes will register as closed, inflating PERCLOS and "
            "manufacturing blinks. Re-enrol this driver.",
            source, ratio * 100, EAR_THRESHOLD_RATIO * 100,
        )
        usable = False
    if num("perclos_baseline") <= 0.0:
        logger.error(
            "THRESHOLD CHECK (%s): perclos_baseline is 0 - PERCLOSCalculator.normalize() "
            "returns 0.0 for a zero baseline, so the PERCLOS term is dead regardless of "
            "how long the eyes are shut", source)
        usable = False
    if not usable:
        logger.error("THRESHOLD CHECK (%s): record NOT USED", source)
    return usable


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
        faces = recognizer.last_face_count
        if faces is not None:
            _screen.face = FACE_NONE if faces == 0 else FACE_SEARCHING
            _screen.face_detail = "" if faces == 0 else (
                f"{counts[int(match['driver_id'])]}/{matches_required}" if match
                else "- no match yet")
        _screen.countdown = max(0.0, deadline - time.monotonic())
        display_text(frame, f"Recognizing... {seen}/{matches_required}", WHITE)
        display_text(frame, f"{max(0.0, deadline - time.monotonic()):.0f}s left", GREY, dy=40)
        draw_banner(frame, "PRE-DRIVE  |  starter LOCKED  |  identifying driver", (0, 220, 255))
        present(frame)
    logger.warning("Pre-drive: driver not recognised within %.0fs (matches seen: %s)",
                   timeout_s, counts or "none")
    return None


def unscored_reason(problem: str) -> str:
    """Debug-CSV reason for a frame :func:`landmark_problem` rejected."""
    return "NO_FACE" if problem == "no_face" else f"INVALID_LANDMARKS:{problem}"


def run_settling(
    extractor: LandmarkExtractor,
    head_pose: HeadPoseEstimator,
    ignition: Any,
    rate: LoopRate,
    ear_threshold: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Pre-drive SETTLING (2026-10-15): between recognition and the assessment,
    the face and landmarks must be valid for ``config.PREDRIVE_SETTLE_SECONDS``
    continuously. A lost face or invalid landmarks restart the period. The
    head poses seen here give the fallback neutral pose for a driver with
    none stored.

    Nothing is scored, except that microsleep detection runs throughout
    (:class:`~modules.pipeline.SettlingWatch`, against ``ear_threshold``): a
    confirmed microsleep ends settling at once and fails the pre-drive.

    Returns:
        ``{"neutral": median_pose(...) or None, "frames": int,
        "microsleep": event or None}`` once settled or on a microsleep, or
        ``None`` if it did not settle within
        ``config.PREDRIVE_SETTLE_TIMEOUT_S`` (an invalid assessment).

    Raises:
        PhaseChanged: if the ignition turns ON meanwhile.
    """
    settle_s = config.PREDRIVE_SETTLE_SECONDS
    deadline = time.monotonic() + config.PREDRIVE_SETTLE_TIMEOUT_S
    watch = SettlingWatch(settle_s, ear_threshold)
    logger.info("Pre-drive: settling - face and landmarks must stay valid for %.1fs "
                "(timeout %.0fs); microsleep detection active", settle_s,
                config.PREDRIVE_SETTLE_TIMEOUT_S)
    while time.monotonic() < deadline:
        check_phase(ignition, Phase.PREDRIVE)
        frame = next_frame()
        rate.tick()
        now, now_mono = time.time(), time.monotonic()
        landmarks, _rect = extractor.extract(frame)
        problem = landmark_problem(landmarks, frame.shape)
        banner = "PRE-DRIVE  |  starter LOCKED  |  settling"
        if problem is not None:
            held = watch.held(now_mono)
            watch.invalid(now_mono)
            if held > 0:
                logger.info("Settling restarted after %.1fs: %s", held, problem)
            _frs_debug.log_unscored(now, "predrive_settling", False, "ALERT",
                                    unscored_reason(problem))
            _screen.face, _screen.countdown = FACE_NONE, settle_s
            display_text(frame, "Hold still...", (0, 220, 255))
            display_text(frame, "Face the camera" if problem == "no_face"
                         else "Keep your whole face in view", GREY, dy=40)
            draw_banner(frame, banner, (0, 220, 255))
            present(frame)
            continue
        pose = head_pose.estimate(landmarks)
        event = watch.valid(landmarks, now, now_mono, pose)
        if event is not None:
            logger.warning("MICROSLEEP during settling: eyes closed %.2fs (min EAR %.3f < "
                           "threshold %.3f) - pre-drive fails as microsleep_detected",
                           event["duration_ms"] / 1000.0, event["min_ear"], ear_threshold)
            _frs_debug.log_unscored(now, "predrive_settling", True, "DANGER",
                                    "MICROSLEEP", pose)
            return {"neutral": None, "frames": len(watch.poses), "microsleep": event}
        _frs_debug.log_unscored(now, "predrive_settling", True, "ALERT",
                                "SETTLING_CLOSED" if watch.closure_in_progress else "SETTLING",
                                pose)
        held = watch.held(now_mono)
        _screen.face, _screen.countdown = FACE_OK, max(0.0, settle_s - held)
        display_text(frame, "Hold still...", (0, 220, 255))
        display_text(frame, f"{max(0.0, settle_s - held):.1f}s", GREY, dy=40)
        draw_banner(frame, banner, (0, 220, 255))
        present(frame)
        if watch.settled(now_mono):
            neutral = median_pose(watch.poses)
            logger.info("Pre-drive: settled after %.1fs valid (%d restarts, %d poses; settle "
                        "neutral %s)", held, watch.restarts, len(watch.poses),
                        "none" if neutral is None else
                        "pitch %.1f yaw %.1f roll %.1f" % (neutral["pitch"], neutral["yaw"],
                                                           neutral["roll"]))
            return {"neutral": neutral, "frames": len(watch.poses), "microsleep": None}
    logger.warning("Pre-drive: did not settle within %.0fs (%d restarts) - face/landmarks "
                   "never valid for %.1fs", config.PREDRIVE_SETTLE_TIMEOUT_S, watch.restarts,
                   settle_s)
    return None


def show_release(rate: LoopRate, message: str) -> None:
    """
    Pre-drive -> monitoring hand-off: show ``message`` in green on live
    frames for :data:`RELEASE_INDICATOR_S`. Called after the start was
    confirmed; the starter stays released and monitoring follows.
    """
    logger.info("Pre-drive: %s - starter released, monitoring starts in %.1fs",
                message, RELEASE_INDICATOR_S)
    deadline = time.monotonic() + RELEASE_INDICATOR_S
    while time.monotonic() < deadline:
        frame = next_frame()
        rate.tick()
        display_text(frame, message, (0, 200, 0))
        display_text(frame, "starter released - monitoring starting", GREY, dy=40)
        relay = _alert_manager.get_relay_state() if _alert_manager else "n/a"
        draw_banner(frame, f"PRE-DRIVE  |  starter {relay}", (0, 200, 0))
        present(frame)


def log_start_confirmed(source: str, cycle_started: float, now: float,
                        assessment: Optional[PredriveAssessment] = None) -> None:
    """
    The start confirmation, logged apart from READY_TO_START entry (Chapter 4
    timing). ``now`` and ``cycle_started`` are ``time.monotonic()``.
    """
    duration = now - cycle_started
    logger.info("START CONFIRMED by %s at %s - pre-drive duration %.1fs (from recognition "
                "to confirmation)", source, _stamp(time.time()), duration)
    _frs_debug.log_event(time.time(), "predrive_confirmed", "ALERT",
                         "START_CONFIRMED by %s predrive_duration=%.1fs" % (source, duration))
    if assessment is not None:
        assessment.mark("confirmed", now, source=source)


def wait_for_start(
    rate: LoopRate,
    message: str,
    cycle_started: float,
    assessment: Optional[PredriveAssessment] = None,
) -> bool:
    """
    ``cycle_started``: ``time.monotonic()`` the pre-drive cycle began.

    READY_TO_START without a rolling assessment behind it - an operator
    override was approved and the starter released. Waits indefinitely for
    the start confirmation (``config.START_CONFIRM_MODE``); there is no
    timeout into monitoring.

    Returns:
        ``True`` once the start is confirmed, ``False`` if the user pressed
        ``r`` (re-run the assessment; the next cycle re-inhibits).
    """
    _start_confirm.reset("new READY_TO_START")
    set_session_state(SessionState.READY_TO_START, message)
    now = time.monotonic()
    logger.info("READY_TO_START at %s (%s) - %.1fs after the pre-drive began; waiting for the "
                "start confirmation (%s), no timeout", _stamp(time.time()), message,
                now - cycle_started, config.START_CONFIRM_MODE)
    _frs_debug.log_event(time.time(), "predrive_ready", "ALERT", f"READY_TO_START {message}")
    if assessment is not None:
        assessment.mark("ready_entered", now, source="override")
    while True:
        frame = next_frame()
        rate.tick()
        display_text(frame, f"{message} - you may start the vehicle", (0, 200, 0))
        display_text(frame, f"Start the vehicle, {_start_confirm.prompt}", GREY, dy=40)
        draw_banner(frame, "PRE-DRIVE  |  READY TO START  |  starter UNLOCKED", (0, 200, 0))
        if present(frame) == KEY_RETRY:
            logger.info("Pre-drive: re-assessment requested from the preview window")
            return False
        source = _start_confirm.confirmed(SessionState.READY_TO_START)
        if source is not None:
            log_start_confirmed(source, cycle_started, time.monotonic(), assessment)
            return True


def await_override(
    api_client: APIClient,
    alert_manager: AlertManager,
    ignition: Any,
    rate: LoopRate,
    driver_id: Optional[int],
    reason: LockReason,
    assessment_id: Optional[int],
    provenance: Optional[Dict[str, Any]] = None,
    continue_to_monitoring: bool = True,
    display_reason: Optional[str] = None,
    cycle_started: Optional[float] = None,
) -> bool:
    """
    Starter stays inhibited (LOCKED): raise one override request and poll
    the operator's decision until approved (-> release starter), the user
    presses ``r`` (-> re-assess) or the session is ended (END SESSION).

    ``display_reason`` replaces the lock reason's text on the unit's own
    screen only (an invalid assessment is raised to the backend as
    ``driver_not_recognized`` but shown as what it is).

    Every :class:`LockReason` resolves through this one path. ``provenance``
    (``Calibration.provenance()``) goes on the request so the operator can
    see which calibration was involved - for ``FOREIGN_DEVICE_BASELINE``,
    which unit it was captured on.

    Returns:
        ``True`` if the override was approved and the start then confirmed
        (READY_TO_START via :func:`wait_for_start` - an approval releases
        the starter but, as for a pass, does not enter monitoring by
        itself). ``False`` on ``r``, here or on the READY screen.
        ``continue_to_monitoring`` is kept for the caller's sake: the
        supervisor decides what a confirmation leads to.
    """
    reason_text = display_reason or LOCK_REASON_TEXT[reason]
    logger.warning("Pre-drive: starter stays LOCKED - %s (driver=%s)", reason.value, driver_id)
    alert_manager.lock_relay()  # already locked; explicit for clarity + log
    set_session_state(SessionState.LOCKED, reason_text)

    request_id = api_client.request_override(config.DEVICE_ID, driver_id, reason, assessment_id,
                                             provenance)
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
                    config.DEVICE_ID, driver_id, reason, assessment_id, provenance)
            elif status == "pending":
                polled = api_client.check_override_request(request_id)
                if polled is not None and polled != status:
                    status = polled
                    logger.warning("Override request %s -> %s", request_id, status)

        if status == "approved":
            logger.warning("Operator override APPROVED (request %s) - releasing starter", request_id)
            alert_manager.unlock_relay()
            alert_manager.set_alert_level("ALERT")
            return wait_for_start(rate, "OVERRIDE APPROVED",
                                  time.monotonic() if cycle_started is None
                                  else cycle_started)

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
            return False


class DriverCheck:
    """
    Is the driver who was recognised still the one in the seat? (2026-10-19)

    Pre-drive recognises the driver once (:func:`identify_driver_bounded`).
    While the rolling assessment can grant READY_TO_START, this re-checks:

    * a face lost for longer than ``config.NO_FACE_FAULT_S`` makes the
      identity *unconfirmed* until the face is matched to the same driver
      again - attempted back to back meanwhile;
    * once eligible (the first 30 s are over) a confirmed identity is
      re-checked every ``config.PREDRIVE_REIDENTIFY_S``;
    * a match to a *different* enrolled driver makes it unconfirmed at once
      (READY_TO_START is lost), and once that driver has matched
      ``config.PREDRIVE_RECOGNITION_MATCHES`` times it is a change of driver
      (:meth:`observe` returns their id). A same-driver match confirms
      again.

    "No match" never changes a confirmed identity: one blurred frame must not
    toggle the starter under a driver holding the key. After a face loss it
    keeps the identity unconfirmed, so a stranger who sits down cannot inherit
    the previous driver's READY_TO_START.

    The recognition itself runs in the worker process and comes back a frame
    or more after it was submitted. ``generation`` tags each request; every
    face loss starts a new generation, and a result for a frame from before
    the loss is discarded - it says nothing about who sat down after it.
    """

    def __init__(self, driver_id: int, interval_s: Optional[float] = None,
                 matches_required: Optional[int] = None) -> None:
        self.driver_id = int(driver_id)
        self.interval_s = float(config.PREDRIVE_REIDENTIFY_S if interval_s is None
                                else interval_s)
        self.matches_required = int(config.PREDRIVE_RECOGNITION_MATCHES
                                    if matches_required is None else matches_required)
        self.confirmed = True
        self.why = ""
        self.others: Dict[int, int] = {}
        self.attempts = 0
        self.generation = 0
        self.owner = next(_recognition_owner)
        # No periodic checks until start_periodic() (the eligibility point).
        self.next_due = float("inf")

    def start_periodic(self, now_mono: float) -> None:
        self.next_due = now_mono + self.interval_s

    def face_lost(self, gap_s: float) -> None:
        self.generation += 1
        if self.confirmed:
            self.confirmed, self.why = False, f"face lost {gap_s:.1f}s"
            logger.info("Pre-drive: face lost for %.1fs - driver %s must be re-identified "
                        "before READY_TO_START", gap_s, self.driver_id)

    def due(self, now_mono: float) -> bool:
        return not self.confirmed or now_mono >= self.next_due

    def tag(self) -> Tuple[str, int, int]:
        """Tag for a request submitted now: this check, this generation."""
        return ("predrive", self.owner, self.generation)

    def observe(self, match: Optional[Dict[str, Any]], now_mono: float,
                tag: Optional[Tuple[str, int, int]] = None) -> Optional[int]:
        """
        One recognition result, for a request tagged ``tag`` (:meth:`tag`).
        Returns the new driver's id on a change of driver, else ``None``.
        A result for another owner, or from before a face loss, is discarded.
        """
        if tag is not None and tag != self.tag():
            logger.info("Pre-drive: re-identification result %s discarded (%s)", tag,
                        "from before a face loss" if tag[:2] == self.tag()[:2]
                        else "not this check's")
            return None
        self.attempts += 1
        self.next_due = now_mono + self.interval_s
        if match is None:
            if not self.confirmed:
                logger.info("Pre-drive: re-identification attempt %d - no match; driver %s "
                            "still unconfirmed (%s)", self.attempts, self.driver_id, self.why)
            return None
        did = int(match["driver_id"])
        if did == self.driver_id:
            if not self.confirmed:
                logger.info("Pre-drive: driver %s re-identified (%s) - identity confirmed",
                            did, self.why)
            self.confirmed, self.why = True, ""
            self.others.clear()
            return None
        self.others[did] = self.others.get(did, 0) + 1
        self.confirmed, self.why = False, f"matched driver {did}"
        logger.warning("Pre-drive: face matched driver %s (%s), not %s - %d/%d matches for a "
                       "change of driver", did, match.get("name"), self.driver_id,
                       self.others[did], self.matches_required)
        return did if self.others[did] >= self.matches_required else None


def _not_ready_text(w: Dict[str, Any], assessment: PredriveAssessment) -> str:
    """Why the rolling window does not pass, for the log and the screen."""
    if w["reason"] == "microsleep":
        return "microsleep in the last %.0fs" % assessment.duration_s
    if w["reason"] == "invalid":
        return "valid frames %.0f%% < %.0f%% in the last %.0fs" % (
            w["valid_fraction"] * 100, assessment.min_valid_fraction * 100,
            assessment.duration_s)
    return "worst %.0fs FRS %.3f >= %.2f in the last %.0fs" % (
        assessment.worst_window_s, w["worst_window_mean"], assessment.pass_threshold,
        assessment.duration_s)


def close_assessment(assessment: PredriveAssessment, driver_id: Optional[int], reason: str,
                     **detail: Any) -> None:
    """End the rolling run: timeline ``ended``, CSV stream closed, files written / updated."""
    assessment.finish(reason, time.monotonic(), **detail)
    if not assessment.files_written:
        try:
            assessment.write_files(config.DEVICE_ID, driver_id)
        except OSError as exc:
            logger.error("Could not write assessment files: %s", exc)
    s = assessment.timeline_summary()
    logger.info("Pre-drive run ended (%s): %.1fs assessed, READY_TO_START entered %d time(s), "
                "first at %s, start confirmed at %s, pre-drive duration %s, worst frame gap "
                "%.3fs", reason,
                assessment.elapsed(), s["ready_entries"],
                _stamp(s["first_ready_at"]) if s["first_ready_at"] else "never",
                _stamp(s["confirmed_at"]) if s["confirmed_at"] else "never",
                "%.1fs" % s["predrive_duration_s"] if s["predrive_duration_s"] is not None
                else "n/a (not confirmed)", s["max_frame_gap_s"])


# ---------------------------------------------------------------------------
# Phase 1: pre-drive assessment
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
    diag_ear: bool = False,
    continue_to_monitoring: bool = True,
) -> bool:
    """
    One pre-drive cycle (2026-10-19: start-gated, rolling).

    1. Relay inhibited (``set_phase(PREDRIVE)``).
    2. IDENTIFYING: recognise the driver (bounded) -> else DRIVER_NOT_RECOGNIZED.
    3. This driver's calibration from this unit  -> none at all: NO_BASELINE;
       only from another unit: FOREIGN_DEVICE_BASELINE (never assessed on it).
    4. SETTLING (:func:`run_settling`): face + landmarks valid for
       ``config.PREDRIVE_SETTLE_SECONDS`` continuously (20 s timeout);
       nothing scored, but a microsleep confirmed here fails at once as
       MICROSLEEP_DETECTED.
    5. ASSESSING: fresh pipeline (every rolling buffer empty), neutral head
       pose (stored, else the settling median), then the assessment - which
       no longer ends at ``config.PREDRIVE_ASSESSMENT_SECONDS``. Every frame
       recorded, invalid frames counted and never scored.
    6. Eligibility, at 30 s: the first 30 s are judged exactly as the fixed
       run was, persisted (CSV/JSON) and POSTed to /assessments. A fail is a
       HARD failure. Priority: MICROSLEEP_DETECTED (settling or assessment)
       > invalid assessment (fewer than ``config.PREDRIVE_MIN_VALID_FRACTION``
       valid frames, or no settling: FAULT, raised as DRIVER_NOT_RECOGNIZED)
       > FATIGUE_DETECTED.
    7. From then on the pass rule is re-applied every frame to the most
       recent 30 s (:meth:`PredriveAssessment.window_verdict`). READY_TO_START
       - starter released, green LED, "Assessment passed - you may start the
       vehicle" - while it passes, a face has been seen within
       ``config.NO_FACE_FAULT_S`` and the driver is confirmed
       (:class:`DriverCheck`). READY is not a latch: when any of these stops
       holding the starter is re-inhibited and the run goes on ASSESSING,
       and it is regained when they hold again. A confirmed microsleep after
       eligibility is still a hard failure (a microsleep fails outright, as
       in settling). A change of driver ends the cycle (``False``): the
       caller restarts pre-drive for the new driver.
    8. MONITORING only on the start confirmation (``config.START_CONFIRM_MODE``,
       the BEGIN MONITORING button), polled only in READY_TO_START. There is
       no timeout: without it the unit stays where it is indefinitely.
       READY_TO_START entry, loss and the confirmation are logged and
       recorded in the assessment's timeline separately.
    9. Any hard failure -> :func:`await_override` (LOCKED); the starter stays
       inhibited until an operator approves (then READY_TO_START, still
       waiting for the confirmation) or the session is reset (``r`` re-runs,
       END SESSION returns to IDLE).

    END SESSION (:class:`SessionEnded`, raised by :func:`present`) can come
    at any point; an assessment in progress is closed and its files written
    before it propagates.

    The display shows "Recognizing...", "Hold still..." and "Assessing..."
    for the three stages (Hershey fonts have no ellipsis character).

    ``continue_to_monitoring`` off (``--force-phase predrive``, the phase
    tested alone) changes nothing here; the supervisor does not enter
    monitoring after the confirmation.

    Returns:
        ``True`` once the start is confirmed (pass, or approved override):
        the starter is released and the caller goes into monitoring.
        ``False`` if the user asked to re-run (``r``), the driver changed,
        or the ignition turned ON before a release (the caller re-reads the
        phase; the starter stays inhibited).
    """
    alert_manager.set_phase(Phase.PREDRIVE)
    # Every pre-drive cycle starts inhibited, including a re-run requested
    # with 'r' after a pass (set_phase is a no-op when already in PREDRIVE).
    alert_manager.lock_relay()
    alert_manager.set_alert_level("ALERT")
    _screen.reset_session()
    # Durations from here are monotonic (2026-10-19): a wall-clock step (an
    # NTP sync after a cold boot) must not move the 30 s minimum or the
    # rolling window. time.time() is only recorded and logged.
    cycle_started = time.monotonic()
    cycle_started_wall = time.time()
    global _released_driver
    _calibration_cache.clear()
    _released_driver = None
    logger.info("=== PRE-DRIVE phase (no ignition input) - starter %s ===",
                alert_manager.get_relay_state())

    driver: Optional[Dict[str, Any]] = None
    driver_id: Optional[int] = None
    calibration: Optional[Calibration] = None
    # Provenance of the record the backend returned, used or not.
    calibration_record: Optional[Dict[str, Any]] = None
    lock_reason: Optional[LockReason] = None
    assessment_id: Optional[int] = None
    # Set when the run is an invalid assessment (FAULT): the real cause, for
    # the local log, the debug CSV and the unit's screen.
    invalid_cause: Optional[str] = None
    settle: Optional[Dict[str, Any]] = None
    assessment: Optional[PredriveAssessment] = None

    try:
        # 2. Recognise
        set_session_state(SessionState.IDENTIFYING)
        driver = identify_driver_bounded(recognizer, ignition, rate)
        if driver is None:
            lock_reason = LockReason.DRIVER_NOT_RECOGNIZED
        else:
            driver_id = int(driver["driver_id"])
            _screen.driver_name = str(driver["name"])
            # 3. Calibration - this driver's own, captured on this unit, or
            # lock. One captured on another unit is never allowed to release
            # the starter (fail-secure: a baseline that reads low passes a
            # fatigued driver), so it locks without an assessment and the
            # override request carries its provenance for the operator.
            calibration, calibration_record = load_calibration(api_client, driver_id,
                                                               Phase.PREDRIVE)
            if calibration is None:
                # No record, or a broken one. A broken record from another
                # unit also lands here, deliberately: no_baseline is the
                # accurate category and the operator's action is the same,
                # whereas foreign_device_baseline would imply that approving
                # leaves a usable baseline. Its provenance still goes on the
                # override request as context.
                lock_reason = LockReason.NO_BASELINE
            elif calibration.source != CALIBRATION_SOURCE_DEVICE:
                logger.warning("Pre-drive: driver %s's only calibration (%s) was captured on "
                               "%s, not %s - locking for operator override",
                               driver_id, calibration.calibration_id,
                               calibration.captured_on_device_id or "an unrecorded device",
                               config.DEVICE_ID)
                lock_reason = LockReason.FOREIGN_DEVICE_BASELINE
            else:
                thresholds = calibration.thresholds
                # Monitoring after this cycle's start reuses it (no fetch).
                _calibration_cache[driver_id] = calibration

        if lock_reason is None:
            # 4. Settle: the face and landmarks must be valid for
            # PREDRIVE_SETTLE_SECONDS continuously; nothing is scored.
            set_session_state(SessionState.SETTLING)
            settle = run_settling(extractor, head_pose, ignition, rate,
                                  thresholds.get("ear_threshold"))
            if settle is not None and settle["microsleep"] is not None:
                # A microsleep outranks everything else: fail now, no 30 s run.
                lock_reason = LockReason.MICROSLEEP_DETECTED
                _frs_debug.log_event(time.time(), "predrive_verdict", "DANGER",
                                     "MICROSLEEP_DETECTED during settling (%.2fs closure)"
                                     % (settle["microsleep"]["duration_ms"] / 1000.0))
            elif settle is None:
                invalid_cause = (
                    "INVALID_ASSESSMENT: face/landmarks never valid for %.1fs within %.0fs "
                    "(valid 0.0%% of the assessment - it never started)"
                    % (config.PREDRIVE_SETTLE_SECONDS, config.PREDRIVE_SETTLE_TIMEOUT_S))
                logger.warning("Pre-drive: %s - FAULT, starter stays inhibited (raised as "
                               "driver_not_recognized)", invalid_cause)
                lock_reason = LockReason.DRIVER_NOT_RECOGNIZED

        if lock_reason is None:
            # 5. Assessment. A fresh pipeline, so every rolling buffer
            # (PERCLOS, blink duration / frequency, microsleep, EAR median,
            # escalation timers) starts empty here: nothing from recognition
            # or settling is in it.
            pipeline = MetricsPipeline(
                head_pose,
                blink_window_s=config.PREDRIVE_ASSESSMENT_SECONDS,
                pose_release_hold=pose_release_hold,
                diag_ear=diag_ear,
            )
            stored_neutral = load_neutral(driver_id)
            if stored_neutral is not None:
                pipeline.set_pose_neutral(stored_neutral, "stored")
            else:
                # None too if settling saw too few poses: the pipeline then
                # seeds it from the assessment's first seconds.
                pipeline.set_pose_neutral(settle["neutral"], "settling")
            assessment = PredriveAssessment()
            assessment.calibration = calibration.provenance()
            assessment.predrive_started_mono = cycle_started
            assessment.predrive_started_at = cycle_started_wall
            assessment.start(time.monotonic())
            set_session_state(SessionState.ASSESSING)
            logger.info("Pre-drive: assessment started for driver %s (%s) - rolling, %d s "
                        "minimum, start-gated (%s confirmation), calibration %s (%s); scoring "
                        "baselines %s",
                        driver_id, driver["name"], config.PREDRIVE_ASSESSMENT_SECONDS,
                        config.START_CONFIRM_MODE,
                        calibration.calibration_id, calibration.source,
                        {k: round(float(v), 4) for k, v in
                         pipeline.scoring_thresholds(thresholds).items()
                         if k in ("perclos_baseline", "blink_duration_baseline",
                                  "blink_frequency_baseline")})

            check = DriverCheck(driver_id)
            inline_recognition = InlineRecognition(recognizer)
            last_valid_mono = time.monotonic()
            # The first-30 s verdict, once eligible; None before.
            result: Optional[Dict[str, Any]] = None
            ready = False
            not_ready = ""

            # now: time.monotonic().
            def enter_ready(now: float, w: Dict[str, Any]) -> None:
                alert_manager.unlock_relay()
                alert_manager.set_alert_level("ALERT")
                _start_confirm.reset("new READY_TO_START")
                set_session_state(SessionState.READY_TO_START, "assessment passed")
                assessment.mark("ready_entered", now, worst_window_mean=w["worst_window_mean"],
                                valid_fraction=w["valid_fraction"])
                logger.info("READY_TO_START at %s - %.1fs after the pre-drive began, %.1fs into "
                            "the assessment (last %.0fs: worst %.0fs mean %.3f, valid %.1f%%) - "
                            "starter released; monitoring waits for the start confirmation "
                            "(%s), no timeout", _stamp(time.time()), now - cycle_started,
                            assessment.elapsed(now), config.PREDRIVE_ASSESSMENT_SECONDS,
                            config.PREDRIVE_WORST_WINDOW_SECONDS, w["worst_window_mean"],
                            w["valid_fraction"] * 100, config.START_CONFIRM_MODE)
                _frs_debug.log_event(time.time(), "predrive_ready", "ALERT",
                                     "READY_TO_START worst_%.0fs_mean=%.3f valid=%.1f%%"
                                     % (config.PREDRIVE_WORST_WINDOW_SECONDS,
                                        w["worst_window_mean"], w["valid_fraction"] * 100))

            def lose_ready(now: float, why: str) -> None:
                alert_manager.lock_relay()
                set_session_state(SessionState.ASSESSING, why)
                assessment.mark("ready_lost", now, reason=why)
                logger.warning("READY_TO_START LOST at %s (%s) - starter inhibited, assessing "
                               "(%.1fs into the assessment)", _stamp(time.time()), why,
                               assessment.elapsed(now))
                _frs_debug.log_event(time.time(), "predrive_ready_lost", "ALERT",
                                     f"READY_TO_START lost: {why}")

            while True:
                check_phase(ignition, Phase.PREDRIVE)
                if result is None and assessment.is_complete(time.monotonic()):
                    # 6. Eligibility: the first 30 s, judged as the fixed run was.
                    result = assessment.result()
                    assessment.freeze()
                    lock_reason, invalid_cause = _first_window_verdict(result)
                    try:
                        assessment.write_files(config.DEVICE_ID, driver_id)
                        assessment.stream_to_csv()
                    except OSError as exc:
                        logger.error("Could not write assessment files: %s", exc)
                    assessment_id = api_client.post_assessment(
                        config.DEVICE_ID, driver_id, result, lock_reason)
                    if lock_reason is not None:
                        if invalid_cause is None:
                            _frs_debug.log_event(time.time(), "predrive_verdict", "DANGER",
                                                 "%s worst_%.0fs_mean=%.3f microsleeps=%d"
                                                 % (lock_reason.value.upper(),
                                                    result["worst_window_s"],
                                                    result["worst_window_mean"],
                                                    result["microsleeps"]))
                        close_assessment(assessment, driver_id, "hard_failure",
                                         lock_reason=lock_reason.value)
                        break
                    logger.info("Pre-drive PASSED the first %.0fs (worst %.0fs mean %.3f < %.2f) - "
                                "eligible; READY_TO_START while the last %.0fs keep passing",
                                result["duration_s"], result["worst_window_s"],
                                result["worst_window_mean"], result["pass_threshold"],
                                config.PREDRIVE_ASSESSMENT_SECONDS)
                    _frs_debug.log_event(time.time(), "predrive_verdict", "ALERT",
                                         "PASS worst_%.0fs_mean=%.3f valid=%.1f%%"
                                         % (result["worst_window_s"], result["worst_window_mean"],
                                            result["valid_fraction"] * 100))
                    check.start_periodic(time.monotonic())

                frame = next_frame()
                fps = rate.tick()
                now, now_mono = time.time(), time.monotonic()
                eligible = result is not None
                remaining = assessment.seconds_remaining(now_mono)

                # Re-identification (DriverCheck) runs in the recognition
                # worker process: submitted on a face frame, answered on a
                # later one, polled on every frame - the loop never waits
                # for identify() (~1 s on the Pi). Inline only as a fallback.
                channel = recognition_channel(inline_recognition)
                identified = channel.poll()

                landmarks, _rect = extractor.extract(frame)
                problem = landmark_problem(landmarks, frame.shape)
                m: Optional[FrameMetrics] = None
                if problem is not None:
                    # No face, or landmarks that cannot be measured: counted,
                    # never scored. Policy disabled here: verdict.level is
                    # ALERT. Too many of these void the window instead.
                    verdict = pipeline.note_no_face(now, now_mono)
                    assessment.add_invalid(problem, now_mono)
                    _frs_debug.log_unscored(now, "predrive_assessing", False, verdict.level,
                                            unscored_reason(problem))
                    level = verdict.level
                    gap = now_mono - last_valid_mono
                    if gap > config.NO_FACE_FAULT_S:
                        check.face_lost(gap)
                else:
                    if check.due(now_mono) and not channel.busy:
                        if channel.submit(frame, check.tag()) and channel.blocking:
                            # Fallback: identify() just ran here. Re-stamp, as
                            # monitoring does, so the pipeline sees the gap
                            # and abandons any closure across it.
                            now, now_mono = time.time(), time.monotonic()
                            identified = identified or channel.poll()
                    last_valid_mono = now_mono
                    m = pipeline.process(landmarks, thresholds, now, now_mono)
                    assessment.add(m, now_mono)
                    _frs_debug.log_frame(m, "predrive_assessing", pipeline.pose_neutral_source)
                    level = m.level
                    if eligible and m.microsleep_event is not None:
                        # A microsleep fails outright, also after eligibility.
                        lock_reason = LockReason.MICROSLEEP_DETECTED
                        if ready:
                            lose_ready(now_mono, "microsleep")
                        logger.warning(
                            "Pre-drive: MICROSLEEP %.1fs into the assessment (eyes closed "
                            "%.2fs) after the first %.0fs had passed - hard failure, starter "
                            "inhibited", assessment.elapsed(now_mono),
                            m.microsleep_event["duration_ms"] / 1000.0,
                            config.PREDRIVE_ASSESSMENT_SECONDS)
                        _frs_debug.log_event(now, "predrive_verdict", "DANGER",
                                             "MICROSLEEP_DETECTED after eligibility "
                                             "(%.2fs closure)"
                                             % (m.microsleep_event["duration_ms"] / 1000.0))
                        close_assessment(assessment, driver_id, "hard_failure",
                                         lock_reason=lock_reason.value)
                        break

                if identified is not None:
                    new_driver = check.observe(identified[1], now_mono, tag=identified[0])
                    if new_driver is not None:
                        if ready:
                            lose_ready(now_mono, f"driver changed to {new_driver}")
                        alert_manager.lock_relay()
                        logger.warning("Pre-drive: DRIVER CHANGED from %s to %s - starter "
                                       "inhibited; restarting pre-drive for the new driver",
                                       driver_id, new_driver)
                        close_assessment(assessment, driver_id, "driver_changed",
                                         new_driver_id=new_driver)
                        return False

                # 7. Rolling verdict -> READY_TO_START (not a latch).
                if eligible:
                    w = assessment.window_verdict(now_mono)
                    face_gap = now_mono - last_valid_mono
                    if face_gap > config.NO_FACE_FAULT_S:
                        not_ready = f"no face for {face_gap:.1f}s"
                    elif not check.confirmed:
                        not_ready = f"confirming the driver ({check.why})"
                    elif not w["passed"]:
                        not_ready = _not_ready_text(w, assessment)
                    else:
                        not_ready = ""
                    if not not_ready and not ready:
                        enter_ready(now_mono, w)
                        ready = True
                    elif not_ready and ready:
                        lose_ready(now_mono, not_ready)
                        ready = False

                # LEDs / buzzer: green while READY_TO_START (the verdict),
                # else the frame's level as driver feedback. The relay is
                # driven only by enter_ready / lose_ready.
                alert_manager.set_alert_level("ALERT" if ready else level)

                if ready:
                    banner, color = ("PRE-DRIVE  |  READY TO START  |  starter "
                                     f"{alert_manager.get_relay_state()}"), (0, 200, 0)
                elif eligible:
                    banner, color = (f"PRE-DRIVE  |  assessing  |  starter "
                                     f"{alert_manager.get_relay_state()}"), (0, 220, 255)
                else:
                    banner, color = (f"PRE-DRIVE  |  assessing {remaining:4.1f}s left  |  "
                                     f"starter {alert_manager.get_relay_state()}"), (0, 220, 255)
                if ready:
                    display_text(frame, "Assessment passed - you may start the vehicle",
                                 (0, 200, 0))
                    display_text(frame, f"Start the vehicle, {_start_confirm.prompt}", GREY,
                                 dy=40)
                else:
                    display_text(frame, "Assessing...", (0, 220, 255))
                    if problem is not None:
                        display_text(frame, "NO FACE" if problem == "no_face"
                                     else "FACE NOT FULLY VISIBLE", dy=40)
                    elif not_ready:
                        display_text(frame, f"Not ready: {not_ready}", GREY, dy=40)
                if problem is not None:
                    _screen.face, _screen.level = FACE_NONE, level
                else:
                    draw_overlay(frame, driver, m, fps, pipeline, thresholds, landmarks)
                    if not check.confirmed:
                        _screen.face, _screen.face_detail = FACE_SEARCHING, "- confirming driver"
                _screen.countdown = None if eligible else remaining
                draw_banner(frame, banner, color)
                if debug_pose:
                    draw_pose_debug(frame, pipeline.pose_debounce, now_mono, fps)
                key = present(frame)
                if ready and key == KEY_RETRY:
                    # As on the old "start vehicle" screen: re-run from the top.
                    logger.info("Pre-drive: re-assessment requested from the preview window")
                    lose_ready(time.monotonic(), "re-assessment requested")
                    close_assessment(assessment, driver_id, "rerun")
                    return False

                # 8. Start confirmation - only ever taken in READY_TO_START.
                if ready:
                    source = _start_confirm.confirmed(SessionState.READY_TO_START)
                    if source is not None:
                        log_start_confirmed(source, cycle_started, time.monotonic(),
                                            assessment)
                        close_assessment(assessment, driver_id, "confirmed", source=source)
                        _released_driver = driver
                        if continue_to_monitoring:
                            show_release(rate, "START CONFIRMED")
                        return True

        # 9. Lock path (every hard failure). An invalid assessment shows FAULT
        # and its real cause on the unit; the backend gets the lock reason.
        if invalid_cause is not None:
            _frs_debug.log_event(time.time(), "predrive_verdict", "FAULT", invalid_cause)
        alert_manager.set_alert_level("FAULT" if invalid_cause is not None else "ALERT")
        released = await_override(api_client, alert_manager, ignition, rate, driver_id,
                                  lock_reason, assessment_id, calibration_record,
                                  continue_to_monitoring=continue_to_monitoring,
                                  display_reason=("INVALID ASSESSMENT" if invalid_cause is not None
                                                  else None),
                                  cycle_started=cycle_started)
        if released:
            # Approved override: monitoring reuses this cycle's calibration
            # only if one was loaded and cached (not for no_baseline /
            # foreign_device_baseline / not recognised).
            _released_driver = driver
        return released

    except (SessionEnded, QuitRequested, KeyboardInterrupt) as exc:
        # The trial's record says how it ended (END SESSION, or STOP / quit).
        if assessment is not None and assessment.timeline_summary()["end_reason"] is None:
            close_assessment(assessment, driver_id, "session_ended"
                             if isinstance(exc, SessionEnded) else "stopped")
        raise
    except PhaseChanged:
        logger.info("Pre-drive: ignition turned ON - leaving pre-drive (starter %s)",
                    alert_manager.get_relay_state())
        return False


def _first_window_verdict(result: Dict[str, Any]) -> Tuple[Optional[LockReason], Optional[str]]:
    """
    The eligibility verdict on the first ``config.PREDRIVE_ASSESSMENT_SECONDS``
    - the fixed assessment's verdict, unchanged. Logs it.

    Returns:
        ``(lock_reason, invalid_cause)``: ``(None, None)`` on a pass.
    """
    logger.info("Pre-drive result: passed=%s void=%s valid=%.1f%% worst_%.0fs_mean=%.3f "
                "mean=%.3f median=%.3f max=%.3f n=%d invalid=%d %s "
                "pose_override_frames=%d yawns=%d microsleeps=%d",
                result["passed"], result["void"], result["valid_fraction"] * 100,
                result["worst_window_s"], result["worst_window_mean"],
                result["mean_frs"], result["median_frs"], result["max_frs"],
                result["n_samples"], result["n_no_face"], result["invalid_reasons"],
                result["pose_override_frames"], result["yawns"],
                result["microsleeps"])
    # Priority (2026-10-15): microsleep_detected > invalid assessment
    # > FRS fail. A confirmed microsleep was measured on valid frames
    # throughout, so it stands even if the run as a whole was invalid.
    if result["failed_on_microsleep"]:
        # Checked before the aggregate so the lock reason names the
        # real finding: a microsleep fails outright, and the worst
        # window can easily still be under the threshold when it does.
        if result["void"]:
            logger.warning("Pre-drive: assessment also INVALID (valid frames %.1f%%) - "
                           "the microsleep takes priority",
                           result["valid_fraction"] * 100)
        logger.warning(
            "Pre-drive: FAILED ON MICROSLEEP - %d confirmed during the %.0fs "
            "assessment (worst %.0fs mean %.3f vs threshold %.2f; the aggregate "
            "alone would have %s). Sleep intrusion in a stationary vehicle is "
            "disqualifying on its own.",
            result["microsleeps"], result["duration_s"], result["worst_window_s"],
            result["worst_window_mean"], result["pass_threshold"],
            "FAILED" if result["worst_window_mean"] >= result["pass_threshold"]
            else "PASSED",
        )
        return LockReason.MICROSLEEP_DETECTED, None
    if result["void"]:
        # Not enough valid frames to judge: an invalid assessment
        # (FAULT), not fatigue. Raised as driver_not_recognized - the
        # backend has no separate reason - with the real cause here.
        invalid_cause = ("INVALID_ASSESSMENT: valid frames %.1f%% < %.0f%% (%s)"
                         % (result["valid_fraction"] * 100,
                            result["min_valid_fraction"] * 100,
                            ", ".join(f"{k} {v}" for k, v in
                                      result["invalid_reasons"].items()) or "no frames"))
        logger.warning("Pre-drive: %s - FAULT, starter stays inhibited (raised as "
                       "driver_not_recognized)", invalid_cause)
        return LockReason.DRIVER_NOT_RECOGNIZED, invalid_cause
    if not result["passed"]:
        logger.warning(
            "Pre-drive: FAILED on aggregate - worst %.0fs mean %.3f >= %.2f",
            result["worst_window_s"], result["worst_window_mean"],
            result["pass_threshold"],
        )
        return LockReason.FATIGUE_DETECTED, None
    return None, None


# ---------------------------------------------------------------------------
# IDLE and END SESSION (2026-10-19)
# ---------------------------------------------------------------------------

def run_idle(rate: LoopRate) -> None:
    """
    IDLE, after END SESSION: starter inhibited, LEDs and buzzer off, the
    camera still read (so the heartbeat keeps ticking) but nothing processed.
    Returns when START NEXT DRIVER (BEGIN, made in IDLE) is tapped.
    """
    set_session_state(SessionState.IDLE, "waiting for the next driver")
    logger.info("IDLE - starter %s; waiting for the next driver (START NEXT DRIVER on the "
                "launcher, or key 'b')",
                _alert_manager.get_relay_state() if _alert_manager else "n/a")
    while True:
        frame = next_frame()
        rate.tick()
        display_text(frame, "Session ended", WHITE)
        display_text(frame, "Waiting for the next driver - tap START NEXT DRIVER", GREY, dy=40)
        draw_banner(frame, "IDLE  |  starter LOCKED  |  waiting for the next driver", GREY)
        present(frame)
        if _requests.take(BEGIN, SessionState.IDLE) is not None:
            logger.info("IDLE -> next driver: starting pre-drive")
            return


def end_session(alert_manager: AlertManager) -> None:
    """
    END SESSION (2026-10-19): the stand-in for key-OFF, which cannot be
    sensed. Inhibits the starter (leaving monitoring first, so the lock is
    allowed), stops any active alert and forgets the driver and any pending
    request. The loop that was running has already closed its own state
    (assessment files, monitoring faults) on the way out. The caller then
    runs :func:`run_idle`; the process keeps running.
    """
    was = _session_state.value if _session_state is not None else "-"
    _requests.discard(END_SESSION)
    _requests.discard(BEGIN, "session ended")
    alert_manager.set_phase(Phase.PREDRIVE)
    alert_manager.lock_relay()
    alert_manager.idle()
    _screen.reset_session()
    # No recognition process outlives the session; the next driver's starts
    # when IDLE is left.
    stop_recognition("END SESSION")
    global _released_driver
    _calibration_cache.clear()
    _released_driver = None
    logger.warning("SESSION ENDED by the operator at %s (was %s) - starter %s, alerts off, "
                   "pre-drive / READY_TO_START state cleared", _stamp(time.time()), was,
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
    diag_ear: bool = False,
    from_release: bool = False,
    profile_s: Optional[float] = None,
) -> None:
    """
    Monitor continuously while the ignition is ON. Returns when it turns OFF.

    ``from_release``: entered straight from a pre-drive release (pass or
    approved override), when the key may not have been turned yet. The
    ignition reading OFF then means "not started yet", not "trip over", so
    this returns only once the ignition has been seen ON and has gone OFF
    again. Under ``--sequence`` (no ignition input) that never happens and
    monitoring runs until stopped.

    ``profile_s``: ``--profile-loop`` - time every frame by category and log
    a :class:`LoopProfiler` summary every ``profile_s`` seconds.

    Alerts (LEDs, buzzer) and backend DANGER notifications only. The relay is
    never driven here; ``AlertManager`` refuses ``lock_relay()`` in this
    phase, so nothing in this loop can inhibit the starter even by mistake.
    A driver without a usable calibration (unrecognised, no record, record
    failing the threshold check, backend offline) is monitored on an EAR
    baseline self-seeded from their own EAR over the first
    ``SELF_SEED_S`` of face time; until it freezes, only the head-pose and
    microsleep overrides run. The remaining baselines are
    :data:`DEFAULT_THRESHOLDS`.

    Face recognition runs on face frames only until the driver is first
    recognised, then never again this session (2026-09-29). At ~1 s per call
    on the Pi (``face_locations`` + ``face_encodings``) the old 1.5 s
    re-identify cadence held the loop at ~11 fps; it cannot move to a
    thread because dlib holds the GIL for the whole call. Until a match it
    is retried every :data:`UNRECOGNISED_RETRY_S` (was: every face frame,
    ~1 fps), spaced from the answer to the previous attempt. What an
    unrecognised driver gets is unchanged: no driver, no calibration,
    self-seeded EAR baseline, ``driver_id`` null on events and faults.

    Since 2026-10-19 the call runs in the recognition worker process
    (modules/recognition_worker.py), like pre-drive's re-identification: a
    face frame is submitted, the answer is taken on a later face frame and
    the loop never waits. An answer for a frame from before a face loss
    longer than ``config.NO_FACE_FAULT_S`` is discarded and the attempt
    repeated. Only if the worker is unavailable does ``identify()`` run on
    the loop, as before. The worst gap between two frames is logged at the
    end and saved with the run's summary in ``logs/monitoring/``.

    A different driver (2026-10-19): if recognition matches someone other
    than the driver who passed pre-drive, that driver has not been assessed.
    Scoring switches at once to a fresh self-seeded baseline (no calibration
    is fetched or waited for), a driver-mismatch fault is opened (reported as
    ``no_ear_baseline``, see :data:`FAULT_TYPE_DRIVER_MISMATCH`) and the
    screen says so for the rest of the session.

    Calibration (2026-10-19) never blocks either. Entered from a pre-drive
    that loaded and used the driver's calibration (``from_release``), that
    calibration is used from the first frame and nothing is fetched;
    recognition then confirms the driver once. Otherwise, once the driver is
    identified, their calibration comes from the pre-drive cache if it holds
    them, else from :class:`CalibrationFetch` on a background thread; until
    it answers the current values (self-seeded baseline) stay in use, and if
    it brings nothing (backend unreachable, no record) that is logged and
    monitoring carries on unchanged.

    Frames with no face follow ``modules.pipeline.NoFacePolicy``: the level
    is held (never lowered) and a DANGER is latched for up to
    ``config.NO_FACE_FAULT_S``; past it the level is FAULT, from any level,
    until a valid face returns and escalation restarts from the current
    score. The FAULT opens a backend fault on ``POST /monitoring-faults``:
    ``no_face`` from ALERT / WARNING, ``danger_latched`` (critical) from a
    latched DANGER. Each is opened
    with the last scored frame's metrics, re-POSTed every
    ``MONITORING_FAULT_REFRESH_S`` while open, and resolved when the face
    returns or the ignition turns off.

    A self-seed that gives up (``SELF_SEED_MAX_ATTEMPTS`` discarded seeds)
    puts the level at FAULT and opens a third fault, ``no_ear_baseline``,
    independent of any no-face gap; it resolves when a recognised driver
    restarts the pipeline or the ignition turns off.
    """
    global _profiler
    alert_manager.set_phase(Phase.MONITORING)
    alert_manager.set_alert_level("ALERT")
    _screen.reset_session()
    set_session_state(SessionState.MONITORING)
    logger.info("=== MONITORING phase (%s) - starter %s, lock refused ===",
                "entered from pre-drive release" if from_release else "ignition ON",
                alert_manager.get_relay_state())

    prof: Any = _NO_PROFILER
    if profile_s:
        prof = _profiler = LoopProfiler(profile_s)
        logger.info("Loop profiling ON: summary every %.0fs, per-frame CSV %s",
                    profile_s, prof._csv_path)

    pipeline = MetricsPipeline(head_pose, pose_release_hold=pose_release_hold,
                               no_face_policy=True, diag_ear=diag_ear)
    logger.info("Monitoring: FRS weights %s, theoretical max %s",
                pipeline.frs_calc.weights, pipeline.frs_calc.theoretical_max())

    driver: Optional[Dict[str, Any]] = None
    current_driver_id: Optional[int] = None
    # None = no usable calibration: the pipeline self-seeds the EAR pair and
    # takes the other baselines from DEFAULT_THRESHOLDS.
    calibration: Optional[Calibration] = None
    thresholds: Dict[str, float] = dict(DEFAULT_THRESHOLDS)
    defaults_logged = False
    # Unrecognised-driver retry schedule (UNRECOGNISED_RETRY_S).
    next_identify = float("-inf")
    identify_attempts = 0
    # Identification still wanted (until a match is accepted).
    identify_pending = True
    # Calibration fetch running off the frame loop (CalibrationFetch), if any.
    calibration_fetch: Optional[CalibrationFetch] = None
    calibration_source: Optional[str] = None
    calibration_after: Optional[float] = None
    # The driver pre-drive released the starter for (None: none, or not from a
    # release) and, once someone else is recognised, the mismatch and its fault.
    released_id: Optional[int] = (int(_released_driver["driver_id"])
                                  if from_release and _released_driver is not None
                                  and _released_driver.get("driver_id") is not None else None)
    mismatch: Optional[Dict[str, Any]] = None
    mismatch_fault: Optional[OpenFault] = None
    # Recognition requests: this run's owner token; the generation moves on
    # at every face loss longer than NO_FACE_FAULT_S (older answers are void).
    inline_recognition = InlineRecognition(recognizer)
    identify_owner = next(_recognition_owner)
    identify_generation = 0
    # Frame-loop health for the run's summary (monotonic).
    started_mono, started_wall = time.monotonic(), time.time()
    frames = 0
    prev_frame_mono: Optional[float] = None
    max_frame_gap = 0.0
    gaps_over = 0
    identified_after: Optional[float] = None
    end_reason = "stopped"
    last_danger_push = float("-inf")
    # Effective level (FRS band, or DANGER under the head-pose override) as
    # of the previous scored frame, so transitions can be logged once rather
    # than every frame. ``None`` until the first face is processed.
    last_level: Optional[str] = None
    # Last scored frame, for the last_known block of a fault report.
    last_m: Optional[FrameMetrics] = None
    # Open monitoring fault (no_face or danger_latched), if any.
    open_fault: Optional[OpenFault] = None
    # Open no_ear_baseline fault (the self-seed gave up), if any.
    seed_fault: Optional[OpenFault] = None
    # See from_release: whether this session has seen the ignition ON yet.
    ignition_seen_on = False

    def apply_calibration(cal: Calibration, source: str, now_mono: float) -> None:
        """Score from here on against ``cal`` (the identified driver's)."""
        nonlocal calibration, thresholds, seed_fault, last_danger_push
        nonlocal calibration_source, calibration_after
        calibration, thresholds = cal, cal.thresholds
        calibration_source = source
        calibration_after = now_mono - started_mono
        logger.info("Monitoring: driver %s's calibration %s in use (%s)", current_driver_id,
                    cal.calibration_id, source)
        # Fresh history so metrics accumulated before recognition (on a
        # self-seeded baseline) don't leak into this driver's. That includes
        # a self-seed that gave up: it starts over.
        pipeline.reset()
        # This driver's neutral head pose from enrollment on this unit;
        # without one it is seeded from the next few seconds.
        stored_neutral = load_neutral(current_driver_id)
        if stored_neutral is not None:
            pipeline.set_pose_neutral(stored_neutral, "stored")
        if seed_fault is not None:
            logger.info("Driver identified - no_ear_baseline fault resolved (calibration "
                        "loaded)")
            with prof.section("api"):
                seed_fault.push(api_client, "resolved", now_mono,
                                datetime.now(timezone.utc), "driver_identified")
            seed_fault = None
        last_danger_push = float("-inf")

    def summary() -> None:
        """Log and save how the frame loop did this run (every exit)."""
        duration = time.monotonic() - started_mono
        record = {
            "device_id": config.DEVICE_ID, "driver_id": current_driver_id,
            "started_at": started_wall, "ended_at": time.time(),
            "duration_s": round(duration, 3), "end_reason": end_reason,
            "frames": frames, "mean_fps": round(frames / duration, 2) if duration > 0 else None,
            "max_frame_gap_s": round(max_frame_gap, 3),
            "frame_gaps_over_limit": gaps_over, "max_frame_gap_limit_s": config.MAX_FRAME_GAP_S,
            "identify_attempts": identify_attempts,
            "identified_after_s": None if identified_after is None else round(identified_after, 3),
            "calibration_source": calibration_source,
            "driver_mismatch": mismatch,
            "calibration_after_s": (None if calibration_after is None
                                    else round(calibration_after, 3)),
            "recognition": "worker" if _recognition is not None and not _recognition.failed
                           else "inline",
        }
        logger.info("Monitoring ended (%s): %.1fs, %d frames, %.1f fps, worst frame gap %.3fs "
                    "(%d over %.1fs), recognition %s, %d identify attempt(s)", end_reason,
                    duration, frames, record["mean_fps"] or 0.0, max_frame_gap, gaps_over,
                    config.MAX_FRAME_GAP_S, record["recognition"], identify_attempts)
        directory = config.LOGS_DIR / "monitoring"
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started_wall))
        path = directory / (f"{config.DEVICE_ID}_{current_driver_id if current_driver_id is not None else 'unknown'}"
                            f"_{stamp}.json")
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("Could not write the monitoring summary %s: %s", path, exc)

    def close(resolution: str) -> None:
        """Stop profiling and resolve any fault still open on the backend."""
        global _profiler
        prof.stop()
        _profiler = None
        if open_fault is not None:
            # The driver still unseen at the end: close the record so the
            # portal does not show a fault open forever.
            open_fault.push(api_client, "resolved", time.monotonic(),
                            datetime.now(timezone.utc), resolution)
        if seed_fault is not None:
            seed_fault.push(api_client, "resolved", time.monotonic(),
                            datetime.now(timezone.utc), resolution)
        if mismatch_fault is not None:
            mismatch_fault.push(api_client, "resolved", time.monotonic(),
                                datetime.now(timezone.utc), resolution)

    # Started by the confirmed start of a pre-drive that passed: same driver,
    # calibration already loaded (and checked) there - used from the first
    # frame, no backend fetch. Recognition still confirms the driver once.
    handoff = _released_driver if from_release else None
    cached = (_calibration_cache.get(int(handoff["driver_id"]))
              if handoff is not None else None)
    if cached is not None:
        driver, current_driver_id = handoff, int(handoff["driver_id"])
        calibration, thresholds = cached, cached.thresholds
        calibration_source, calibration_after = "pre-drive hand-off", 0.0
        stored_neutral = load_neutral(current_driver_id)
        if stored_neutral is not None:
            pipeline.set_pose_neutral(stored_neutral, "stored")
        logger.info("Monitoring: driver %s (%s) from pre-drive - calibration %s reused, no "
                    "backend fetch; recognition confirms the driver once", current_driver_id,
                    handoff.get("name"), cached.calibration_id)

    try:
        while True:
            prof.frame()
            # The phase is pinned for the run (modules.ignition.ForcedIgnition;
            # TODO: ignition sense, deferred hardware), so in practice this
            # loop ends only by END SESSION or quit.
            with prof.section("gpio"):
                ignition_on = ignition.phase() is Phase.MONITORING
            if ignition_on:
                ignition_seen_on = True
            elif ignition_seen_on or not from_release:
                break
            with prof.section("capture"):
                frame = next_frame()
            fps = rate.tick()
            now, now_mono = time.time(), time.monotonic()
            frames += 1
            if prev_frame_mono is not None:
                gap = now_mono - prev_frame_mono
                max_frame_gap = max(max_frame_gap, gap)
                gaps_over += gap > config.MAX_FRAME_GAP_S
            prev_frame_mono = now_mono
            if alert_manager.relay_locked:
                # Monitoring without a pre-drive release (--force-phase
                # monitoring): the starter stays inhibited. Nothing here can
                # release it.
                banner, banner_color = ("MONITORING  |  starter LOCKED - no pre-drive "
                                        "assessment passed"), (0, 0, 255)
            else:
                banner, banner_color = "MONITORING  |  relay never engages", (0, 200, 0)
            if mismatch is not None:
                banner, banner_color = "MONITORING  |  DRIVER CHANGED - NOT ASSESSED", RED
                # First status line on every frame for the rest of the session.
                display_text(frame, f"DRIVER CHANGED - driver {mismatch['driver_id']} NOT "
                             f"ASSESSED (pre-drive: driver {mismatch['expected_driver_id']})", RED)
                if mismatch_fault is not None:
                    with prof.section("api"):
                        mismatch_fault.refresh_elapsed(api_client, now_mono)

            # Landmarks. No face never lowers the level and never raises it to
            # DANGER: the pipeline's NoFacePolicy holds, latches or faults.
            with prof.section("landmarks"):
                landmarks, _rect = extractor.extract(frame)
                # Invalid landmarks are handled exactly like no face (2026-10-15).
                problem = landmark_problem(landmarks, frame.shape)
            if problem is not None:
                with prof.section("pipeline"):
                    verdict = pipeline.note_no_face(now, now_mono)
                if verdict.gap_s > config.NO_FACE_FAULT_S:
                    identify_generation += 1      # answers from before this are void
                _frs_debug.log_unscored(now, "monitoring", False, verdict.level,
                                        f"{unscored_reason(problem)}:{verdict.band}")
                if verdict.level != last_level:
                    logger.info("Level %s -> %s (no face %.1fs, %s)", last_level or "(none)",
                                verdict.level, verdict.gap_s, verdict.band)
                last_level = verdict.level
                with prof.section("gpio"):
                    alert_manager.set_alert_level(verdict.level)
                if verdict.event in FAULT_TYPE_FOR_EVENT:
                    open_fault = OpenFault(
                        fault_uuid=str(uuid.uuid4()),
                        fault_type=FAULT_TYPE_FOR_EVENT[verdict.event],
                        driver_id=current_driver_id,
                        entry_level=verdict.entry_level,
                        started_at=datetime.now(timezone.utc) - timedelta(seconds=verdict.gap_s),
                        last_known=last_m.last_known() if last_m is not None else None,
                        gap_s=verdict.gap_s,
                        last_push_mono=now_mono,
                        opened_mono=now_mono - verdict.gap_s,
                    )
                    with prof.section("api"):
                        open_fault.push(api_client, "open", now_mono)
                elif open_fault is not None:
                    open_fault.gap_s = verdict.gap_s
                    if now_mono - open_fault.last_push_mono >= MONITORING_FAULT_REFRESH_S:
                        with prof.section("api"):
                            open_fault.push(api_client, "open", now_mono)
                if seed_fault is not None:
                    with prof.section("api"):
                        seed_fault.refresh_elapsed(api_client, now_mono)
                with prof.section("display"):
                    _screen.face, _screen.face_detail = FACE_NONE, f"{verdict.gap_s:.1f}s"
                    _screen.level = verdict.level
                    if driver:
                        _screen.driver_name = str(driver["name"])
                    display_text(frame, f"NO FACE {verdict.gap_s:.1f}s  [{verdict.band}]",
                                 LEVEL_BGR.get(verdict.level, WHITE))
                    draw_banner(frame, banner, banner_color)
                    if debug_pose:
                        draw_pose_debug(frame, pipeline.pose_debounce, now_mono, fps)
                present(frame)   # times its own heartbeat (api) and imshow (display)
                continue

            # First face after a fault resolves it. Done here rather than from
            # m.gap_ended: a driver change below resets the pipeline first.
            fault_resolved = open_fault is not None
            if open_fault is not None:
                resolved = datetime.now(timezone.utc)
                logger.info("Face re-acquired after %.1fs - %s fault resolved",
                            now_mono - open_fault.opened_mono, open_fault.fault_type)
                with prof.section("api"):
                    open_fault.push(api_client, "resolved", now_mono, resolved, "face_reacquired")
                open_fault = None

            # Identify - until the driver is recognised, then never again this
            # session (see docstring). Submitted to the worker on a face frame,
            # answered on a later one; inline only if the worker is unavailable.
            identified = None
            if identify_pending:
                channel = recognition_channel(inline_recognition)
                with prof.section("recognition"):
                    identified = channel.poll()
                    if (identified is None and now_mono >= next_identify
                            and not channel.busy
                            and channel.submit(frame, ("monitoring", identify_owner,
                                                       identify_generation))):
                        identify_attempts += 1
                        if channel.blocking:
                            # Fallback: identify() ran here. Re-stamp the frame so
                            # recognition time is never counted as eye-closure
                            # time: the pipeline sees the gap and abandons any
                            # closure in progress (config.MAX_FRAME_GAP_S).
                            now, now_mono = time.time(), time.monotonic()
                            identified = channel.poll()
                if identified is not None and identified[0] != ("monitoring", identify_owner,
                                                                identify_generation):
                    logger.info("Monitoring: recognition answer %s discarded (%s) - asking "
                                "again", identified[0],
                                "from before a face loss" if identified[0][:2] ==
                                ("monitoring", identify_owner) else "not this run's")
                    identified = None
                    next_identify = float("-inf")
            if identified is not None:
                match = identified[1]
                # Spaced from the answer to this attempt, so the loop always gets
                # UNRECOGNISED_RETRY_S between attempts.
                next_identify = now_mono + UNRECOGNISED_RETRY_S
                if match is None and current_driver_id is not None:
                    # Pre-drive's driver, not (yet) confirmed: a non-match does
                    # not contradict them, so their calibration stays.
                    logger.info("Monitoring: recognition attempt %d - no match; keeping "
                                "pre-drive's driver %s, next attempt in %.0fs",
                                identify_attempts, current_driver_id, UNRECOGNISED_RETRY_S)
                elif match is None:
                    if not defaults_logged:
                        logger.warning("Monitoring: driver not recognised - self-seeding the EAR "
                                       "baseline from this driver's own EAR (overrides only for "
                                       "the first %.0fs of face time); retrying recognition "
                                       "every %.0fs", SELF_SEED_S, UNRECOGNISED_RETRY_S)
                        defaults_logged = True
                    else:
                        logger.info("Monitoring: recognition attempt %d - no match, driver stays "
                                    "UNRECOGNISED; next attempt in %.0fs",
                                    identify_attempts, UNRECOGNISED_RETRY_S)
                elif released_id is not None and int(match["driver_id"]) != released_id:
                    # Someone other than the driver who passed pre-drive: not
                    # assessed. Never scored on the previous driver's
                    # calibration, and nothing is waited for - a fresh
                    # self-seed from this frame on.
                    driver = match
                    current_driver_id = int(match["driver_id"])
                    identify_pending = False
                    identified_after = now_mono - started_mono
                    mismatch = {"expected_driver_id": released_id,
                                "driver_id": current_driver_id,
                                "name": match.get("name"),
                                "at_s": round(now_mono - started_mono, 3)}
                    logger.error("DRIVER MISMATCH: recognised driver %s (%s, confidence %.2f, "
                                 "attempt %d), but pre-drive assessed driver %s - this driver "
                                 "has NOT been assessed. Scoring switches to a self-seeded EAR "
                                 "baseline now; fault raised (as %s).", current_driver_id,
                                 match.get("name"), match["confidence"], identify_attempts,
                                 released_id, FAULT_TYPE_DRIVER_MISMATCH)
                    calibration, thresholds = None, dict(DEFAULT_THRESHOLDS)
                    calibration_fetch = None
                    calibration_source = "self-seeded (driver mismatch)"
                    calibration_after = now_mono - started_mono
                    pipeline.reset()
                    stored_neutral = load_neutral(current_driver_id)
                    if stored_neutral is not None:
                        pipeline.set_pose_neutral(stored_neutral, "stored")
                    if seed_fault is not None:
                        # That self-seed was the previous driver's; this one's
                        # starts over.
                        with prof.section("api"):
                            seed_fault.push(api_client, "resolved", now_mono,
                                            datetime.now(timezone.utc), "driver_identified")
                        seed_fault = None
                    mismatch_fault = OpenFault(
                        fault_uuid=str(uuid.uuid4()),
                        fault_type=FAULT_TYPE_DRIVER_MISMATCH,
                        driver_id=current_driver_id,
                        entry_level=last_level or "ALERT",
                        started_at=datetime.now(timezone.utc),
                        last_known=None,
                        gap_s=0.0,
                        last_push_mono=now_mono,
                        opened_mono=now_mono,
                    )
                    with prof.section("api"):
                        mismatch_fault.push(api_client, "open", now_mono)
                    last_danger_push = float("-inf")
                elif int(match["driver_id"]) == current_driver_id:
                    identify_pending = False
                    logger.info("Monitoring: driver %s confirmed by recognition (attempt %d) - "
                                "pre-drive's calibration kept, recognition stops for this "
                                "session", current_driver_id, identify_attempts)
                else:
                    if current_driver_id is not None:
                        logger.warning("Monitoring: recognised driver %s, not pre-drive's %s - "
                                       "switching driver", match["driver_id"], current_driver_id)
                    driver = match
                    current_driver_id = int(match["driver_id"])
                    identify_pending = False
                    identified_after = now_mono - started_mono
                    logger.info("Driver identified: %s (id=%s, confidence %.2f, attempt %d) - "
                                "recognition stops for this session", match["name"],
                                current_driver_id, match["confidence"], identify_attempts)
                    # A foreign-device calibration IS used here (flagged on every
                    # event), unlike pre-drive where it locks. Decided 2026-09-28:
                    # a real alert-state baseline measured at the wrong camera
                    # angle beats one self-seeded from a driver who may already
                    # be drowsy - that failure is silent and defeats detection
                    # for exactly the drivers who need it. The angle error is
                    # unquantified (a stated limitation). Self-seed only when
                    # there is no calibration, or (a flagged repair) when the
                    # record has no EAR baseline.
                    cached = _calibration_cache.get(current_driver_id)
                    if cached is not None:
                        apply_calibration(cached, "cached from pre-drive", now_mono)
                    else:
                        # Off the frame loop (2026-10-19): the GET can take up to
                        # API_TIMEOUT, or not come back at all. Until it answers
                        # the current values stay in use.
                        calibration_fetch = CalibrationFetch(api_client, current_driver_id)
                        logger.info("Monitoring: fetching driver %s's calibration off the frame "
                                    "loop - scoring continues on %s until it arrives",
                                    current_driver_id,
                                    "the current calibration" if calibration is not None
                                    else "the self-seeded EAR baseline")

            # The calibration fetch, if one is out: applied on the first face
            # frame after it answers; never waited for.
            if calibration_fetch is not None:
                fetched = calibration_fetch.poll()
                if fetched is not None:
                    fetch, calibration_fetch = calibration_fetch, None
                    waited = now_mono - fetch.started
                    if fetch.driver_id != current_driver_id:
                        logger.info("Monitoring: calibration for driver %s arrived after a "
                                    "driver change - discarded", fetch.driver_id)
                    elif fetched[0] is not None:
                        apply_calibration(fetched[0], "fetched in %.1fs, off the frame loop"
                                          % waited, now_mono)
                    else:
                        logger.warning("Monitoring: no usable calibration for driver %s after "
                                       "%.1fs (backend unreachable, no record, or unusable) - "
                                       "keeps running on %s; re-enrol with `main.py --enroll` "
                                       "if this persists", current_driver_id, waited,
                                       "the current calibration" if calibration is not None
                                       else "the self-seeded EAR baseline")
                        calibration_source = calibration_source or "none (fetch failed)"
                        if seed_fault is not None:
                            # The self-seed had given up: it starts over for
                            # this driver, as on identification before.
                            pipeline.reset()
                            logger.info("Driver identified - no_ear_baseline fault resolved "
                                        "(self-seed restarts)")
                            with prof.section("api"):
                                seed_fault.push(api_client, "resolved", now_mono,
                                                datetime.now(timezone.utc), "driver_identified")
                            seed_fault = None

            with prof.section("pipeline"):
                m = pipeline.process(landmarks, thresholds, now, now_mono,
                                     self_seed=calibration is None or calibration.self_seed_ear)
            _frs_debug.log_frame(m, "monitoring", pipeline.pose_neutral_source)
            last_m = m
            if m.seed_failed and seed_fault is None:
                seed_fault = OpenFault(
                    fault_uuid=str(uuid.uuid4()),
                    fault_type=FAULT_TYPE_NO_EAR_BASELINE,
                    driver_id=current_driver_id,
                    entry_level=last_level or "ALERT",
                    started_at=datetime.now(timezone.utc),
                    last_known=None,     # nothing was ever scored
                    gap_s=0.0,
                    last_push_mono=now_mono,
                    opened_mono=now_mono,
                )
                with prof.section("api"):
                    seed_fault.push(api_client, "open", now_mono)
            elif seed_fault is not None:
                with prof.section("api"):
                    seed_fault.refresh_elapsed(api_client, now_mono)
            # (A gap that opened a fault was already logged above when it was resolved.)
            if m.gap_ended is not None and m.gap_ended.face_lost and not fault_resolved:
                logger.info("Face re-acquired after %.1fs (%s from %s)%s",
                            m.gap_ended.gap_s, m.gap_ended.band, m.gap_ended.entry_level,
                            " - DANGER latched until %.0fs below DANGER"
                            % pipeline.no_face.latch.release_hold if m.no_face_latch else "")

            # Log the full FRS breakdown whenever the effective level changes, so
            # the session log shows which term carried the score into the new
            # band rather than just the total.
            if m.level != last_level:
                logger.info(
                    "Level %s -> %s: %s%s%s",
                    last_level or "(none)", m.level,
                    pipeline.frs_calc.format_breakdown(m.frs_result) if m.scored
                    else "not scored (EAR self-seed gave up - no baseline)" if m.seed_failed
                    else "not scored (self-seeding EAR baseline)",
                    f"  [{m.override_reason} override forcing DANGER]" if m.overridden else
                    f"  [{m.level_reason}]",
                    f"  [escalation {m.band}; raw band {m.frs_result['level']}]"
                    if m.scored and not m.overridden and m.band != m.frs_result["level"] else "",
                )
                logger.debug(
                    "Level %s inputs: ear=%.3f (norm %.3f) perclos=%.1f%% (norm %.3f) "
                    "blink_dur=%.0fms (norm %.3f) blink_freq=%.1f (norm %.3f) "
                    "mar=%.3f (norm %.3f) yawn_norm=%.3f",
                    m.level, m.ear, m.ear_norm, m.perclos, m.perclos_norm,
                    m.blink_duration_ms, m.bd_norm, m.blink_freq, m.bf_norm,
                    m.mar, m.mar_norm, m.yawn_norm,
                )
                last_level = m.level

            # Physical alerts - LEDs and buzzer only. m.level is DANGER while the
            # head-pose override is active regardless of the (lower) FRS level.
            with prof.section("gpio"):
                alert_manager.set_alert_level(m.level)

            # Operator notification: DANGER to the backend, rate-limited.
            if m.level == "DANGER" and now_mono - last_danger_push >= DANGER_EVENT_INTERVAL:
                with prof.section("api"):
                    api_client.push_fatigue_event(
                        current_driver_id, m.effective_result(), m.ear, m.perclos,
                        relay_triggered=False, phase=Phase.MONITORING,
                        provenance=(calibration.provenance() if calibration is not None
                                    else SELF_SEEDED_PROVENANCE),
                    )
                last_danger_push = now_mono

            with prof.section("display"):
                draw_overlay(frame, driver, m, fps, pipeline, thresholds, landmarks,
                             calibrated=calibration is not None)
                draw_banner(frame, banner, banner_color)
                if debug_pose:
                    draw_pose_debug(frame, pipeline.pose_debounce, now_mono, fps)
            present(frame)   # times its own heartbeat (api) and imshow (display)

    except SessionEnded:
        # END SESSION stands in for key-OFF (it cannot be sensed), so open
        # faults resolve as "ignition_off" - the backend's existing value.
        end_reason = "session_ended"
        close("ignition_off")
        logger.info("Monitoring: session ended by the operator")
        raise
    except BaseException:
        end_reason = "stopped"         # quit / Ctrl-C / SIGTERM / a crash
        raise
    else:
        end_reason = "ignition_off"
        close("ignition_off")
        logger.info("Monitoring: ignition turned OFF - returning to pre-drive")
    finally:
        summary()


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
        "--driver-id", type=int, default=None, metavar="N",
        help="with --enroll: the driver's DB id (skips the terminal prompt)",
    )
    parser.add_argument(
        "--mock-gpio", action="store_true",
        help="force AlertManager into mock mode even on a Pi (no GPIO is driven)",
    )
    parser.add_argument(
        "--force-phase", choices=[p.value for p in Phase], default=None,
        help="stay in this phase (testing one phase in isolation); in predrive a confirmed "
             "start does NOT continue into monitoring - it returns to IDLE",
    )
    parser.add_argument(
        "--sequence", action="store_true",
        help="full session (the default - there is no ignition input): pre-drive, then - "
             "once the starter is released (pass or approved override) AND the start is "
             "confirmed (BEGIN MONITORING) - monitoring until END SESSION or stop",
    )
    parser.add_argument(
        "--profile-loop", type=float, nargs="?", const=60.0, default=None, metavar="SECONDS",
        help="monitoring: time every frame by category (capture, landmarks, recognition, "
             "pipeline, gpio, api, display) and log a summary every SECONDS (default 60); "
             "per-frame CSV under logs/",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="no window at all - neither the data screen nor video (unattended data "
             "collection). Keys q / r / b / e are then unavailable - use the launcher's "
             "buttons, `kill -USR1` (begin) / `kill -USR2` (end session), or Ctrl-C",
    )
    parser.add_argument(
        "--show-video", action="store_true",
        help="DIAGNOSTIC: show the annotated camera image in the window instead of the data "
             "screen (camera aim / landmark problems). Off by default: the in-vehicle "
             "display does not show the driver's face",
    )
    parser.add_argument(
        "--debug-stream", type=int, nargs="?", const=config.DEBUG_STREAM_PORT, default=None,
        metavar="PORT",
        help="DIAGNOSTIC: serve the annotated camera view as MJPEG on the LAN "
             f"(default port {config.DEBUG_STREAM_PORT}); open the logged URL in a browser. "
             "Never recorded",
    )
    parser.add_argument(
        "--fullscreen", action="store_true",
        help="fullscreen window (the touchscreen launcher passes this)",
    )
    parser.add_argument(
        "--reserve-bottom", type=int, default=0, metavar="PX",
        help="leave PX at the bottom of the data screen empty (the launcher's STOP strip)",
    )
    parser.add_argument(
        "--state-file", default=None, metavar="PATH",
        help="write the session state (IDLE / ... / READY_TO_START / MONITORING) to PATH as "
             "JSON on every change (used by launcher.py for its strip buttons)",
    )
    parser.add_argument(
        "--result-file", default=None, metavar="PATH",
        help="with --enroll: write the outcome as JSON to PATH (used by launcher.py)",
    )
    parser.add_argument(
        "--debug-pose", action="store_true",
        help="overlay the head-pose debounce timers / override state on the preview",
    )
    parser.add_argument(
        "--diag-ear", action="store_true",
        help="TEMPORARY DIAGNOSTIC: log raw EAR, the active closure threshold and the "
             "closed/open decision on every transition. Use when PERCLOS, blink and "
             "microsleep all report nothing - they share this one comparison",
    )
    parser.add_argument(
        "--pose-release-hold", type=float, default=HEAD_POSE_RELEASE_HOLD_S, metavar="SECONDS",
        help="seconds the head pose must be normal before a head-pose alert "
             f"is released (default {HEAD_POSE_RELEASE_HOLD_S})",
    )
    args = parser.parse_args(argv)
    if args.driver_id is not None and not args.enroll:
        parser.error("--driver-id only makes sense with --enroll")
    if args.result_file is not None and not args.enroll:
        parser.error("--result-file only makes sense with --enroll")
    if args.show_video and args.no_preview:
        parser.error("--show-video and --no-preview contradict each other")
    if args.sequence and (args.force_phase or args.enroll):
        parser.error("--sequence cannot be combined with --force-phase or --enroll")
    if args.profile_loop is not None and args.profile_loop <= 0:
        parser.error("--profile-loop SECONDS must be positive")
    if args.state_file is not None and args.enroll:
        parser.error("--state-file does not apply to --enroll")
    return args


def main(argv: Optional[list] = None) -> int:
    """
    Start-up sequence, then enrollment or the phase supervisor loop.

    ``cleanup()`` is guaranteed to run via ``try/finally`` so the camera is
    always released and GPIO reset (the starter is left inhibited -
    fail-secure).

    Returns:
        Process exit code.
    """
    global _camera, _alert_manager, _head_pose, _ignition, _heartbeat, _display_available
    global _display_mode, _fullscreen, _reserve_bottom, _stream, _rate
    global _start_confirm, _state_file, _recognition_factory

    args = parse_args(argv)
    setup_logging()
    if not args.enroll:
        # Before any hardware: an unimplemented confirmation mode must stop
        # the run here, not after the starter could be released.
        _start_confirm = make_start_confirm(config.START_CONFIRM_MODE, _requests)
        _state_file = args.state_file
    _fullscreen, _reserve_bottom = args.fullscreen, max(0, args.reserve_bottom)
    if args.no_preview:
        _display_available = False
        _display_mode = "none"
        logger.info("Display window OFF (--no-preview) - stop with Ctrl-C or the launcher")
    elif args.show_video:
        _display_mode = "video"
        logger.warning("--show-video: the window shows the CAMERA IMAGE (diagnostic). "
                       "The in-vehicle default shows no face.")
    elif args.enroll:
        _display_mode = "enroll"
        logger.info("Display: enrollment screen %dx%d%s (no camera image; position the driver "
                    "with --debug-stream)", config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT,
                    ", fullscreen" if _fullscreen else "")
    else:
        logger.info("Display: data screen %dx%d%s (no camera image; --show-video to diagnose)",
                    config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT,
                    ", fullscreen" if _fullscreen else "")
    logger.info("=== Driver Fatigue Detection System starting ===")
    exit_code = 0
    interrupted = False
    if args.debug_stream is not None:
        try:
            _stream = DebugStream(args.debug_stream, config.DEBUG_STREAM_FPS,
                                  stream_token(os.environ.get("FATIGUE_STREAM_TOKEN")))
            banner = (f"DIAGNOSTIC VIDEO STREAM ON - open {_stream.url} in a browser on the "
                      f"same network ({config.DEBUG_STREAM_FPS:.0f} fps max, LAN only, "
                      f"nothing recorded)")
            logger.warning(banner)
        except OSError as exc:
            logger.error("Debug stream NOT started on port %d: %s - continuing without it",
                         args.debug_stream, exc)

    # SIGTERM (the launcher's escalation, `kill`) would otherwise end the
    # process without running the finally below - leaving the relay pin at
    # its last level. Raise it as an interrupt so cleanup() inhibits the
    # starter.
    try:
        previous_sigterm = signal.signal(signal.SIGTERM, _raise_on_sigterm)
    except ValueError:  # not the main thread (tests)
        previous_sigterm = None
    # SIGHUP (the desktop session ending) the same way, where it exists.
    previous_sighup = None
    if hasattr(signal, "SIGHUP"):
        try:
            previous_sighup = signal.signal(signal.SIGHUP, _raise_on_sigterm)
        except ValueError:
            pass
    # The launcher's BEGIN MONITORING / START NEXT DRIVER (SIGUSR1) and END
    # SESSION (SIGUSR2). Installed before the first state is published, so
    # the launcher never signals a child that would die of SIGUSR1.
    request_signals = [] if args.enroll else _install_request_signals()

    try:
        # 0. Alerts first (auto-detects GPIO): the relay is driven to INHIBIT
        #    before anything else runs. The phase is set once the ignition
        #    is read (step 5).
        _alert_manager = AlertManager(mock=args.mock_gpio, phase=Phase.PREDRIVE)

        # 1. Config is imported at module load; log the key values.
        logger.info("Config: device %s, camera %dx%d@%dfps, API %s, calibration %ds, "
                    "pre-drive rolling %ds minimum (worst %.0fs window), start confirmation "
                    "%s, no ignition input",
                    config.DEVICE_ID, config.CAMERA_WIDTH, config.CAMERA_HEIGHT,
                    config.CAMERA_FPS, config.API_BASE_URL, config.CALIBRATION_DURATION,
                    config.PREDRIVE_ASSESSMENT_SECONDS, config.PREDRIVE_WORST_WINDOW_SECONDS,
                    config.START_CONFIRM_MODE)

        # 2. Backend
        api_client = APIClient()
        if not api_client.ping():
            logger.warning("Laravel backend is offline - continuing; pre-drive will lock "
                           "(no baseline) and monitoring will use defaults")

        # 3. Landmarks
        if not config.LANDMARK_MODEL.exists():
            logger.error("Landmark model not found at %s (see README.md)", config.LANDMARK_MODEL)
            exit_code = _enroll_result(1, "error", f"Landmark model not found at "
                                                   f"{config.LANDMARK_MODEL}. Nothing was captured.")
            return exit_code
        extractor = LandmarkExtractor(str(config.LANDMARK_MODEL), config.SCALE_FACTOR)

        # 4. Driver recognition
        recognizer = DriverRecognizer(api_client)
        loaded = recognizer.load_encodings()
        if loaded == 0 and not args.enroll:
            logger.warning("No driver encodings loaded - every driver will be UNKNOWN. "
                           "Run `python main.py --enroll` first.")
        if not args.enroll:
            # Recognition in its own process for pre-drive re-identification
            # and monitoring (dlib holds the GIL, so a thread would still stall
            # the frame loop). Stopped on END SESSION, restarted for the next
            # driver, stopped by cleanup() on every exit.
            _recognition_factory = KnownFaces.of(recognizer)
            start_recognition()

        # 5. Phase. There is no ignition input (TODO: ignition sense, deferred
        #    hardware), so the phase is pinned: --force-phase, else pre-drive
        #    for good - monitoring (entered from a confirmed start) never sees
        #    an ON -> OFF edge and runs until END SESSION or stop. Without
        #    flags this is the same as --sequence.
        if args.force_phase:
            _ignition = ForcedIgnition(Phase(args.force_phase))
        else:
            _ignition = ForcedIgnition(Phase.PREDRIVE,
                                       source="--sequence" if args.sequence
                                       else "default (no ignition input)")
        initial_phase = _ignition.phase()

        # 6. Alerts were created in step 0 (relay inhibited); set the phase.
        _alert_manager.set_phase(initial_phase)

        # 6b. Head pose (camera intrinsics derived from the frame size)
        _head_pose = HeadPoseEstimator(
            frame_width=config.CAMERA_WIDTH, frame_height=config.CAMERA_HEIGHT
        )

        # 7. Camera
        _camera = Camera()

        if args.enroll:
            _rate = LoopRate()
            exit_code = run_enrollment(api_client, extractor, driver_id=args.driver_id)
        else:
            rate = _rate = LoopRate()
            # Heartbeat to the portal, ticked from present() in every phase
            # loop. Reads the relay state from _alert_manager at send time;
            # with --debug-stream it also keeps the portal's live view listed.
            _heartbeat = Heartbeat(api_client, _alert_manager, stream=_stream)
            logger.info("Supervisor started in %s - %s to stop "
                        "(heartbeat every %.0fs, firmware %s)",
                        initial_phase.value,
                        "press 'q' in the window or Ctrl-C" if _display_available else "Ctrl-C",
                        config.HEARTBEAT_INTERVAL_SECONDS,
                        config.FIRMWARE_VERSION)
            # 8. Session supervisor. A pre-drive that releases the starter
            #    (pass or approved override) waits in READY_TO_START for the
            #    start confirmation and only then goes into monitoring -
            #    except under --force-phase predrive, which tests that phase
            #    alone (back to IDLE). Every pre-drive starts by re-inhibiting
            #    the starter. END SESSION, from any state, ends up here: the
            #    starter is inhibited, alerts stop and the unit waits in IDLE
            #    for the next driver - the process keeps running.
            continue_to_monitoring = args.force_phase is None
            idle_first = False
            while True:
                try:
                    if idle_first:
                        idle_first = False
                        run_idle(rate)
                        start_recognition()
                    if _ignition.phase() is Phase.PREDRIVE:
                        released = run_predrive_assessment(
                            api_client, extractor, recognizer, _alert_manager, _head_pose,
                            _ignition, rate, debug_pose=args.debug_pose,
                            diag_ear=args.diag_ear, pose_release_hold=args.pose_release_hold,
                            continue_to_monitoring=continue_to_monitoring,
                        )
                        if not released:
                            # 'r' (re-run) or a change of driver: a fresh
                            # pre-drive, starter still inhibited.
                            continue
                        if not continue_to_monitoring:
                            logger.info("--force-phase predrive: start confirmed - monitoring "
                                        "is not entered (phase tested alone); back to IDLE")
                            _alert_manager.lock_relay()
                            _alert_manager.idle()
                            idle_first = True
                            continue
                        run_monitoring(
                            api_client, extractor, recognizer, _alert_manager, _head_pose,
                            _ignition, rate, debug_pose=args.debug_pose,
                            diag_ear=args.diag_ear, pose_release_hold=args.pose_release_hold,
                            from_release=True, profile_s=args.profile_loop,
                        )
                    else:
                        run_monitoring(
                            api_client, extractor, recognizer, _alert_manager, _head_pose,
                            _ignition, rate, debug_pose=args.debug_pose,
                            diag_ear=args.diag_ear, pose_release_hold=args.pose_release_hold,
                            profile_s=args.profile_loop,
                        )
                except SessionEnded:
                    end_session(_alert_manager)
                    idle_first = True

    except QuitRequested:
        logger.info("Quit requested from the preview window")
        interrupted = True
    except KeyboardInterrupt:
        logger.info("Interrupted (Ctrl-C / SIGINT / SIGTERM / SIGHUP)")
        interrupted = True
    except Exception:
        logger.exception("Fatal error in main loop")
        exit_code = 1
    finally:
        cleanup()
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        if previous_sighup is not None:
            signal.signal(signal.SIGHUP, previous_sighup)
        for signum, previous in request_signals:
            signal.signal(signum, previous)
        if args.enroll and args.result_file:
            write_enroll_result(args.result_file, args.driver_id, exit_code, interrupted)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
