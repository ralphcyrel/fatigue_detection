# Driver Fatigue Detection System

Camera-based, real-time driver fatigue detection running on embedded hardware.

| | |
|---|---|
| **Compute** | Raspberry Pi 4B (8 GB) |
| **Camera** | Raspberry Pi NOIR Camera Module 3 (no IR filter, works with IR illumination at night) |
| **OS / Runtime** | Raspberry Pi OS Bookworm, Python 3.9+ |
| **Backend** | Laravel REST API (driver records, face encodings, fatigue events) |

## Purpose

The system watches the driver's face and continuously computes several eye-based
drowsiness indicators:

- **EAR** — Eye Aspect Ratio, a per-frame measure of how open the eyes are
- **Blink duration** — how long each blink lasts (long blinks = microsleeps)
- **Blink frequency** — blinks per minute (drops sharply with fatigue)
- **PERCLOS** — percentage of time the eyes are closed over a 60 s window
- **MAR / yawns** — Mouth Aspect Ratio; a sustained wide opening is a yawn
- **Head pose** — a sustained nod / look-away / tilt overrides the level to DANGER
- **FRS** — Fatigue Risk Score, a weighted combination of the above

## Operating phases

There is **no ignition input** (2026-10-19). The Pi's only connection to the
starter circuit is the relay control on GPIO 17: SMPS +5 V feeds the key
switch's BAT terminal, START feeds relay COM and relay NO feeds the motor, so
the motor runs only while the key is held at START *and* the relay is
energised. The ignition-position sense (GPIO 25) and a relay-feedback divider
(GPIO 5) are deferred hardware work; no code reads either pin.

| Phase | What happens | Starter relay |
|---|---|---|
| **Enrollment** (`--enroll`) | Operator-supervised, at hiring: face encoding + a 4 s closed-eye capture (the landmarks must show the eyelid closing: closed/open EAR ≤ 0.60) + 60 s alert-state calibration → backend. | untouched |
| **Pre-drive assessment** | Recognise the driver (≤ 10 s, 3 consistent matches) → fetch *their* calibration captured on this unit (none → lock `no_baseline`; only another unit's → lock `foreign_device_baseline`; never another driver's, never defaults) → settle 3 s → **rolling** assessment through the shared pipeline, **30 s minimum**, no maximum. At 30 s the first 30 s are judged as before (worst 5 s mean FRS `< 0.40`, ≥ 80 % valid frames, no microsleep); a fail is a hard failure. After that the same rule is applied to the most recent 30 s on every frame. | **Only phase that drives it.** Starts inhibited. Released in **READY_TO_START** (rolling window passes, face seen within 2 s, same driver) or on an approved operator override; inhibited again as soon as READY_TO_START is lost. |
| **Continuous monitoring** | Entered **only** on the start confirmation (`START_CONFIRM_MODE = "touch"`: BEGIN MONITORING on the launcher strip), never on a timeout. Same pipeline; LEDs, buzzer and `POST /fatigue-events` only. Unrecognised driver / no usable calibration → EAR baseline self-seeded from the driver's own EAR over the first 5 s (head-pose and microsleep overrides only until then); other baselines generic. After 6 discarded seeds the level is FAULT and a `no_ear_baseline` monitoring fault is raised. | **Never engaged.** `AlertManager.lock_relay()` refuses with a WARNING in this phase. |

Session states (published to the launcher with `main.py --state-file`):

```
IDLE ──START NEXT DRIVER──▶ IDENTIFYING ─▶ SETTLING ─▶ ASSESSING ◀──────────────┐
                                 │             │          │ ≥ 30 s and the last   │ window fails / no face
                                 ▼             ▼          ▼ 30 s pass             │ > 2 s / driver changes
                              LOCKED ◀─── hard failure ── READY_TO_START ─────────┘
                    (override approved ─▶ READY_TO_START)      │ BEGIN MONITORING
                                                               ▼
                                                           MONITORING
END SESSION (any state) ─▶ starter inhibited, alerts off ─▶ IDLE   (the process keeps running)
```

Hard failures keep the starter inhibited and raise one operator override
request (`POST /override-requests`, polled every 5 s); they clear only by an
approval (→ READY_TO_START, still waiting for BEGIN MONITORING), `r`, or END
SESSION. Priority: `microsleep_detected` > invalid assessment (sent as
`driver_not_recognized`) > `fatigue_detected`. A confirmed microsleep is a hard
failure at any point, also after the first 30 s. A change of driver (another
enrolled driver matched 3 times by the re-identification every
`PREDRIVE_REIDENTIFY_S`, or after a face loss) restarts pre-drive for the new
driver.

The READY_TO_START entries / losses and the start confirmation are logged
separately (`READY_TO_START at …`, `START CONFIRMED by … - pre-drive duration
…`) and recorded in the assessment JSON (`timeline`, `first_ready_at`,
`confirmed_at`, `predrive_duration_s`) for reporting each trial's real
pre-drive duration.

Every assessment writes its full per-frame series to `logs/assessments/<device>_<driver>_<UTC>.csv`
(+ a `.json` summary) and sends the same series in `POST /assessments`.

## Modules

| Module | Description |
|---|---|
| `modules/landmark.py` | Detects the largest face with dlib HOG and extracts 68 facial landmarks (downscaled detection, full-res prediction). |
| `modules/face_recognition_module.py` | Identifies the driver by matching the live face against encodings fetched from the API. |
| `modules/calibration.py` | Learns the driver's personal baseline EAR / blink statistics over the first 60 s of a session. |
| `modules/ear.py` | Computes the Eye Aspect Ratio from the six eye landmarks per eye. |
| `modules/blink.py` | Detects blinks from the EAR time series and measures their duration and frequency. |
| `modules/perclos.py` | Tracks the proportion of eye-closed frames over a sliding 60 s window. |
| `modules/mar.py` | Mouth Aspect Ratio + duration-gated yawn detector (2× baseline for ≥ 1.5 s). |
| `modules/head_pose.py` | Pitch / yaw / roll from the landmarks via solvePnP. |
| `modules/frs.py` | Combines EAR, blink duration, blink frequency, PERCLOS (+ additive yawn term) into a single Fatigue Risk Score. |
| `modules/pipeline.py` | **Shared per-frame pipeline** (`MetricsPipeline`) used by both phases, plus the head-pose debounce. |
| `modules/assessment.py` | Pre-drive scoring: worst-5 s-window aggregate on the first 30 s and on the rolling last 30 s, per-frame CSV/JSON, READY / confirmation timeline. |
| `modules/phase.py` | `Phase`, `SessionState` and `LockReason` enums. |
| `modules/start_confirm.py` | Operator requests (BEGIN / END SESSION) and the start confirmation behind `START_CONFIRM_MODE`. |
| `modules/ignition.py` | Pinned phase source (`ForcedIgnition`). The GPIO 25 ignition sense was removed (deferred hardware). |
| `modules/alert.py` | Drives the GPIO LEDs and buzzer from the alert level, and the **starter-inhibit relay** from the pre-drive verdict only. |
| `modules/api.py` | HTTP client for the Laravel backend (encodings, thresholds, enrollment, fatigue events, assessments, override requests). |

Shared constants live in `config/config.py`.

## Project layout

```
fatigue-detection/
├── modules/          # detection pipeline modules (see table above)
├── config/config.py  # central configuration
├── data/             # landmark model + calibration data (git-ignored contents)
├── logs/             # runtime logs
├── models/           # additional trained models
├── deploy/           # autostart wrapper + .desktop entry + install notes for the launcher
├── main.py           # entry point (CLI)
├── launcher.py       # touchscreen menu that shells out to main.py
├── requirements.txt
└── README.md
```

## Setup

1. Install system build dependencies (needed to compile dlib on the Pi):

   ```bash
   sudo apt update && sudo apt install -y build-essential cmake libopenblas-dev liblapack-dev python3-picamera2
   ```

2. Create and activate a virtual environment. `--system-site-packages` lets the
   venv see the apt-installed `picamera2`/`libcamera` bindings:

   ```bash
   python3 -m venv --system-site-packages venv
   source venv/bin/activate
   ```

3. Install Python dependencies (dlib will compile from source — allow 30–60 min):

   ```bash
   pip install -r requirements.txt
   ```

4. Download the dlib 68-point landmark model into `data/`:

   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   mv shape_predictor_68_face_landmarks.dat data/
   ```

5. Point the system at your Laravel backend via environment variables:

   ```bash
   export FATIGUE_API_BASE_URL="http://<server>/api"
   export FATIGUE_API_TOKEN="<bearer token>"
   ```

## Hardware notes

- **Starter relay (GPIO 17)** — in series with the starter solenoid signal, *not* the
  ignition or fuel circuit: it can prevent a start but can never stop a running engine.
  The board is a 5 V optocoupler relay module with a **low-level trigger**, and the
  starter is wired to its **normally-open** contact (fail-secure): pin **LOW** energises
  the coil and allows a start; pin **HIGH** releases it and inhibits the starter (the safe
  default). `main.py` drives the pin HIGH before anything else at start-up and on every
  exit path (including SIGTERM), and the launcher drives it HIGH again after every
  session, so an unpowered Pi, a crash or a killed session all leave the starter inhibited.
  Add `"gpio=17=op,dh"` to `/boot/firmware/config.txt` so GPIO 17 is held HIGH (relay
  off) during boot.
- **No ignition sense, no relay feedback** — the relay control is the Pi's only
  connection to the starter circuit (key BAT ← SMPS +5 V, START → relay COM, relay NO →
  motor). The ignition-position sense (was GPIO 25) and a relay-contact feedback divider
  (GPIO 5) are deferred hardware for next semester; nothing reads either pin. When the
  ignition sense exists it replaces the touch confirmation (`START_CONFIRM_MODE = "gpio"`,
  which today stops `main.py` at startup with `NotImplementedError`).
- Set `FATIGUE_DEVICE_ID` per vehicle; it keys assessments, override requests and the
  heartbeat. Bump `FIRMWARE_VERSION` in `config/config.py` on release; the portal shows it.

## How to run

```bash
python main.py                                   # pre-drive -> (BEGIN) monitoring; same as --sequence
python main.py --mock-gpio                       # the same, no GPIO driven
python main.py --mock-gpio --force-phase predrive     # pre-drive only; a confirmed start -> IDLE
python main.py --mock-gpio --force-phase monitoring   # stay in monitoring
python main.py --debug-pose                      # overlay head-pose debounce timers
python main.py --enroll                          # enrol a driver (prompts for the id)
python main.py --enroll --driver-id 7            # enrol driver 7, no prompt
python tools/eye_check.py --label driver7        # where the eye landmarks sit + closed/open EAR ratio
```

Preview-window keys: `q` quit · `r` re-run the pre-drive assessment (in READY_TO_START
or after a denied override) · `b` = BEGIN MONITORING / START NEXT DRIVER · `e` = END
SESSION. Without a window (`--no-preview`): `kill -USR1 <pid>` (begin) and
`kill -USR2 <pid>` (end session), which is what the launcher's buttons send.

### Touchscreen launcher (no keyboard)

`launcher.py` is a Tkinter menu for demoing the unit with only the touchscreen
attached. It shells out to `main.py` with the flags above, so the CLI stays the
single source of truth:

| Button | Runs |
|---|---|
| Enroll driver | picks a driver from `GET /drivers` (the one assigned to this `FATIGUE_DEVICE_ID` is listed first), then `main.py --enroll --driver-id N` |
| Pre-drive assessment → monitoring | `main.py --sequence --state-file logs/session_state.json` |
| Monitoring only (test) | `main.py --force-phase monitoring --state-file logs/session_state.json` |

The header shows the device id and whether the backend answers `/ping`. While a
session runs a separate bottom strip carries a **STOP** button (sends SIGINT =
Ctrl-C) and, for pre-drive / monitoring sessions, the session controls read from
the state `main.py` publishes: **BEGIN MONITORING** only in READY_TO_START,
**START NEXT DRIVER** only in IDLE (SIGUSR1), and **END SESSION** (SIGUSR2) behind
a confirmation — CANCEL appears where END SESSION was and YES where STOP was, and
the question cancels itself after 6 s. END SESSION leaves `main.py` running in
IDLE with the starter inhibited and the LEDs / buzzer off. When `main.py` exits
(STOP, or a crash) the launcher drives the relay to INHIBIT and the menu returns
with the result, back at the
launcher's startup geometry (re-asserted, then checked; see
[`deploy/README.md`](deploy/README.md) for the log lines). Layout is sized for
the 800×480 HDMI touchscreen and scales up to a monitor; it fits the monitor the
window is on (not the whole X / VNC desktop).

```bash
python launcher.py                        # fullscreen
python launcher.py --windowed             # development: 800x480 window
python launcher.py --screen-size 800x480  # exactly 800x480, borderless at 0,0:
                                          # check the panel layout on a desktop
```

To start it automatically when the Pi powers on see [`deploy/README.md`](deploy/README.md)
(XDG autostart: `bash deploy/install_autostart.sh`; also the desktop icon and how to go
back to the CLI).

## Tests

`tests/` runs the phase loops, the enrollment flow, the display and the debug
stream against fakes - no camera, GPIO (forced to mock even on a Pi), backend
or `logs/` writes. The slow tests run the real-time loops (~2.5 min in all).

```bash
pip install -r requirements-dev.txt
python -m pytest                    # everything
python -m pytest -m "not slow"      # quick checks only (~10 s)
```

## Backend endpoints used

| Method | Path | Purpose |
|---|---|---|
| GET | `/drivers` | driver roster for the launcher's picker (`id`, `full_name`, `device_id`, `is_enrolled`) |
| GET | `/drivers/encodings` | face encodings for recognition |
| GET | `/drivers/{id}/thresholds` | per-driver baselines (incl. `mar_baseline`, `yawn_threshold`) |
| POST | `/drivers/{id}/enroll` | encoding + baselines (incl. `ear_closed_baseline`, see `deploy/BACKEND_CHANGES_2026-09-28.md` item 6) |
| POST | `/fatigue-events` | monitoring-phase DANGER notification (`phase`, `ignition_on`, `device_id`) |
| POST | `/assessments` | pre-drive verdict, aggregates and full per-frame `samples` |
| POST | `/override-requests` | starter stays inhibited; `reason`, `driver_id` (nullable), `assessment_id` |
| GET | `/override-requests/{id}` | `status`: `pending` / `approved` / `denied` |
| GET | `/ping` | reachability |
| POST | `/devices/{id}/heartbeat` | every 30 s in both phases: `confirmed_state` (relay as actually driven: `normal` / `interrupted` / `unknown`), `firmware_version`; response `commanded_state` is logged at debug only |
