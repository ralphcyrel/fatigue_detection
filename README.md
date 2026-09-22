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

The ignition state (a fuse tap on an ignition-switched circuit, read on GPIO 25)
selects the phase:

| Phase | Ignition | What happens | Starter relay |
|---|---|---|---|
| **Enrollment** (`--enroll`) | — | Operator-supervised, at hiring: face encoding + 60 s alert-state calibration → backend. | untouched |
| **Pre-drive assessment** | OFF | Recognise the driver (≤ 10 s, 3 consistent matches) → fetch *their* thresholds (no defaults) → 30 s assessment through the shared pipeline. Verdict = worst 5 s rolling mean of FRS `< 0.40`. | **Only phase that drives it.** Starts inhibited; released on PASS or an approved operator override. |
| **Continuous monitoring** | ON | Same pipeline; LEDs, buzzer and `POST /fatigue-events` only. Unrecognised driver / no baseline → generic defaults (degraded monitoring beats none). | **Never engaged.** `AlertManager.lock_relay()` refuses with a WARNING in this phase. Entering the phase does not release it either (key-ON precedes cranking). |

Three lock reasons keep the starter inhibited after pre-drive — `fatigue_detected`,
`driver_not_recognized`, `no_baseline` — and all three resolve through one operator
override request (`POST /override-requests`, polled every 5 s). Every ignition-OFF
returns to pre-drive and re-locks; a pass is not carried across ignition cycles.

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
| `modules/assessment.py` | Pre-drive scoring: worst-5 s-window aggregate, per-frame CSV/JSON. |
| `modules/phase.py` | `Phase` and `LockReason` enums. |
| `modules/ignition.py` | Debounced ignition sense input (mock / real GPIO, same pattern as `alert.py`). |
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
├── deploy/           # systemd unit + install notes for the touchscreen launcher
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
  Wired **normally-open** (fail-secure): the coil must be energised (pin HIGH) to allow a
  start, so an unpowered Pi, a crash, or `GPIO.cleanup()` all leave the starter inhibited.
- **Ignition sense (GPIO 25)** — fuse tap on an ignition-switched circuit through an
  optocoupler (or a divider clamped to 3.3 V — never 12 V straight to the pin). Pulled
  down, so a disconnected tap reads OFF (pre-drive). Debounced 0.3 s in software.
- Set `FATIGUE_DEVICE_ID` per vehicle; it keys assessments, override requests and the
  heartbeat. Bump `FIRMWARE_VERSION` in `config/config.py` on release; the portal shows it.

## How to run

```bash
python main.py                                   # real ignition + GPIO
python main.py --mock-gpio                       # no hardware: 'i' toggles mock ignition
python main.py --mock-gpio --force-phase predrive     # stay in pre-drive ('r' re-runs)
python main.py --mock-gpio --force-phase monitoring   # stay in monitoring
python main.py --debug-pose                      # overlay head-pose debounce timers
python main.py --enroll                          # enrol a driver (prompts for the id)
python main.py --enroll --driver-id 7            # enrol driver 7, no prompt
```

Preview-window keys: `q` quit · `i` toggle mock ignition · `r` re-run the pre-drive
assessment (after a pass or a denied override).

### Touchscreen launcher (no keyboard)

`launcher.py` is a Tkinter menu for demoing the unit with only the touchscreen
attached. It shells out to `main.py` with the flags above, so the CLI stays the
single source of truth:

| Button | Runs |
|---|---|
| Enroll driver | picks a driver from `GET /drivers` (the one assigned to this `FATIGUE_DEVICE_ID` is listed first), then `main.py --enroll --driver-id N` |
| Pre-drive assessment | `main.py --force-phase predrive` |
| Continuous monitoring | `main.py --force-phase monitoring` |
| Follow ignition | `main.py` (real ignition input selects the phase) |

The header shows the device id and whether the backend answers `/ping`. While a
session runs the launcher shrinks to a bottom strip with a **STOP** button
(sends SIGINT = Ctrl-C); when `main.py` exits the menu returns with the result.
Layout scales with the screen, from the 480×320 panel up to an HDMI monitor.

```bash
python launcher.py               # fullscreen
python launcher.py --windowed    # development: 480x320 window
```

To start it on boot see [`deploy/README.md`](deploy/README.md) (systemd unit,
plus how to disable it again for CLI development).

## Backend endpoints used

| Method | Path | Purpose |
|---|---|---|
| GET | `/drivers` | driver roster for the launcher's picker (`id`, `full_name`, `device_id`, `is_enrolled`) |
| GET | `/drivers/encodings` | face encodings for recognition |
| GET | `/drivers/{id}/thresholds` | per-driver baselines (incl. `mar_baseline`, `yawn_threshold`) |
| POST | `/drivers/{id}/enroll` | encoding + baselines |
| POST | `/fatigue-events` | monitoring-phase DANGER notification (`phase`, `ignition_on`, `device_id`) |
| POST | `/assessments` | pre-drive verdict, aggregates and full per-frame `samples` |
| POST | `/override-requests` | starter stays inhibited; `reason`, `driver_id` (nullable), `assessment_id` |
| GET | `/override-requests/{id}` | `status`: `pending` / `approved` / `denied` |
| GET | `/ping` | reachability |
| POST | `/devices/{id}/heartbeat` | every 30 s in both phases: `confirmed_state` (relay as actually driven: `normal` / `interrupted` / `unknown`), `firmware_version`; response `commanded_state` is logged at debug only |
