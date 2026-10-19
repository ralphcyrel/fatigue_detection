#!/bin/bash
# Start the touchscreen launcher at login - the XDG autostart entry
# (deploy/fatigue-launcher.desktop, installed into ~/.config/autostart/ by
# deploy/install_autostart.sh) runs this. Raspberry Pi OS Bookworm (labwc,
# Wayfire or X11 desktop - all of them run ~/.config/autostart entries).
#
# What it does, in order:
#   1. derives every path from its own location (no username, no /home/pi);
#   2. sends its own output and launcher.py's stdout + stderr to
#      logs/launcher_autostart.log (rotated at 5 MB);
#   3. refuses to start a second copy (lock on logs/.launcher_autostart.lock,
#      held for as long as the launcher runs);
#   4. activates the project venv at ./venv;
#   5. loads the FATIGUE_* backend settings the same way as
#      deploy/start_launcher.sh (see there for why ~/.bashrc is not sourced);
#   6. waits for the display - Tk must be able to open it and xrandr must list
#      a monitor - retrying every DELAY_S seconds instead of failing on a
#      cold boot; gives up (exit 1) after DISPLAY_TRIES attempts;
#   7. waits for the camera to be listed (rpicam-hello / libcamera-hello)
#      the same way, but only warns if it never shows up: the launcher itself
#      does not use the camera, and each main.py session reports a missing one;
#   8. execs launcher.py (fullscreen kiosk; extra arguments are passed on).
#
# It changes nothing on the system: no boot configuration, no autologin, no
# systemd unit. Undo with deploy/install_autostart.sh's printed command.
#
# Tunables (environment): FATIGUE_AUTOSTART_DISPLAY_TRIES (30),
# FATIGUE_AUTOSTART_CAMERA_TRIES (10), FATIGUE_AUTOSTART_DELAY_S (2).

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT/logs"
LOG="$LOG_DIR/launcher_autostart.log"

DISPLAY_TRIES="${FATIGUE_AUTOSTART_DISPLAY_TRIES:-30}"
CAMERA_TRIES="${FATIGUE_AUTOSTART_CAMERA_TRIES:-10}"
DELAY_S="${FATIGUE_AUTOSTART_DELAY_S:-2}"

mkdir -p "$LOG_DIR" || exit 1
if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG" 2>/dev/null || echo 0)" -gt 5242880 ]; then
    mv -f "$LOG" "$LOG.1"
fi
# Everything from here on - and launcher.py after the exec - goes to the log.
exec >>"$LOG" 2>&1

log() { echo "$(date '+%F %T') launcher_autostart: $*"; }

log "===== starting: project $PROJECT, user $(id -un), DISPLAY=${DISPLAY:-unset}," \
    "WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-unset}, XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-unset}"

# One launcher only (two would fight over the camera). fd 9 stays open
# through the exec, so the lock lasts as long as launcher.py runs.
exec 9>"$LOG_DIR/.launcher_autostart.lock"
if ! flock -n 9; then
    log "another launcher_autostart.sh is already running - exiting"
    exit 0
fi

if [ ! -f "$PROJECT/venv/bin/activate" ]; then
    log "ERROR: no venv at $PROJECT/venv - create it as described in README.md"
    exit 1
fi
# shellcheck disable=SC1091
. "$PROJECT/venv/bin/activate"
log "python: $(command -v python) ($(python --version 2>&1))"

# Backend settings, as deploy/start_launcher.sh reads them.
SOURCE=""
for f in /etc/fatigue-detection.env "$HOME/.config/fatigue-detection.env"; do
    if [ -r "$f" ]; then
        set -a; . "$f"; set +a
        SOURCE="$f"
        break
    fi
done
if [ -z "$SOURCE" ] && [ -r "$HOME/.bashrc" ]; then
    eval "$(grep -E '^[[:space:]]*export[[:space:]]+FATIGUE_[A-Z_]+=' "$HOME/.bashrc")"
    SOURCE="$HOME/.bashrc (export lines)"
fi
log "settings from: ${SOURCE:-NOTHING FOUND}; FATIGUE_API_BASE_URL=${FATIGUE_API_BASE_URL:-(NOT SET)}" \
    "FATIGUE_API_TOKEN=$([ -n "${FATIGUE_API_TOKEN:-}" ] && echo '(set)' || echo '(NOT SET)')" \
    "FATIGUE_DEVICE_ID=${FATIGUE_DEVICE_ID:-(NOT SET)}"

# Tk (and main.py's cv2 window) talk X: native X11, or Xwayland under a
# Wayland compositor. The autostart entry inherits DISPLAY from the desktop
# session; :0 is the fallback.
export DISPLAY="${DISPLAY:-:0}"
export PYTHONUNBUFFERED=1

display_ready() {
    python -c 'import tkinter; tkinter.Tk().destroy()' 2>/dev/null || return 1
    # The launcher sizes itself to the monitor xrandr reports; wait for one.
    if command -v xrandr >/dev/null 2>&1; then
        xrandr --listactivemonitors 2>/dev/null | grep -qE '^[[:space:]]*[0-9]+:' || return 1
    fi
    return 0
}

ok=""
for i in $(seq 1 "$DISPLAY_TRIES"); do
    if display_ready; then
        log "display $DISPLAY ready (attempt $i/$DISPLAY_TRIES)"
        ok=1
        break
    fi
    log "display $DISPLAY not ready (attempt $i/$DISPLAY_TRIES) - retrying in ${DELAY_S}s"
    sleep "$DELAY_S"
done
if [ -z "$ok" ]; then
    log "ERROR: display $DISPLAY never became usable - giving up (log out and in, or run" \
        "$SCRIPT_DIR/launcher_autostart.sh by hand)"
    exit 1
fi

CAM_TOOL="$(command -v rpicam-hello || command -v libcamera-hello || true)"
if [ -z "$CAM_TOOL" ]; then
    log "no rpicam-hello / libcamera-hello - camera check skipped"
else
    ok=""
    for i in $(seq 1 "$CAMERA_TRIES"); do
        if "$CAM_TOOL" --list-cameras 2>/dev/null | grep -qE '^[[:space:]]*[0-9]+[[:space:]]*:'; then
            log "camera listed by $(basename "$CAM_TOOL") (attempt $i/$CAMERA_TRIES)"
            ok=1
            break
        fi
        log "camera not listed yet (attempt $i/$CAMERA_TRIES) - retrying in ${DELAY_S}s"
        sleep "$DELAY_S"
    done
    [ -n "$ok" ] || log "WARNING: no camera found - starting the launcher anyway; sessions" \
                        "will fail until the camera is connected"
fi

cd "$PROJECT" || exit 1
log "exec launcher.py $*"
exec python launcher.py "$@"
