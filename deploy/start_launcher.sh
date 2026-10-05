#!/bin/bash
# Start the touchscreen launcher from a desktop icon, with the backend
# settings loaded and the project venv's Python.
#
# Why not just `source ~/.bashrc`: Raspberry Pi OS's ~/.bashrc starts with
#     case $- in *i*) ;; *) return;; esac
# so in a script started from a desktop icon (a NON-interactive shell) it
# returns on that line and the exports further down never run. The launcher
# and every main.py it starts then have no FATIGUE_API_BASE_URL /
# FATIGUE_API_TOKEN and talk to http://localhost:8000/api - enrollments never
# reach the portal. A terminal is interactive, so `python main.py --enroll`
# from a terminal works with the same ~/.bashrc.
#
# Settings are read from, in order:
#   1. /etc/fatigue-detection.env   (the same file the systemd unit uses)
#   2. ~/.config/fatigue-detection.env
#   3. the `export FATIGUE_...=` lines of ~/.bashrc (read directly, so the
#      interactive guard does not matter)
# Each file is plain KEY=value lines - no quotes, no spaces, no "export" (the
# same file is read by systemd EnvironmentFile= and by this shell), e.g.
#   FATIGUE_API_BASE_URL=http://192.168.1.20/api
#   FATIGUE_API_TOKEN=...
#   FATIGUE_DEVICE_ID=pi-01
#   FATIGUE_STREAM_TOKEN=...   (optional: fixed debug-stream URL token, 8-64
#                               of A-Z a-z 0-9 _ - ; else random per launcher run)
#
# Desktop entry (~/Desktop/fatigue-launcher.desktop):
#   [Desktop Entry]
#   Type=Application
#   Name=Fatigue Launcher
#   Exec=/bin/bash /home/pi/fatigue-detection/deploy/start_launcher.sh
#   Terminal=false

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$PROJECT/logs/start_launcher.log"
mkdir -p "$PROJECT/logs"

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

PY="$PROJECT/venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

{
    echo "===== $(date '+%F %T') start_launcher.sh"
    echo "settings from: ${SOURCE:-NOTHING FOUND}"
    echo "FATIGUE_API_BASE_URL=${FATIGUE_API_BASE_URL:-(NOT SET)}"
    echo "FATIGUE_API_TOKEN=$([ -n "$FATIGUE_API_TOKEN" ] && echo '(set)' || echo '(NOT SET)')"
    echo "FATIGUE_DEVICE_ID=${FATIGUE_DEVICE_ID:-(NOT SET)}"
    echo "python: $PY"
} >> "$LOG"

cd "$PROJECT" || exit 1
exec "$PY" launcher.py "$@" >> "$LOG" 2>&1
