#!/bin/bash
# Install the touchscreen launcher's XDG autostart entry for the CURRENT user:
# copies deploy/fatigue-launcher.desktop into ~/.config/autostart/ (with this
# checkout's path filled in) and makes deploy/launcher_autostart.sh
# executable. Run it as the desktop user, not with sudo:
#
#     bash deploy/install_autostart.sh
#
# It only reads the system: no boot configuration, no autologin, no systemd
# unit is changed. It prints how to verify the install and how to undo it.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
WRAPPER="$SCRIPT_DIR/launcher_autostart.sh"
TEMPLATE="$SCRIPT_DIR/fatigue-launcher.desktop"
DEST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"
DEST="$DEST_DIR/fatigue-launcher.desktop"

if [ "$(id -u)" -eq 0 ]; then
    echo "ERROR: run this as the desktop user (no sudo) - as root it would install into" >&2
    echo "       /root/.config/autostart, which the desktop session never reads." >&2
    exit 1
fi
# The path goes inside a double-quoted Exec= argument (and through sed).
case "$PROJECT" in
    *'"'* | *'`'* | *'$'* | *'\'* | *'|'* | *'%'*)
        echo "ERROR: $PROJECT contains a character a .desktop Exec= line cannot carry" >&2
        echo "       (one of \" \` \$ \\ | %) - move the checkout and run this again." >&2
        exit 1 ;;
esac

chmod +x "$WRAPPER" "$SCRIPT_DIR/install_autostart.sh"
mkdir -p "$DEST_DIR"
sed "s|@PROJECT_DIR@|$PROJECT|g" "$TEMPLATE" > "$DEST"
chmod 644 "$DEST"
echo "Installed $DEST"
echo "  Exec: $(grep '^Exec=' "$DEST" | cut -d= -f2-)"
if command -v desktop-file-validate >/dev/null 2>&1; then
    desktop-file-validate "$DEST" && echo "  desktop-file-validate: OK"
fi

# ---- checks (read-only) -------------------------------------------------------
echo
[ -f "$PROJECT/venv/bin/activate" ] \
    || echo "WARNING: no venv at $PROJECT/venv - the wrapper will refuse to start (see README.md)."
if command -v systemctl >/dev/null 2>&1 \
        && systemctl is-enabled fatigue-launcher.service >/dev/null 2>&1; then
    echo "WARNING: the old systemd unit fatigue-launcher.service is ENABLED as well. Two"
    echo "         launchers would fight over the camera. Disable it (not done for you):"
    echo "             sudo systemctl disable --now fatigue-launcher"
fi
BOOT_CFG=/boot/firmware/config.txt
if [ -r "$BOOT_CFG" ] && ! grep -qE '^[[:space:]]*gpio=17=op,dh' "$BOOT_CFG"; then
    echo "REMINDER: $BOOT_CFG has no 'gpio=17=op,dh' line, so the starter relay pin is not"
    echo "          held at INHIBIT while the Pi boots (see deploy/README.md; not changed here)."
fi
echo "Note: the launcher starts when the desktop session starts, i.e. after login. For it"
echo "      to come up with no keyboard the Pi must boot to the desktop with autologin"
echo "      (raspi-config -> System Options -> Boot / Auto Login); this script does not"
echo "      change that setting."

cat <<EOF

Verify:
    sudo reboot                                  # or log out and back in
    pgrep -af launcher.py                        # the launcher is running
    tail -n 50 $PROJECT/logs/launcher_autostart.log
    ls -l $DEST

Run it by hand once (same environment, same log):
    $WRAPPER &

Undo (stop starting it at login):
    rm $DEST
    pkill -f "python launcher.py"     # stop the running one (SIGTERM: it stops its
                                      # session and inhibits the starter first)
EOF
