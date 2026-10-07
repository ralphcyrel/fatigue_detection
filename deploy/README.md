# Deploying the touchscreen launcher on the Pi

`launcher.py` gives the unit a touch-only menu (Enroll / Pre-drive /
Monitoring / Follow ignition / Quit) and shells out to `main.py`. This
folder holds the systemd unit that starts it on boot.

## Prerequisites

- Project installed at `/home/pi/fatigue-detection` with the venv at
  `venv/` (per the main README). Edit the paths in the unit file otherwise.
- `python3-tk` (present on the Pi OS desktop image; otherwise
  `sudo apt install python3-tk`). The venv was created with
  `--system-site-packages`, so it sees the apt package.
- **Desktop autologin** enabled: `sudo raspi-config` → *System Options* →
  *Boot / Auto Login* → *Desktop Autologin*. The launcher needs a display,
  so the unit waits for X to come up rather than starting a session itself.
- Backend settings in `/etc/fatigue-detection.env` (readable by `pi`):

  ```
  FATIGUE_API_BASE_URL=http://<server>/api
  FATIGUE_API_TOKEN=<bearer token>
  FATIGUE_DEVICE_ID=pi-01
  # optional - a fixed debug-stream URL token (else a random one per launcher run)
  # FATIGUE_STREAM_TOKEN=pick-8-to-64-chars
  ```

  `sudo chmod 640 /etc/fatigue-detection.env && sudo chown root:pi /etc/fatigue-detection.env`

## Install / enable

```bash
cd /home/pi/fatigue-detection
sudo cp deploy/fatigue-launcher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fatigue-launcher
```

Reboot to confirm it comes up on its own. Logs:

```bash
journalctl -u fatigue-launcher -f       # launcher stdout + systemd events
tail -f logs/launcher.log               # launcher's own log
tail -f logs/session.log                # stdout/stderr of each main.py session
tail -f logs/fatigue.log                # main.py's normal log
```

## Disable while developing

```bash
sudo systemctl disable --now fatigue-launcher   # stop now AND don't start on boot
sudo systemctl stop fatigue-launcher            # stop until next boot only
sudo systemctl start fatigue-launcher           # start it again by hand
sudo systemctl enable --now fatigue-launcher    # back to plug-and-go
```

With the service stopped, run `python main.py ...` from a terminal as usual.
Starting the launcher while the service is running would fight over the
camera - `systemctl status fatigue-launcher` tells you which state it's in.

## Try it without installing the service

```bash
source venv/bin/activate
python launcher.py               # fullscreen on the touchscreen
python launcher.py --windowed    # 800x480 window (Escape leaves fullscreen)
python launcher.py --screen-size 800x480   # force the size if detection picks
                                           # the wrong one (e.g. under VNC)
```

The launcher logs the size it laid out for and where it came from, e.g.
`Display 800x480 (monitor HDMI-1), scale 1.50, button font 29 px`.

## Starting from a desktop icon

Use `deploy/start_launcher.sh` as the icon's `Exec=` (the script's header
has a sample `.desktop` file). Do **not** `source ~/.bashrc` in a launcher
script: Raspberry Pi OS's `~/.bashrc` returns immediately in a
non-interactive shell, so the `FATIGUE_*` exports never run and every
enrollment goes to `http://localhost:8000/api`. The script reads
`/etc/fatigue-detection.env` (same file as the service), then
`~/.config/fatigue-detection.env`, then the `export FATIGUE_...` lines of
`~/.bashrc`, and logs what it found to `logs/start_launcher.log`.

The launcher's header turns red ("NO API SETTINGS") when it was started
without them. Each session's first lines in `logs/session.log` show the
exact command, interpreter and API settings the child received.

## Notes

- While a session runs the launcher collapses to a strip along the bottom
  of the screen with a **STOP** button; `main.py`'s fullscreen data screen
  sits above it. STOP sends SIGINT, the same as Ctrl-C, so `main.py` cleans
  up normally.
- The in-vehicle display never shows the camera image during pre-drive or
  monitoring (group decision, 2026-10-05). For camera aim / landmark
  problems: `main.py --show-video` (local window), or the **Video stream**
  button, which starts sessions with `--debug-stream`: an MJPEG view at
  `http://<pi-ip>:8080/<token>/` for a laptop or phone on the same network.
  Every other path is a 404. It is LAN-only, never recorded; the full URL
  is shown on the menu, the confirm screen and the strip, and logged.
- Enrollment shows no camera image either: a face-box outline with text
  guidance (step 1 positions the driver before the face capture), progress
  bars and the closed-eye countdown. Fine positioning is done on the stream.
- If the strip ends up hidden behind the session window on a Wayland
  desktop, switch the session to X11 (`raspi-config` → *Advanced Options* →
  *Wayland* → *X11*); dock/topmost hints are honoured reliably there.
