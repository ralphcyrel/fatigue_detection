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
python launcher.py --windowed    # 480x320 window (Escape leaves fullscreen)
```

## Notes

- While a session runs the launcher collapses to a strip along the bottom
  of the screen with a **STOP** button; the cv2 preview appears above it.
  STOP sends SIGINT, the same as Ctrl-C, so `main.py` cleans up normally.
- If the strip ends up hidden behind the preview on a Wayland desktop,
  switch the session to X11 (`raspi-config` → *Advanced Options* →
  *Wayland* → *X11*); dock/topmost hints are honoured reliably there.
- The preview is drawn at the camera frame size. On a 480×320 panel a
  640×480 frame is clipped, which is fine for a demo but worth knowing.
