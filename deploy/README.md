# Deploying the touchscreen launcher on the Pi

`launcher.py` gives the unit a touch-only menu (Enroll / Pre-drive →
monitoring / Monitoring only / Video stream / Quit) and shells out to
`main.py`. This folder holds what starts it automatically when the Pi powers
on (an XDG autostart entry), a desktop-icon script, and the older systemd
unit (superseded - see below).

## Prerequisites

- Project checked out at `~/fatigue-detection` (any path works - nothing
  below hard-codes a user or a directory) with the venv at `venv/` (per the
  main README).
- `python3-tk` (present on the Pi OS desktop image; otherwise
  `sudo apt install python3-tk`). The venv was created with
  `--system-site-packages`, so it sees the apt package.
- **Boot to the desktop with autologin**: `sudo raspi-config` → *System
  Options* → *Boot / Auto Login* → *Desktop Autologin* (the default on the
  Pi OS desktop image). XDG autostart entries run when the desktop session
  starts, so without autologin the launcher only appears after someone logs
  in. The scripts here never change this setting.
- Backend settings in `/etc/fatigue-detection.env` (readable by the desktop
  user), or `~/.config/fatigue-detection.env`:

  ```
  FATIGUE_API_BASE_URL=http://<server>/api
  FATIGUE_API_TOKEN=<bearer token>
  FATIGUE_DEVICE_ID=pi-01
  # optional - a fixed debug-stream URL token (else a random one per launcher run)
  # FATIGUE_STREAM_TOKEN=pick-8-to-64-chars
  ```

  `sudo chmod 640 /etc/fatigue-detection.env && sudo chown root:$USER /etc/fatigue-detection.env`

- **Starter relay held off during boot**: add `"gpio=17=op,dh"` to
  `/boot/firmware/config.txt` so GPIO 17 is held HIGH (relay off) during
  boot. The relay module is low-level trigger, so a LOW pin would allow a
  start before the launcher runs. (`install_autostart.sh` reminds you if the
  line is missing; it does not add it.)

## Start on power-on (XDG autostart)

Three files:

| File | What it is |
|---|---|
| `deploy/launcher_autostart.sh` | Wrapper: derives every path from its own location, logs to `logs/launcher_autostart.log`, refuses a second copy, activates `./venv`, loads the `FATIGUE_*` settings (as `start_launcher.sh` does), **waits for the display** (Tk can open it and `xrandr` lists a monitor; 30 tries, 2 s apart) and for the camera (`rpicam-hello --list-cameras`; 10 tries, only a warning if it never appears), then `exec`s `launcher.py`. |
| `deploy/fatigue-launcher.desktop` | The XDG autostart entry. A template: its `Exec=` has `@PROJECT_DIR@`, filled in by the installer. |
| `deploy/install_autostart.sh` | Copies the entry into `~/.config/autostart/` with this checkout's path, makes the wrapper executable, and prints how to verify and undo. Read-only checks: warns if the old systemd unit is enabled too, reminds about `gpio=17=op,dh`. |

Works on Raspberry Pi OS Bookworm with any of its desktops (labwc, Wayfire
or X11): all of them run `~/.config/autostart/*.desktop` at login. No
systemd service, no autologin or boot change.

### Install

Run as the desktop user (not with `sudo`):

```bash
cd ~/fatigue-detection
bash deploy/install_autostart.sh
sudo reboot
```

### Verify

```bash
pgrep -af launcher.py                          # it is running
tail -n 50 ~/fatigue-detection/logs/launcher_autostart.log
ls -l ~/.config/autostart/fatigue-launcher.desktop
```

A cold boot logs something like:

```
launcher_autostart: ===== starting: project /home/<user>/fatigue-detection, ...
launcher_autostart: display :0 not ready (attempt 1/30) - retrying in 2s
launcher_autostart: display :0 ready (attempt 2/30)
launcher_autostart: camera listed by rpicam-hello (attempt 1/10)
launcher_autostart: exec launcher.py
... launcher's own log lines (also in logs/launcher.log) ...
```

To try the wrapper without rebooting: `~/fatigue-detection/deploy/launcher_autostart.sh &`
(it exits at once, logging why, if a launcher it started is already running).

### Disable / undo

```bash
rm ~/.config/autostart/fatigue-launcher.desktop   # no longer starts at login
pkill -f "python launcher.py"                     # stop the running one now
```

`pkill` sends SIGTERM; the launcher stops its `main.py` session (which
inhibits the starter on the way out) and drives the relay to INHIBIT before
it exits. To re-enable, run `bash deploy/install_autostart.sh` again.

### Logs

```bash
tail -f logs/launcher_autostart.log     # wrapper + launcher stdout/stderr (rotated at 5 MB)
tail -f logs/launcher.log               # launcher's own log
tail -f logs/session.log                # stdout/stderr of each main.py session
tail -f logs/fatigue.log                # main.py's normal log
```

Retry counts can be changed with `FATIGUE_AUTOSTART_DISPLAY_TRIES`,
`FATIGUE_AUTOSTART_CAMERA_TRIES` and `FATIGUE_AUTOSTART_DELAY_S` (e.g. in
the env file).

## The old systemd unit (superseded)

`deploy/fatigue-launcher.service` is a *system* service that ran the GUI
before there was a desktop session to run it in, with `/home/pi` paths
hard-coded. It is kept only for reference; use the autostart entry above.
**Never enable both** - two launchers fight over the camera. If it was
installed earlier:

```bash
sudo systemctl disable --now fatigue-launcher
```

## Session controls on the strip

While a pre-drive / monitoring session runs, the strip along the bottom of
the screen shows, besides STOP:

- **BEGIN MONITORING** - only while `main.py` is in READY_TO_START (the
  assessment passes and the starter is released). It is the start
  confirmation: there is no ignition sense, so monitoring never starts on
  its own.
- **START NEXT DRIVER** - only in IDLE, after END SESSION.
- **END SESSION** - asks again: CANCEL takes its place, YES takes STOP's
  place, and the question cancels itself after 6 s. On YES `main.py`
  inhibits the starter, stops any alert, closes the session and waits in
  IDLE for the next driver; it keeps running. STOP still ends `main.py`
  altogether (back to the menu).

The buttons follow `logs/session_state.json`, which `main.py` rewrites on
every state change; the taps reach it as SIGUSR1 (begin) and SIGUSR2 (end
session), and `main.py` ignores a tap made in any other state.

Each `main.py` runs in its own process group with one helper process, the
face-recognition worker (`modules/recognition_worker.py`). `main.py` stops
it on END SESSION (a new one starts with START NEXT DRIVER) and on every
exit; if `main.py` is killed it dies with it, and the launcher kills the
group after `main.py` exits. To check for strays:
`pgrep -af "multiprocessing.spawn"` should list nothing while no session
runs. Because of the separate group, Ctrl-C in the terminal that started the
launcher reaches the launcher only, which stops the session itself.

## Try it without autostart

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
has a sample `.desktop` file). Stop the autostarted launcher first
(`pkill -f "python launcher.py"`) - only one can hold the camera. Do **not** `source ~/.bashrc` in a launcher
script: Raspberry Pi OS's `~/.bashrc` returns immediately in a
non-interactive shell, so the `FATIGUE_*` exports never run and every
enrollment goes to `http://localhost:8000/api`. The script reads
`/etc/fatigue-detection.env` (same file as the service), then
`~/.config/fatigue-detection.env`, then the `export FATIGUE_...` lines of
`~/.bashrc`, and logs what it found to `logs/start_launcher.log`.

The launcher's header turns red ("NO API SETTINGS") when it was started
without them. Each session's first lines in `logs/session.log` show the
exact command, interpreter and API settings the child received.

## Portal live view (privacy)

When the **Video stream** button is on, each pre-drive / monitoring session
lists its stream on the operator portal's device page (*Devices → pi-01 →
Live view*), so operators can open it without typing the URL.

- **It shows the driver's face** to any authenticated operator who opens it.
  The group has not yet confirmed this use; it can be switched off on the
  portal server with `PORTAL_LIVE_VIEW=false` in its `.env` (no code change;
  run `./docker.sh cache-clear` there if the config is cached).
- **Frames never leave the Pi.** Only the stream's address (LAN IP, port,
  URL token) is sent to the portal. Browsers fetch the video from the Pi
  itself, and nothing is recorded anywhere.
- **It exists only during an active session** with the stream on. The portal
  drops the entry when the session ends, or 90 s after the last heartbeat if
  the unit stops without saying so.
- **It is reachable only from the local network.** The Pi refuses
  non-private addresses, so a portal user elsewhere sees the link, but it
  won't open for them.

## Notes

- While a session runs, a separate strip window along the bottom of the
  screen carries a **STOP** button; `main.py`'s fullscreen screen sits above
  the launcher's own (unchanged, still fullscreen) window. STOP sends
  SIGINT, the same as Ctrl-C, so `main.py` cleans up normally. When the
  session ends the launcher re-asserts its startup geometry, raises itself
  and checks again 150 / 600 / 1500 ms later. Each step is logged to
  `logs/start_launcher.log` (and `logs/launcher.log`) as
  `Window [<stage>]: requested WxH+X+Y, actual WxH+X+Y (...), layout WxH`;
  `NOT the startup geometry - re-asserting` means the window manager had
  moved or resized it.
- The in-vehicle display never shows the camera image during pre-drive or
  monitoring (group decision, 2026-10-05). For camera aim / landmark
  problems: `main.py --show-video` (local window), or the **Video stream**
  button, which starts sessions with `--debug-stream`: an MJPEG view at
  `http://<pi-ip>:8080/<token>/` for a laptop or phone on the same network.
  Every other path is a 404. It is LAN-only, never recorded; the full URL
  is shown on the menu, the confirm screen and the strip, and logged.
- Enrollment is the one exception (2026-10-07): the enrollment screen shows
  the live camera as a mirror view, with the face box and the dashed target
  zone on it, next to the text guidance (step 1 positions the driver before
  the face capture), progress bars and the closed-eye countdown. It is
  redrawn at the data screen's 10 Hz, so it costs the capture loop nothing
  measurable.
- The strip is an unmanaged (override-redirect) window, raised whenever it
  reports being covered and every 500 ms: as a managed dock / topmost
  window it ended up under `main.py`'s fullscreen window on the Pi
  (2026-10-19), so BEGIN MONITORING and END SESSION could not be touched.
  `Strip covered (...) - raised above main.py's window` in
  `logs/start_launcher.log` means that happened and was undone; the count
  is logged when the session ends. If the strip still ends up hidden on a
  Wayland desktop, switch the session to X11 (`raspi-config` → *Advanced
  Options* → *Wayland* → *X11*).
