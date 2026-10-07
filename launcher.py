"""
Touchscreen launcher for the Driver Fatigue Detection unit.

Lets the Pi be demonstrated with nothing but a touchscreen: plug in and a
menu appears. Every action shells out to ``main.py`` with the matching CLI
flags - nothing from ``main`` is imported, so the CLI stays the single
source of truth for how a session runs::

    Enroll driver             -> main.py --enroll --driver-id N
    Pre-drive -> monitoring   -> main.py --sequence   (pass / approved override
                                 continues into monitoring, no ignition input)
    Monitoring only (test)    -> main.py --force-phase monitoring
    Follow ignition           -> main.py            (real ignition input)

While a session runs the launcher collapses to a slim always-on-top strip
along the bottom of the screen with a STOP button (sends SIGINT, which
``main.py`` already handles as Ctrl-C and cleans up after). Above it,
``main.py`` shows its fullscreen data screen - no camera image (group
decision, 2026-10-05). When the child exits the menu comes back with the
result; an enrollment gets a full result screen (saved / refused and why /
backend dropped the closed-eye baseline / backend rejected).

"Video stream" toggles ``main.py --debug-stream`` for the next sessions: a
LAN-only MJPEG view of the camera for a laptop or phone (enrollment
positioning, the defense projector). The strip shows the URL to open.

Layout is proportional (grid weights + pixel fonts scaled from the size of
the window), sized for the 800x480 HDMI touchscreen; touch targets stay
well above the ~10 px error of a resistive panel. The size is that of the
monitor the window is on (xrandr), not the whole X screen - which spans
every monitor and, over VNC, can be far larger than the panel - and it is
re-taken every time the menu comes back from a session; after that the
layout follows whatever size the window manager actually gives the window.
Button text is fitted so the longest menu label stays whole.

    python launcher.py                        # fullscreen (kiosk)
    python launcher.py --windowed             # 800x480 window, for development
    python launcher.py --screen-size 800x480  # exactly 800x480, borderless at
                                              # 0,0 - verify the panel layout
                                              # on a desktop

The child runs with this launcher's environment (FATIGUE_API_BASE_URL,
FATIGUE_API_TOKEN, FATIGUE_DEVICE_ID) and with the project venv's Python
when the launcher itself was started outside it; the header turns red when
the API settings are missing, which is what a desktop icon that sources
``~/.bashrc`` non-interactively produces (see deploy/start_launcher.sh).

Tkinter only - ships with Python (``python3-tk`` on Raspberry Pi OS).
See ``deploy/fatigue-launcher.service`` to start it on boot.
"""

import argparse
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple
from urllib.parse import urlparse

from config import config
from modules.api import APIClient
from modules.debug_stream import lan_address, stream_token

logger = logging.getLogger("launcher")

# Reference size the sizes below are designed for: the 800x480 panel's
# shape at a height of 320. The scale is set by whichever of width and
# height runs out first, so no window shape pushes a row or a column off an
# edge, and capped so a 1080p panel doesn't end up with comically big text.
# On the 800x480 panel the scale is 1.5.
REFERENCE_HEIGHT = 320
REFERENCE_WIDTH = 800 * REFERENCE_HEIGHT // 480      # 533
MIN_SCALE = 0.75
MAX_SCALE = 2.5

# Sizes at the reference size, in px. Buttons are finger-sized: on the
# 800x480 panel a menu row is ~95 px and the strip 66 px - far above the
# ~10 px error of the resistive touchscreen.
ROW_MIN_PX = 64
HEADER_PX = 40
# Kept slimmer than a menu row: main.py's data screen reserves exactly this
# much (``--reserve-bottom``) so the strip never covers a reading.
STRIP_PX = 44
BUTTON_PADX_PX = 6       # inside a button, around its text
BUTTON_PADY_PX = 2
# Fonts are in pixels, not points, so the layout does not depend on the DPI
# the X server reports (a VNC session reports anything). These equal the
# old 12 / 15 / 9 pt at the Pi's 96 dpi. Text that must stay whole (menu
# labels, the stream URL, the strip) is fitted below these sizes when the
# space is short; MIN_FONT_PX is as small as fitting goes.
FONT_TITLE_PX = 16
FONT_BUTTON_PX = 20
FONT_SMALL_PX = 12
MIN_FONT_PX = 8

# Menu button labels. The button font is fitted so the longest stays whole
# in a menu cell at any screen size (tests/test_launcher.py checks 800x480
# and 1920x1080).
LABEL_ENROLL = "Enroll driver"
LABEL_PREDRIVE = "Pre-drive assessment\n→ monitoring"
LABEL_MONITORING = "Monitoring only\n(test)"
LABEL_IGNITION = "Follow ignition\n(real input)"
LABEL_STREAM = "Video stream: {}\n(laptop / phone view)"
LABEL_QUIT = "Quit"
MENU_LABELS = (LABEL_ENROLL, LABEL_PREDRIVE, LABEL_MONITORING, LABEL_IGNITION,
               LABEL_STREAM.format("OFF"), LABEL_STREAM.format("ON"), LABEL_QUIT)

# Backend pill text by reachability (None = not known yet).
PILL_TEXT: Dict[Optional[bool], str] = {
    None: "Backend: checking…", True: "Backend: ONLINE", False: "Backend: OFFLINE",
}

REFIT_DELAY_MS = 100         # let a window-manager resize settle before re-laying out

PING_INTERVAL_S = 10.0       # backend reachability refresh
RESULT_POLL_MS = 100         # how often the Tk thread drains worker results
CHILD_POLL_MS = 500          # how often the running screen checks the child
STOP_GRACE_S = 10.0          # SIGINT -> terminate() escalation
SESSION_LOG = config.LOGS_DIR / "session.log"   # child stdout/stderr
ENROLL_RESULT = config.LOGS_DIR / "enroll_result.json"   # main.py --result-file

# Colours (plain tk, works without ttk themes on the Pi).
BG = "#1e1e1e"
FG = "#f0f0f0"
BTN = "#2f3b4a"
BTN_ACTIVE = "#3f4f63"
BTN_PRIMARY = "#2b6cb0"
BTN_STOP = "#b02b2b"
BTN_QUIT = "#4a4a4a"
BTN_ASSIGNED = "#2b7a4b"     # driver assigned to this unit
ONLINE = "#2e9e5b"
OFFLINE = "#c0392b"
UNKNOWN = "#7f8c8d"
WARN_BG = "#8a1f1f"          # header when the API settings are missing
RESULT_OK = "#2e9e5b"
RESULT_WARN = "#b7791f"
RESULT_BAD = "#c0392b"

# Menu actions: label -> main.py flags. Enroll is handled separately (needs
# a driver id from the picker).
#
# "Pre-drive" is the whole session: a pass (or an approved override) releases
# the starter and continues into monitoring by itself. "Monitoring only" is a
# test entry: it skips the 30 s assessment, and it starts with the starter
# INHIBITED (nothing in monitoring can release it) - use it to exercise
# monitoring alone, e.g. with --profile-loop, or for a driver who cannot pass.
#
# Every session keeps main.py's window on purpose: the data screen's level,
# lock reason and assessment result are how an observer sees the system's
# reasoning. It shows no camera image; the camera view, when needed, is the
# LAN video stream. main.py --no-preview is for unattended data collection
# from the command line, not a launcher default.
SESSIONS: Dict[str, List[str]] = {
    "Pre-drive assessment": ["--sequence"],
    "Monitoring only (test)": ["--force-phase", "monitoring"],
    "Follow ignition": [],
}


# What main.py's exit codes mean, for sessions without a result file (and as
# the fallback when an enrollment did not write one).
EXIT_TEXT: Dict[int, str] = {
    0: "finished OK",
    1: "crashed - see logs/session.log",
    2: "bad driver id - nothing saved",
    3: "too few face frames - nothing saved",
    5: "backend rejected or unreachable - nothing saved",
    6: "REFUSED: no eyelid contrast - nothing saved",
    7: "saved, but the backend DROPPED the closed-eye baseline",
}


def child_python() -> str:
    """
    Interpreter for main.py: this one, unless the launcher was started
    outside the project venv while one exists. A desktop icon running
    ``python3 launcher.py`` would otherwise start main.py on the system
    Python, without face_recognition / dlib, and it would die on import.
    """
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    venv = config.BASE_DIR / "venv" / ("Scripts/python.exe" if sys.platform == "win32"
                                       else "bin/python")
    if not in_venv and venv.exists():
        logger.warning("Launcher runs on %s (not the project venv) - starting main.py with %s",
                       sys.executable, venv)
        return str(venv)
    return sys.executable


def api_settings_problem() -> Optional[str]:
    """Why the backend settings look missing (None when they look set)."""
    missing = [name for name in ("FATIGUE_API_BASE_URL", "FATIGUE_API_TOKEN")
               if not os.environ.get(name)]
    if not missing:
        return None
    return (f"{' and '.join(missing)} not set - enrollment cannot reach the backend. "
            "Start the launcher with deploy/start_launcher.sh.")


def setup_logging() -> None:
    """Launcher log to the console and ``logs/launcher.log``."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)
    file_handler = RotatingFileHandler(
        config.LOGS_DIR / "launcher.log", maxBytes=1024 * 1024, backupCount=2
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Screen size and text fitting
# ---------------------------------------------------------------------------

def layout_scale(width: int, height: int) -> float:
    """UI scale for a ``width`` x ``height`` px window."""
    return max(MIN_SCALE, min(width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT, MAX_SCALE))


def parse_screen_size(text: str) -> Tuple[int, int]:
    """``--screen-size`` value ``WxH`` -> (W, H)."""
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", text)
    if not match:
        raise argparse.ArgumentTypeError(f"expected WxH, e.g. 800x480 (got {text!r})")
    w, h = int(match.group(1)), int(match.group(2))
    # Below this the scale would drop under MIN_SCALE and rows overflow.
    if w < 400 or h < 240:
        raise argparse.ArgumentTypeError(f"{w}x{h} is too small (minimum 400x240)")
    return w, h


class Monitor(NamedTuple):
    name: str
    x: int
    y: int
    width: int
    height: int
    primary: bool


# " 0: +*HDMI-1 800/154x480/86+0+0  HDMI-1" (size px/mm, then the offset)
_XRANDR_MONITOR = re.compile(
    r"^\s*\d+:\s+\+?(\*?)(\S+)\s+(\d+)/\d+x(\d+)/\d+([+-]\d+)([+-]\d+)")


def parse_xrandr_monitors(text: str) -> List[Monitor]:
    """Monitors listed by ``xrandr --listactivemonitors``."""
    monitors = []
    for line in text.splitlines():
        match = _XRANDR_MONITOR.match(line)
        if match:
            primary, name, w, h, x, y = match.groups()
            monitors.append(Monitor(name, int(x), int(y), int(w), int(h), bool(primary)))
    return monitors


def detect_monitors() -> List[Monitor]:
    """The X server's monitors; empty off Linux or when xrandr is unavailable."""
    if not sys.platform.startswith("linux"):
        return []
    try:
        out = subprocess.run(["xrandr", "--listactivemonitors"], capture_output=True,
                             text=True, timeout=2, check=False).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("xrandr failed (%s) - sizing to the whole X screen", exc)
        return []
    return parse_xrandr_monitors(out)


def monitor_at(monitors: Sequence[Monitor], x: int, y: int) -> Optional[Monitor]:
    """The monitor containing (x, y); else the primary one; else the first."""
    for mon in monitors:
        if mon.x <= x < mon.x + mon.width and mon.y <= y < mon.y + mon.height:
            return mon
    return next((mon for mon in monitors if mon.primary), monitors[0] if monitors else None)


def text_size(font: tkfont.Font, text: str) -> Tuple[int, int]:
    """(width, height) in px of ``text`` (explicit line breaks only) in ``font``."""
    lines = text.split("\n")
    return max(font.measure(line) for line in lines), font.metrics("linespace") * len(lines)


def fit_font(font: tkfont.Font, texts: Sequence[str], max_w: int, max_h: int,
             start_px: int) -> int:
    """
    Set ``font`` to the largest pixel size <= ``start_px`` at which every
    text fits ``max_w`` x ``max_h`` (never below MIN_FONT_PX); returns it.
    """
    size = max(start_px, MIN_FONT_PX)
    while True:
        font.configure(size=-size)
        if size <= MIN_FONT_PX or all(
                w <= max_w and h <= max_h for w, h in (text_size(font, t) for t in texts)):
            return size
        size -= 1


def elide(font: tkfont.Font, text: str, max_w: int) -> str:
    """``text`` shortened with "…" to fit ``max_w`` px ("" when nothing fits)."""
    if font.measure(text) <= max_w:
        return text
    while text and font.measure(text + "…") > max_w:
        text = text[:-1]
    return text.rstrip() + "…" if text else ""


# ---------------------------------------------------------------------------
# Backend access (off the Tk thread)
# ---------------------------------------------------------------------------

class Backend:
    """
    Thin wrapper around ``APIClient`` for the launcher.

    All calls run on worker threads so a slow or absent backend never
    freezes the UI. Tk is not thread-safe, so results go through a queue
    that the Tk thread drains from an ``after`` timer (``pump``). The lock
    serialises calls because ``requests.Session`` is not thread-safe either.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.client = APIClient()
        self._lock = threading.Lock()
        self._results: "queue.Queue[tuple]" = queue.Queue()
        self.root.after(RESULT_POLL_MS, self.pump)

    def _run(self, fn: Callable[[], Any], done: Callable[[Any], None]) -> None:
        def worker() -> None:
            with self._lock:
                try:
                    result = fn()
                except Exception:      # never let a worker kill the UI
                    logger.exception("Backend call failed")
                    result = None
            self._results.put((done, result))
        threading.Thread(target=worker, daemon=True).start()

    def pump(self) -> None:
        """Deliver finished results to their callbacks on the Tk thread."""
        while True:
            try:
                done, result = self._results.get_nowait()
            except queue.Empty:
                break
            done(result)
        self.root.after(RESULT_POLL_MS, self.pump)

    def ping(self, done: Callable[[bool], None]) -> None:
        self._run(self.client.ping, done)

    def drivers(self, done: Callable[[Optional[List[Dict[str, Any]]]], None]) -> None:
        self._run(self.client.get_drivers, done)


# ---------------------------------------------------------------------------
# Child session
# ---------------------------------------------------------------------------

class Session:
    """One ``main.py`` subprocess and the means to stop it."""

    def __init__(self, label: str, flags: List[str], enroll: bool = False,
                 extra_env: Optional[Dict[str, str]] = None) -> None:
        self.label = label
        self.flags = flags
        self.enroll = enroll
        self.extra_env = extra_env or {}
        self.proc: Optional[subprocess.Popen] = None
        self._log_file = None
        self._log_offset = 0
        self._stop_requested_at: Optional[float] = None

    def start(self) -> None:
        cmd = [child_python(), str(config.BASE_DIR / "main.py"), *self.flags]
        if self.enroll:
            cmd += ["--result-file", str(ENROLL_RESULT)]
            try:
                ENROLL_RESULT.unlink()      # never show a previous run's result
            except FileNotFoundError:
                pass
        # The child gets this process's environment, explicitly: the API
        # settings must reach main.py exactly as the launcher sees them.
        env = dict(os.environ, PYTHONUNBUFFERED="1", **self.extra_env)
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._log_file = open(SESSION_LOG, "a", encoding="utf-8")
        self._log_offset = self._log_file.tell()
        self._log_file.write(
            f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} launcher: {self.label} "
            f"-> {' '.join(cmd)}\n"
            f"      env: FATIGUE_API_BASE_URL={env.get('FATIGUE_API_BASE_URL', '(NOT SET)')} "
            f"FATIGUE_API_TOKEN={'(set)' if env.get('FATIGUE_API_TOKEN') else '(NOT SET)'} "
            f"FATIGUE_DEVICE_ID={env.get('FATIGUE_DEVICE_ID', '(NOT SET)')}\n"
        )
        self._log_file.flush()
        logger.info("Starting %s: %s", self.label, " ".join(cmd))
        # stdin is /dev/null so a stray input() fails fast instead of hanging
        # a keyboard-less unit forever.
        self.proc = subprocess.Popen(
            cmd, cwd=str(config.BASE_DIR), env=env,
            stdin=subprocess.DEVNULL, stdout=self._log_file, stderr=subprocess.STDOUT,
        )

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        """
        Ask the child to stop like Ctrl-C would; escalate if it ignores us.

        First call sends SIGINT (``main.py`` catches KeyboardInterrupt and
        runs ``cleanup()``, releasing the camera and resetting GPIO).
        Called again after ``STOP_GRACE_S`` it terminates, then kills.
        """
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        now = time.time()
        if self._stop_requested_at is None:
            self._stop_requested_at = now
            logger.info("Stop requested for %s (pid %s)", self.label, proc.pid)
            if sys.platform == "win32":
                proc.terminate()          # no SIGINT on Windows (dev only)
            else:
                proc.send_signal(signal.SIGINT)
        elif now - self._stop_requested_at > STOP_GRACE_S:
            logger.warning("%s did not exit after SIGINT - terminating", self.label)
            proc.terminate()
            self._stop_requested_at = now + STOP_GRACE_S   # next escalation = kill
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def close(self) -> Optional[int]:
        """Reap the process (if exited) and close the log; returns the exit code."""
        code = None if self.proc is None else self.proc.poll()
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        return code

    @property
    def stopped_by_user(self) -> bool:
        code = None if self.proc is None else self.proc.poll()
        return self._stop_requested_at is not None or (code is not None and code < 0)

    def result_text(self) -> str:
        code = self.close()
        if self.stopped_by_user:
            return f"{self.label}: stopped"
        text = EXIT_TEXT.get(code, f"exited with code {code} - see logs/session.log")
        return f"{self.label}: {text}"

    def enroll_result(self) -> Dict[str, Any]:
        """
        The enrollment's outcome: main.py's result file, or - if it wrote
        none (killed, crashed on import) - one built from the exit code and
        the last lines of this session's log.
        """
        code = self.close()
        try:
            with open(ENROLL_RESULT, encoding="utf-8") as fh:
                result = json.load(fh)
            if isinstance(result, dict) and "outcome" in result:
                return result
        except (OSError, ValueError):
            pass
        if self.stopped_by_user:
            return {"outcome": "stopped", "exit_code": code,
                    "message": "Enrollment stopped - nothing was saved."}
        return {"outcome": "error", "exit_code": code,
                "message": EXIT_TEXT.get(code, f"main.py exited with code {code}."),
                "reasons": self.log_tail()}

    def log_tail(self, lines: int = 4) -> str:
        """Last lines this session wrote to session.log (e.g. an import traceback)."""
        try:
            with open(SESSION_LOG, encoding="utf-8", errors="replace") as fh:
                fh.seek(self._log_offset)
                tail = [ln.rstrip() for ln in fh.read().splitlines() if ln.strip()]
        except OSError:
            return ""
        return "\n".join(tail[-lines:])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class Launcher:
    """The Tk application: menu, driver picker and running strip."""

    def __init__(self, root: tk.Tk, windowed: bool,
                 screen_size: Optional[Tuple[int, int]] = None) -> None:
        self.root = root
        self.windowed = windowed
        # --screen-size: lay out for exactly this size, in a window of
        # exactly this size, whatever the display reports.
        self.forced_size = screen_size
        self.backend = Backend(root)
        self.session: Optional[Session] = None
        self.backend_online: Optional[bool] = None
        self.last_result = ""
        self._drivers: List[Dict[str, Any]] = []
        self._page = 0
        # "Video stream" toggle: adds main.py --debug-stream to new sessions.
        self.stream_on = False
        # One token for every session this launcher starts, so the URL typed
        # into the laptop keeps working; FATIGUE_STREAM_TOKEN makes it permanent.
        self.stream_token = stream_token(os.environ.get("FATIGUE_STREAM_TOKEN"))
        self._enroll_driver: Optional[Dict[str, Any]] = None
        self.api_problem = api_settings_problem()
        if self.api_problem:
            logger.error("API SETTINGS: %s", self.api_problem)

        # The window the current layout is for. Set by _apply_size, never
        # read back from the display mid-screen.
        self.win_x = self.win_y = 0
        self.win_w = self.win_h = 0
        self.scale = 1.0
        self._mode: Optional[str] = None            # "full" or "strip"
        self._redraw: Callable[[], None] = lambda: None   # current full screen
        self._refit_job: Optional[str] = None

        # Sizes are set by _apply_size. The other fonts are re-fitted
        # wherever they are used, so shrinking one never shrinks text
        # elsewhere.
        self.font_title = tkfont.Font(family="DejaVu Sans", weight="bold")
        self.font_button = tkfont.Font(family="DejaVu Sans", weight="bold")
        self.font_small = tkfont.Font(family="DejaVu Sans")
        self.font_detail = tkfont.Font(family="DejaVu Sans")
        self.font_url = tkfont.Font(family="DejaVu Sans", weight="bold")
        self.font_strip = tkfont.Font(family="DejaVu Sans", weight="bold")
        self.font_stop = tkfont.Font(family="DejaVu Sans", weight="bold")
        self.font_driver = tkfont.Font(family="DejaVu Sans", weight="bold")

        root.title("Fatigue Detection Launcher")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.bind("<Escape>", lambda _e: root.attributes("-fullscreen", False))
        root.bind("<Configure>", self._on_configure)

        self.container: Optional[tk.Frame] = None
        self.picker_body: Optional[tk.Frame] = None
        self._picker_title = ""
        self.show_menu()
        self._schedule_ping()

    # ---- geometry helpers ------------------------------------------------

    def px(self, value: float) -> int:
        return int(value * self.scale)

    def _target_geometry(self) -> Tuple[int, int, int, int, str]:
        """(x, y, width, height, source) the full window should have now."""
        if self.forced_size:
            x = y = 40 if self.windowed else 0
            return (x, y, *self.forced_size, "--screen-size")
        if self.windowed:
            return 40, 40, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT, "--windowed"
        # The monitor the window is on (before the first map: the one at 0,0).
        # Not winfo_screenwidth/height: the X screen spans every monitor.
        cx = self.root.winfo_x() + self.root.winfo_width() // 2
        cy = self.root.winfo_y() + self.root.winfo_height() // 2
        mon = monitor_at(detect_monitors(), cx, cy)
        if mon is not None:
            return mon.x, mon.y, mon.width, mon.height, f"monitor {mon.name}"
        return (0, 0, self.root.winfo_screenwidth(), self.root.winfo_screenheight(),
                "X screen, no xrandr monitors")

    def _apply_size(self, w: int, h: int, source: str) -> None:
        """Scale and fonts for a ``w`` x ``h`` window (logged: the evidence of a re-fit)."""
        self.win_w, self.win_h = w, h
        self.scale = layout_scale(w, h)
        self.font_title.configure(size=-self.px(FONT_TITLE_PX))
        self.font_small.configure(size=-self.px(FONT_SMALL_PX))
        # One button font for every button: as large as the scale allows
        # while the longest menu label fits a menu cell (half the width, a
        # row never shorter than ROW_MIN_PX; 2 px slack for rounding).
        pad = self.px(4)
        button_px = fit_font(
            self.font_button, MENU_LABELS,
            w // 2 - 2 * pad - 2 * self.px(BUTTON_PADX_PX) - 2,
            self.px(ROW_MIN_PX) - 2 * pad - 2 * self.px(BUTTON_PADY_PX) - 2,
            self.px(FONT_BUTTON_PX))
        logger.info("Display %dx%d (%s), scale %.2f, button font %d px",
                    w, h, source, self.scale, button_px)

    def _on_configure(self, event: tk.Event) -> None:
        """The window manager resized the full window: re-lay out to its real size."""
        if (event.widget is not self.root or self._mode != "full" or self.forced_size
                or (event.width, event.height) == (self.win_w, self.win_h)):
            return
        if self._refit_job is not None:
            self.root.after_cancel(self._refit_job)
        self._refit_job = self.root.after(REFIT_DELAY_MS, self._refit)

    def _refit(self) -> None:
        self._refit_job = None
        w, h = self.root.winfo_width(), self.root.winfo_height()
        if self._mode != "full" or w < 2 or (w, h) == (self.win_w, self.win_h):
            return
        if not self.windowed:
            self.win_x, self.win_y = self.root.winfo_x(), self.root.winfo_y()
        self._apply_size(w, h, "window as sized by the window manager")
        self._redraw()

    def _set_x11_type(self, wm_type: str) -> None:
        """_NET_WM_WINDOW_TYPE hint (X11 only; WMs read it when the window maps)."""
        if sys.platform.startswith("linux"):
            try:
                self.root.attributes("-type", wm_type)
            except tk.TclError:
                pass

    def _show_full_window(self) -> None:
        """
        Fullscreen kiosk on the monitor the launcher is on, or a fixed-size
        window (--windowed / --screen-size), with the layout sized to it.
        Runs on every return from the strip, so no size from an earlier
        screen survives; _on_configure then follows the size the window
        manager really gives (e.g. when it fullscreens to another monitor).
        """
        x, y, w, h, source = self._target_geometry()
        self._mode = "full"
        self.win_x, self.win_y = x, y
        self._apply_size(w, h, source)
        self.root.withdraw()
        self.root.attributes("-topmost", False)
        self._set_x11_type("normal")
        # A forced size outside --windowed is borderless: no title bar eats
        # into it and no window manager resizes it.
        self.root.overrideredirect(bool(self.forced_size) and not self.windowed)
        if self.windowed or self.forced_size:
            self.root.attributes("-fullscreen", False)
            self.root.geometry(f"{w}x{h}+{x}+{y}")
        else:
            self.root.geometry(f"{w}x{h}+{x}+{y}")
            self.root.attributes("-fullscreen", True)
        self.root.deiconify()

    def _show_strip_window(self) -> None:
        """Collapse to a bar along the bottom edge that stays above the preview."""
        h = self.px(STRIP_PX)
        self._mode = "strip"
        self.root.withdraw()
        self.root.attributes("-fullscreen", False)
        # Window managers keep dock-type windows above normal ones; topmost
        # covers WMs that ignore the type, and the poll loop re-lifts too.
        self._set_x11_type("dock")
        self.root.attributes("-topmost", True)
        self.root.geometry(f"{self.win_w}x{h}+{self.win_x}+{self.win_y + self.win_h - h}")
        self.root.deiconify()

    def _clear(self) -> tk.Frame:
        """Replace the screen container so grid weights never leak between screens."""
        if self.container is not None:
            self.container.destroy()
        self.container = tk.Frame(self.root, bg=BG)
        self.container.pack(fill="both", expand=True)
        return self.container

    def _new_picker_body(self) -> tk.Frame:
        """Same idea for the picker's body (message / list / confirm views)."""
        if self.picker_body is not None and self.picker_body.winfo_exists():
            self.picker_body.destroy()
        self.picker_body = tk.Frame(self.container, bg=BG)
        self.picker_body.grid(row=1, column=0, sticky="nsew")
        return self.picker_body

    def _button(self, parent: tk.Widget, text: str, command: Callable[[], None],
                bg: str = BTN, font: Optional[tkfont.Font] = None) -> tk.Button:
        # No wraplength: labels carry their own line breaks and the button
        # font is fitted to them (_apply_size), so Tk never re-wraps one.
        return tk.Button(
            parent, text=text, command=command, font=font or self.font_button,
            bg=bg, fg=FG, activebackground=BTN_ACTIVE, activeforeground=FG,
            relief="flat", bd=0, highlightthickness=0,
            padx=self.px(BUTTON_PADX_PX), pady=self.px(BUTTON_PADY_PX), cursor="hand2",
        )

    def _shrink_to_height(self, widgets: Sequence[tk.Widget], font: tkfont.Font, max_h: int,
                          start_px: int) -> None:
        """
        Size ``font`` (used by ``widgets`` alone) from ``start_px`` down
        until none is taller than ``max_h`` px - with wrapped text, or
        glyphs drawn from a taller fallback font.
        """
        size = start_px
        while True:
            font.configure(size=-size)
            self.root.update_idletasks()    # Tk re-measures font users at idle time
            if size <= MIN_FONT_PX or max(w.winfo_reqheight() for w in widgets) <= max_h:
                return
            size -= 1

    # ---- backend status --------------------------------------------------

    def _schedule_ping(self) -> None:
        self.backend.ping(self._on_ping)

    def _on_ping(self, online: bool) -> None:
        self.backend_online = bool(online)
        self._refresh_status()
        self.root.after(int(PING_INTERVAL_S * 1000), self._schedule_ping)

    def _refresh_status(self) -> None:
        pill = getattr(self, "status_pill", None)
        if pill is None or not pill.winfo_exists():
            return
        color = {None: UNKNOWN, True: ONLINE, False: OFFLINE}[self.backend_online]
        pill.configure(text=PILL_TEXT[self.backend_online], bg=color)

    def _header(self, parent: tk.Widget, title: str) -> tk.Frame:
        """Title on the left, device id + backend pill on the right."""
        bar = tk.Frame(parent, bg=BG, height=self.px(HEADER_PX))
        bar.grid_propagate(False)
        bar.columnconfigure(0, weight=1)
        host = urlparse(config.API_BASE_URL).netloc or config.API_BASE_URL
        # The API warning is loud on purpose: without these settings the
        # backend is localhost and every enrollment fails.
        if self.api_problem:
            info, keep, info_padx = f"NO API SETTINGS ({host})", "NO API SETTINGS", self.px(6)
        else:
            info, keep, info_padx = f"Device {config.DEVICE_ID}  ·  {host}", "", 0
        # One line at any width: the pill keeps room for its longest text,
        # title and info share the rest (minus the paddings below; borders
        # are 0) and are shortened with "…" if they must be - the title
        # first, but never into the warning's first words.
        pill_w = max(self.font_small.measure(t) for t in PILL_TEXT.values()) + 4 * self.px(8)
        room = self.win_w - pill_w - 2 * self.px(8) - 2 * self.px(6) - 2 * info_padx - 2
        title = elide(self.font_title, title, room - self.font_small.measure(keep))
        info = elide(self.font_small, info, room - self.font_title.measure(title))
        label = dict(bd=0, highlightthickness=0, pady=0)
        tk.Label(bar, text=title, font=self.font_title, bg=BG, fg=FG, anchor="w",
                 padx=0, **label).grid(row=0, column=0, sticky="nsw", padx=self.px(8))
        if self.api_problem:
            tk.Label(bar, text=info, font=self.font_small, bg=WARN_BG, fg=FG,
                     padx=info_padx, **label).grid(
                row=0, column=1, sticky="nse", padx=self.px(6), pady=self.px(4))
        elif info:
            tk.Label(bar, text=info, font=self.font_small, bg=BG, fg="#bbbbbb",
                     padx=0, **label).grid(row=0, column=1, sticky="nse", padx=self.px(6))
        self.status_pill = tk.Label(bar, font=self.font_small, fg=FG, bg=UNKNOWN,
                                    padx=self.px(8), pady=self.px(3), bd=0)
        self.status_pill.grid(row=0, column=2, sticky="nse", padx=self.px(8))
        bar.rowconfigure(0, weight=1)
        self._refresh_status()
        return bar

    # ---- screens ---------------------------------------------------------

    def _show(self, draw: Callable[[], None]) -> None:
        """
        Draw a full-window screen, re-fitting the window first when coming
        back from the strip; a later resize redraws it with ``draw`` too.
        """
        if self._mode != "full":
            self._show_full_window()
        self._redraw = draw
        draw()

    def show_menu(self) -> None:
        self._show(self._draw_menu)

    def _draw_menu(self) -> None:
        f = self._clear()
        f.columnconfigure((0, 1), weight=1, uniform="col")
        f.rowconfigure(0, weight=0)
        for r in (1, 2, 3):
            f.rowconfigure(r, weight=1, minsize=self.px(ROW_MIN_PX))
        f.rowconfigure(4, weight=0)
        pad = self.px(4)

        self._header(f, "Driver Fatigue Detection").grid(
            row=0, column=0, columnspan=2, sticky="nsew")

        self._button(f, LABEL_ENROLL, self.show_driver_picker, bg=BTN_PRIMARY).grid(
            row=1, column=0, sticky="nsew", padx=pad, pady=pad)
        self._button(f, LABEL_PREDRIVE,
                     lambda: self.start_session("Pre-drive assessment")).grid(
            row=1, column=1, sticky="nsew", padx=pad, pady=pad)
        self._button(f, LABEL_MONITORING,
                     lambda: self.start_session("Monitoring only (test)")).grid(
            row=2, column=0, sticky="nsew", padx=pad, pady=pad)
        self._button(f, LABEL_IGNITION,
                     lambda: self.start_session("Follow ignition")).grid(
            row=2, column=1, sticky="nsew", padx=pad, pady=pad)

        self._button(f, self._stream_label(), self._toggle_stream,
                     bg=BTN_ASSIGNED if self.stream_on else BTN).grid(
            row=3, column=0, sticky="nsew", padx=pad, pady=pad)
        self._button(f, LABEL_QUIT, self.quit, bg=BTN_QUIT).grid(
            row=3, column=1, sticky="nsew", padx=pad, pady=pad)

        note, color = self.last_result, "#dddddd"
        if self.api_problem and not note:
            note, color = self.api_problem, "#ff8a80"
        tk.Label(f, text=note, font=self.font_small, bg=BG, fg=color, anchor="w",
                 justify="left", wraplength=self.win_w - self.px(24)).grid(
            row=4, column=0, columnspan=2, sticky="nsew", padx=self.px(8), pady=(0, pad))
        if self.stream_on:
            # Large enough to read off the panel and type into a laptop, and
            # never cut off: shrunk below the title size if it must be.
            url = f"Video: {self._stream_url()}"
            fit_font(self.font_url, [url], self.win_w - 2 * self.px(8) - 4,
                     self.px(HEADER_PX), self.px(FONT_TITLE_PX))
            tk.Label(f, text=url, font=self.font_url, bg=BG, fg="#9ae6b4", anchor="w").grid(
                row=5, column=0, columnspan=2, sticky="nsew", padx=self.px(8), pady=(0, pad))

    def _stream_label(self) -> str:
        return LABEL_STREAM.format("ON" if self.stream_on else "OFF")

    def _toggle_stream(self) -> None:
        self.stream_on = not self.stream_on
        logger.info("Debug video stream %s for new sessions", "ON" if self.stream_on else "OFF")
        self.show_menu()

    def _stream_url(self) -> str:
        return f"http://{lan_address()}:{config.DEBUG_STREAM_PORT}/{self.stream_token}/"

    def show_driver_picker(self) -> None:
        """Fetch the roster, then render it as pages of large buttons."""
        self._show(lambda: self._picker_frame("Enroll: choose driver"))
        self._page = 0
        self._picker_message("Loading drivers…")
        self.backend.drivers(self._on_drivers)

    def _picker_frame(self, title: str) -> None:
        """Header + an empty body row, which the picker's views fill."""
        f = self._clear()
        f.columnconfigure(0, weight=1)
        f.rowconfigure(1, weight=1)
        self._header(f, title).grid(row=0, column=0, sticky="nsew")
        self._picker_title = title

    def _picker_view(self, draw: Callable[[], None]) -> None:
        """Show one picker body view; a resize rebuilds the frame around it."""
        title = self._picker_title

        def redraw() -> None:
            self._picker_frame(title)
            draw()
        self._redraw = redraw
        draw()

    def _picker_message(self, text: str, retry: bool = False) -> None:
        self._picker_view(lambda: self._draw_picker_message(text, retry))

    def _draw_picker_message(self, text: str, retry: bool) -> None:
        body = self._new_picker_body()
        body.columnconfigure((0, 1), weight=1, uniform="col")
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=0, minsize=self.px(ROW_MIN_PX))
        tk.Label(body, text=text, font=self.font_title, bg=BG, fg=FG,
                 wraplength=self.win_w - self.px(24)).grid(
            row=0, column=0, columnspan=2, sticky="nsew")
        pad = self.px(4)
        self._button(body, "◀  Back", self.show_menu, bg=BTN_QUIT).grid(
            row=1, column=0, sticky="nsew", padx=pad, pady=pad)
        if retry:
            self._button(body, "Retry", self.show_driver_picker, bg=BTN_PRIMARY).grid(
                row=1, column=1, sticky="nsew", padx=pad, pady=pad)

    def _on_drivers(self, drivers: Optional[List[Dict[str, Any]]]) -> None:
        if self.picker_body is None or not self.picker_body.winfo_exists():
            return                      # user already went back
        if drivers is None:
            self._picker_message("Could not fetch drivers - backend unreachable?", retry=True)
            return
        if not drivers:
            self._picker_message("No drivers on the backend yet. Add one in the portal.", retry=True)
            return
        # The driver assigned to this unit is the one being enrolled nearly
        # every time, so it goes first (and is coloured differently).
        mine = str(config.DEVICE_ID)
        self._drivers = sorted(
            drivers,
            key=lambda d: (str(d.get("device_id") or "") != mine,
                           str(d.get("full_name") or "").lower()),
        )
        self._render_driver_page()

    def _render_driver_page(self) -> None:
        self._picker_view(self._draw_driver_page)

    def _draw_driver_page(self) -> None:
        body = self._new_picker_body()
        avail_h = self.win_h - self.px(HEADER_PX)
        row_h = self.px(ROW_MIN_PX)
        per_page = max(3, (avail_h - row_h) // row_h)     # leave one row for nav
        pages = max(1, -(-len(self._drivers) // per_page))
        self._page = min(self._page, pages - 1)
        chunk = self._drivers[self._page * per_page:(self._page + 1) * per_page]
        mine = str(config.DEVICE_ID)
        pad = self.px(3)

        body.columnconfigure((0, 1, 2), weight=1, uniform="nav")
        for r in range(per_page):
            body.rowconfigure(r, weight=1, minsize=row_h)
        body.rowconfigure(per_page, weight=0, minsize=row_h)

        # Two lines per row; a name too long for one line is shortened, not
        # wrapped into a third. Own font, shrunk if a row still comes out
        # too tall (★ / ✓ from a taller fallback font).
        text_w = self.win_w - 2 * pad - 2 * self.px(12) - 2
        button_px = -int(self.font_button.cget("size"))
        self.font_driver.configure(size=-button_px)
        rows = []
        for r, d in enumerate(chunk):
            assigned = str(d.get("device_id") or "") == mine
            enrolled = bool(d.get("is_enrolled"))
            name = d.get("full_name") or f"Driver {d.get('id')}"
            tag = "✓ enrolled" if enrolled else "not enrolled"
            if assigned:
                tag += f"  ·  ★ this unit ({mine})"
            btn = self._button(
                body, "\n".join(elide(self.font_driver, line, text_w) for line in (name, tag)),
                lambda d=d: self._confirm_enroll(d),
                bg=BTN_ASSIGNED if assigned else BTN, font=self.font_driver,
            )
            btn.configure(anchor="w", justify="left", padx=self.px(12))
            btn.grid(row=r, column=0, columnspan=3, sticky="nsew", padx=pad, pady=pad)
            rows.append(btn)
        if rows:
            self._shrink_to_height(rows, self.font_driver, row_h - 2 * pad, button_px)

        self._button(body, "◀  Back", self.show_menu, bg=BTN_QUIT).grid(
            row=per_page, column=0, sticky="nsew", padx=pad, pady=pad)
        if pages > 1:
            prev = self._button(body, "▲ Prev", lambda: self._flip_page(-1))
            nxt = self._button(body, f"▼ Next  ({self._page + 1}/{pages})",
                               lambda: self._flip_page(+1))
            prev.grid(row=per_page, column=1, sticky="nsew", padx=pad, pady=pad)
            nxt.grid(row=per_page, column=2, sticky="nsew", padx=pad, pady=pad)
            if self._page == 0:
                prev.configure(state="disabled")
            if self._page == pages - 1:
                nxt.configure(state="disabled")

    def _flip_page(self, delta: int) -> None:
        self._page += delta
        self._render_driver_page()

    def _confirm_enroll(self, driver: Dict[str, Any]) -> None:
        """One extra tap before a 60 s calibration - picking wrongly is costly."""
        self._picker_view(lambda: self._draw_confirm(driver))

    def _draw_confirm(self, driver: Dict[str, Any]) -> None:
        body = self._new_picker_body()
        body.columnconfigure((0, 1), weight=1, uniform="col")
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=0, minsize=self.px(ROW_MIN_PX))
        name = driver.get("full_name") or f"Driver {driver.get('id')}"
        steps = ("Positioning (text guidance on this screen), face capture, closed-eye check "
                 "(eyes shut on the beep, open on the long beep), then 60 s calibration - "
                 "about 2-3 minutes.")
        note = (f"Already enrolled - this will replace their calibration.\n{steps}"
                if driver.get("is_enrolled") else steps)
        if self.api_problem:
            note += f"\n\n{self.api_problem}"
        if self.stream_on:
            note += f"\n\nPosition the driver with the video stream:\n{self._stream_url()}"
        else:
            note += ("\n\nThe panel shows no camera image. For fine positioning, go back "
                     "and turn the Video stream ON (laptop / phone view).")
        pad = self.px(4)
        text = tk.Label(body, text=f"Enrol {name} (id {driver.get('id')})?\n\n{note}",
                        font=self.font_detail, bg=BG, fg=FG, justify="left",
                        wraplength=self.win_w - self.px(24))
        text.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self._shrink_to_height([text], self.font_detail, self.win_h - self.px(HEADER_PX)
                               - self.px(ROW_MIN_PX), self.px(FONT_SMALL_PX))
        self._button(body, "◀  Back", self._render_driver_page, bg=BTN_QUIT).grid(
            row=1, column=0, sticky="nsew", padx=pad, pady=pad)
        self._button(body, "Start enrollment", bg=BTN_PRIMARY,
                     command=lambda: self.start_enrollment(driver)).grid(
            row=1, column=1, sticky="nsew", padx=pad, pady=pad)

    # ---- sessions ----------------------------------------------------------

    def start_enrollment(self, driver: Dict[str, Any]) -> None:
        """Enrol ``driver`` - the id sent is exactly the roster's ``id``."""
        name = driver.get("full_name") or f"Driver {driver.get('id')}"
        try:
            driver_id = int(driver["id"])
        except (KeyError, TypeError, ValueError):
            logger.error("Roster entry has no usable id: %r", driver)
            self.last_result = f"Enroll {name}: roster entry has no usable id {driver.get('id')!r}"
            self.show_menu()
            return
        logger.info("Enrollment requested for driver id %s (%s)", driver_id, name)
        self._enroll_driver = driver
        self.start_session(f"Enroll {name}", ["--enroll", "--driver-id", str(driver_id)],
                           enroll=True)

    def start_session(self, label: str, flags: Optional[List[str]] = None,
                      enroll: bool = False) -> None:
        if self.session and self.session.running:
            return
        flags = list(SESSIONS[label] if flags is None else flags)
        # The data / enrollment screen fills the panel above the STOP strip.
        flags += ["--fullscreen", "--reserve-bottom", str(self.px(STRIP_PX))]
        extra_env = {}
        if self.stream_on:
            flags += ["--debug-stream", str(config.DEBUG_STREAM_PORT)]
            extra_env["FATIGUE_STREAM_TOKEN"] = self.stream_token
            logger.info("Video stream for this session: %s", self._stream_url())
        session = Session(label, flags, enroll=enroll, extra_env=extra_env)
        try:
            session.start()
        except OSError as exc:
            logger.exception("Could not start %s", label)
            self.last_result = f"{label}: failed to start ({exc})"
            self.session = None
            self.show_menu()
            return
        self.session = session
        self._show_running(session)

    def _show_running(self, session: Session) -> None:
        """Slim bottom strip: what is running + a STOP button."""
        f = self._clear()
        self._show_strip_window()
        pad = self.px(4)
        stop = self._button(f, "■  STOP", self._stop_session, bg=BTN_STOP, font=self.font_stop)
        self._shrink_to_height([stop], self.font_stop, self.px(STRIP_PX) - 2 * pad,
                               -int(self.font_button.cget("size")))
        stop.grid(row=0, column=1, sticky="nsew", padx=pad, pady=pad)
        stop_w = max(self.px(120), stop.winfo_reqwidth() + 2 * pad)
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=0, minsize=stop_w)
        f.rowconfigure(0, weight=1)
        text = f"{session.label} running…"
        if self.stream_on:
            text = f"{self._stream_url()}\n{session.label} running…"
        # A long driver name or the stream URL must stay whole: shrink the
        # text below the title size if it must be.
        fit_font(self.font_strip, [text, f"Stopping {session.label}…"],
                 self.win_w - stop_w - 2 * self.px(10) - 4, self.px(STRIP_PX) - 4,
                 self.px(FONT_TITLE_PX))
        self.running_label = tk.Label(
            f, text=text, font=self.font_strip, bg=BG, fg=FG, anchor="w", justify="left",
            padx=self.px(10), pady=0, bd=0)
        self.running_label.grid(row=0, column=0, sticky="nsew")
        self.root.after(CHILD_POLL_MS, self._poll_session)

    def _stop_session(self) -> None:
        if self.session:
            self.session.stop()
            self.running_label.configure(text=f"Stopping {self.session.label}…")

    def _poll_session(self) -> None:
        if self.session is None:
            return
        if self.session.running:
            self.root.lift()            # stay above the cv2 preview
            self.root.after(CHILD_POLL_MS, self._poll_session)
            return
        session, self.session = self.session, None
        if session.enroll:
            self.show_enroll_result(session.enroll_result())
            return
        self.last_result = session.result_text()
        logger.info(self.last_result)
        self.show_menu()

    # Outcome -> (headline, colour) for the enrollment result screen.
    ENROLL_HEADLINES: Dict[str, tuple] = {
        "saved": ("SAVED", RESULT_OK),
        "saved_unverified": ("SAVED - not verified", RESULT_WARN),
        "dropped": ("SAVED - backend DROPPED the closed-eye baseline", RESULT_WARN),
        "refused": ("REFUSED - nothing saved", RESULT_BAD),
        "rejected": ("BACKEND REJECTED - nothing saved", RESULT_BAD),
        "bad_driver": ("WRONG DRIVER - nothing saved", RESULT_BAD),
        "stopped": ("STOPPED - nothing saved", UNKNOWN),
        "error": ("FAILED - nothing saved", RESULT_BAD),
    }

    def show_enroll_result(self, result: Dict[str, Any]) -> None:
        """Full-screen outcome of an enrollment, every case spelled out."""
        driver = self._enroll_driver or {}
        name = result.get("driver_name") or driver.get("full_name") or \
            f"Driver {result.get('driver_id', driver.get('id'))}"
        headline, color = self.ENROLL_HEADLINES.get(
            str(result.get("outcome")), (f"UNKNOWN OUTCOME {result.get('outcome')!r}", RESULT_BAD))
        logger.info("Enrollment result for %s: %s - %s %s", name, result.get("outcome"),
                    result.get("message"), result.get("reasons", ""))
        self.last_result = f"Enroll {name}: {headline}"

        details = [str(result.get("message") or "")]
        if result.get("reasons"):
            details.append(f"Reason: {result['reasons']}")
        numbers = result.get("baselines") or {}
        if numbers:
            parts = [f"{k.replace('_baseline', '').replace('_', ' ')} {v:.3f}"
                     for k, v in numbers.items()]
            if result.get("contrast") is not None:
                parts.append(f"closed/open {result['contrast']:.2f}")
            details.append("  ·  ".join(parts))
        details.append(f"driver id {result.get('driver_id', driver.get('id'))} on "
                       f"{result.get('api_base_url', config.API_BASE_URL)}  ·  "
                       f"exit code {result.get('exit_code')}")

        again = self._enroll_driver if result.get("outcome") != "saved" else None

        def draw() -> None:
            f = self._clear()
            f.columnconfigure((0, 1), weight=1, uniform="col")
            f.rowconfigure(2, weight=1)
            f.rowconfigure(3, weight=0, minsize=self.px(ROW_MIN_PX))
            pad = self.px(4)
            self._header(f, f"Enrollment: {name}").grid(
                row=0, column=0, columnspan=2, sticky="nsew")
            head = tk.Label(f, text=headline, font=self.font_button, bg=color, fg=FG,
                            wraplength=self.win_w - self.px(24), pady=self.px(8))
            head.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=pad, pady=pad)
            # The details (a log tail on a crash) shrink rather than push
            # the Menu button off the bottom.
            text = tk.Label(f, text="\n\n".join(d for d in details if d), font=self.font_detail,
                            bg=BG, fg=FG, justify="left", anchor="nw",
                            wraplength=self.win_w - self.px(24))
            text.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=self.px(10), pady=pad)
            self._shrink_to_height([text], self.font_detail,
                                   self.win_h - self.px(HEADER_PX) - head.winfo_reqheight()
                                   - 4 * pad - self.px(ROW_MIN_PX), self.px(FONT_SMALL_PX))
            self._button(f, "◀  Menu", self.show_menu, bg=BTN_QUIT).grid(
                row=3, column=0, sticky="nsew", padx=pad, pady=pad)
            if again is not None:
                self._button(f, "Enroll again", lambda: self._confirm_enroll_screen(again),
                             bg=BTN_PRIMARY).grid(row=3, column=1, sticky="nsew",
                                                  padx=pad, pady=pad)
        self._show(draw)

    def _confirm_enroll_screen(self, driver: Dict[str, Any]) -> None:
        """The confirm screen again (it lives inside the picker's frame)."""
        self._show(lambda: self._picker_frame("Enroll: confirm"))
        self._confirm_enroll(driver)

    def quit(self) -> None:
        session = self.session
        if session and session.proc and session.running:
            session.stop()
            try:
                session.proc.wait(timeout=STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                session.proc.kill()
            session.close()
        self.root.destroy()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Touchscreen launcher for main.py")
    parser.add_argument("--windowed", action="store_true",
                        help="run in a panel-sized (800x480) window instead of fullscreen "
                             "(development)")
    parser.add_argument("--screen-size", type=parse_screen_size, metavar="WxH",
                        help="lay out for exactly WxH instead of the detected screen, in a "
                             "borderless WxH window at 0,0 (a decorated one at 40,40 with "
                             "--windowed) - e.g. 800x480 to check the panel layout on a "
                             "desktop, or to override a wrong detection on the unit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging()
    logger.info("=== Launcher starting (device %s, API %s) ===",
                config.DEVICE_ID, config.API_BASE_URL)
    root = tk.Tk()
    app = Launcher(root, windowed=args.windowed, screen_size=args.screen_size)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
