"""
Launcher logic: outcome reporting, environment checks, screen-size
detection, and the layout of every screen at 800x480 (the panel) and
1920x1080 - the layout tests open real Tk windows and skip without a display.
"""

import argparse
import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

tk = pytest.importorskip("tkinter")
import launcher  # noqa: E402

SIZES = [(800, 480), (1920, 1080)]
LONG_NAME = "Maria Fernanda Delacroix-Villanueva y Santiago"


@pytest.fixture
def session_files(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "ENROLL_RESULT", tmp_path / "enroll_result.json")
    monkeypatch.setattr(launcher, "SESSION_LOG", tmp_path / "session.log")
    return tmp_path


def finished(code: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-c", f"import sys; sys.exit({code})"])
    proc.wait()
    return proc


def test_enroll_result_read_from_file(session_files):
    (session_files / "enroll_result.json").write_text(json.dumps(
        {"outcome": "dropped", "exit_code": 7, "message": "m"}))
    s = launcher.Session("Enroll X", [], enroll=True)
    s.proc = finished(7)
    assert s.enroll_result()["outcome"] == "dropped"


def test_enroll_result_falls_back_to_exit_code_and_log(session_files):
    (session_files / "session.log").write_text("ModuleNotFoundError: No module named 'x'\n")
    s = launcher.Session("Enroll X", [], enroll=True)
    s.proc = finished(1)
    result = s.enroll_result()
    assert result["outcome"] == "error" and result["exit_code"] == 1
    assert "ModuleNotFoundError" in result["reasons"]


def test_every_enroll_exit_code_has_text():
    for code in (0, 1, 2, 3, 5, 6, 7):
        assert code in launcher.EXIT_TEXT


def test_api_settings_problem(monkeypatch):
    monkeypatch.setenv("FATIGUE_API_BASE_URL", "http://x/api")
    monkeypatch.setenv("FATIGUE_API_TOKEN", "t")
    assert launcher.api_settings_problem() is None
    monkeypatch.delenv("FATIGUE_API_TOKEN")
    assert "FATIGUE_API_TOKEN" in launcher.api_settings_problem()


# ---- screen size -------------------------------------------------------------

def test_layout_scale():
    assert launcher.layout_scale(800, 480) == pytest.approx(1.5, abs=0.01)
    assert launcher.layout_scale(1920, 1080) == launcher.MAX_SCALE
    # A wide-but-short or tall-but-narrow window scales by the side that runs out.
    assert launcher.layout_scale(1920, 480) == pytest.approx(1.5, abs=0.01)
    assert launcher.layout_scale(800, 1080) == pytest.approx(1.5, abs=0.01)


def test_parse_screen_size():
    assert launcher.parse_screen_size("800x480") == (800, 480)
    assert launcher.parse_screen_size(" 1920X1080 ") == (1920, 1080)
    for bad in ("800", "800x", "x480", "800*480", "300x200"):
        with pytest.raises(argparse.ArgumentTypeError):
            launcher.parse_screen_size(bad)
    assert launcher.parse_args(["--screen-size", "800x480"]).screen_size == (800, 480)
    assert launcher.parse_args([]).screen_size is None


XRANDR = """Monitors: 2
 0: +*HDMI-1 800/154x480/86+0+0  HDMI-1
 1: +VNC-0 1920/508x1080/286+800+0  VNC-0
"""


def test_parse_xrandr_monitors():
    hdmi, vnc = launcher.parse_xrandr_monitors(XRANDR)
    assert hdmi == launcher.Monitor("HDMI-1", 0, 0, 800, 480, True)
    assert vnc == launcher.Monitor("VNC-0", 800, 0, 1920, 1080, False)
    assert launcher.parse_xrandr_monitors("") == []


def test_monitor_at():
    monitors = launcher.parse_xrandr_monitors(XRANDR)
    assert launcher.monitor_at(monitors, 400, 240).name == "HDMI-1"
    assert launcher.monitor_at(monitors, 1500, 500).name == "VNC-0"
    assert launcher.monitor_at(monitors, 5000, 5000).name == "HDMI-1"   # primary
    assert launcher.monitor_at([], 0, 0) is None


# ---- layout ------------------------------------------------------------------

@pytest.fixture(scope="module")
def tk_root():
    """
    One Tk root for every layout test: creating a Tk again after destroying
    one intermittently fails on Windows ('invalid command name
    "tcl_findLibrary"').
    """
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    yield root
    root.destroy()


@pytest.fixture
def make_app(monkeypatch, session_files, tk_root):
    """
    ``make_app(size, windowed=False)``: a real Launcher (backend and network
    stubbed) with the longest texts it can show: no API settings (warning in
    the header and the note line) and a long backend host name.
    """
    monkeypatch.setattr(launcher.Backend, "ping", lambda self, done: None)
    monkeypatch.setattr(launcher.Backend, "drivers", lambda self, done: None)
    monkeypatch.setattr(launcher, "lan_address", lambda: "192.168.100.123")
    monkeypatch.setattr(launcher.config, "API_BASE_URL",
                        "http://fatigue-portal.example-university.edu:8443/api")
    monkeypatch.delenv("FATIGUE_API_BASE_URL", raising=False)
    monkeypatch.delenv("FATIGUE_API_TOKEN", raising=False)

    def make(size, windowed=False):
        app = launcher.Launcher(tk_root, windowed=windowed, screen_size=size)
        tk_root.update()
        return app
    yield make
    # Leave the shared root bare for the next test: no timers, no widgets.
    for job in tk_root.tk.splitlist(tk_root.tk.call("after", "info")):
        tk_root.after_cancel(job)
    for child in tk_root.winfo_children():
        child.destroy()


def widgets(parent):
    for child in parent.winfo_children():
        yield child
        yield from widgets(child)


def cut_off(app):
    """
    Every label / button that needs more room than it got, or sticks out of
    its window (the full window, or the STOP strip while a session runs).
    """
    app.root.update()
    bad = []
    for w in widgets(app.root):
        if not isinstance(w, (tk.Label, tk.Button)) or not w.winfo_ismapped():
            continue
        top = w.winfo_toplevel()
        rw, rh = top.winfo_width(), top.winfo_height()
        x, y = w.winfo_rootx() - top.winfo_rootx(), w.winfo_rooty() - top.winfo_rooty()
        width, height = w.winfo_width(), w.winfo_height()
        if (w.winfo_reqwidth() > width or w.winfo_reqheight() > height
                or x < 0 or y < 0 or x + width > rw or y + height > rh):
            bad.append(f"{w.cget('text')!r}: {width}x{height} at {x},{y} needs "
                       f"{w.winfo_reqwidth()}x{w.winfo_reqheight()} (window {rw}x{rh})")
    return bad


def button(app, text):
    found = [w for w in widgets(app.root) if isinstance(w, tk.Button) and w.cget("text") == text]
    assert len(found) == 1, f"no single button {text!r}"
    return found[0]


def drivers(n=7):
    roster = [{"id": i, "full_name": f"Driver Number {i}", "is_enrolled": i % 2 == 0,
               "device_id": None} for i in range(2, n + 1)]
    return [{"id": 1, "full_name": LONG_NAME, "is_enrolled": True,
             "device_id": str(launcher.config.DEVICE_ID)}] + roster


@pytest.mark.parametrize("size", SIZES)
def test_window_and_layout_use_the_forced_size(make_app, size):
    app = make_app(size)
    assert (app.root.winfo_width(), app.root.winfo_height()) == size
    assert (app.win_w, app.win_h) == size
    assert app.scale == launcher.layout_scale(*size)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("stream_on", [False, True])
def test_menu_fits(make_app, size, stream_on):
    app = make_app(size)
    app.stream_on = stream_on
    app.last_result = f"Enroll {LONG_NAME}: SAVED - backend DROPPED the closed-eye baseline"
    app.show_menu()
    assert cut_off(app) == []
    # The longest labels, measured directly: every line inside the button's
    # padding and within its height - nothing wrapped or clipped.
    for label in (launcher.LABEL_PREDRIVE, launcher.LABEL_IGNITION):
        btn = button(app, label)
        inner_w = btn.winfo_width() - 2 * int(btn.cget("padx"))
        inner_h = btn.winfo_height() - 2 * int(btn.cget("pady"))
        font = app.font_button
        lines = label.split("\n")
        assert max(font.measure(line) for line in lines) <= inner_w, label
        assert font.metrics("linespace") * len(lines) <= inner_h, label
    # Fitting must not overshoot: the panel keeps finger-sized text.
    if size == (800, 480):
        assert -int(app.font_button.cget("size")) >= 20


@pytest.mark.parametrize("size", SIZES)
def test_driver_picker_and_confirm_fit(make_app, size):
    app = make_app(size)
    app.stream_on = True
    app.show_driver_picker()
    roster = drivers()
    app._on_drivers(roster)
    assert cut_off(app) == []
    app._flip_page(+1)
    assert cut_off(app) == []
    app._confirm_enroll(roster[0])      # enrolled + API warning + stream: longest text
    assert cut_off(app) == []


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("result", [
    {"outcome": "dropped", "exit_code": 7, "driver_id": 1, "driver_name": LONG_NAME,
     "message": "Saved, but the backend dropped the closed-eye baseline: it answered "
                "without ear_closed_baseline, so monitoring falls back to the ratio.",
     "baselines": {"ear_baseline": 0.312, "ear_closed_baseline": 0.141, "mar_baseline": 0.402},
     "contrast": 0.45},
    {"outcome": "error", "exit_code": 1, "driver_id": 1,
     "message": "crashed - see logs/session.log",
     "reasons": "Traceback (most recent call last):\n"
                '  File "/home/pi/fatigue-detection/main.py", line 74, in <module>\n'
                "    from modules.face_recognition_module import DriverRecognizer\n"
                "ModuleNotFoundError: No module named 'face_recognition' (is the venv active? "
                "start the launcher with deploy/start_launcher.sh)"},
])
def test_enroll_result_fits(make_app, size, result):
    app = make_app(size)
    app._enroll_driver = drivers()[0]
    app.show_enroll_result(result)
    assert cut_off(app) == []


@pytest.mark.parametrize("size", SIZES)
def test_running_strip_fits(make_app, size):
    """The strip is its own window along the bottom; the full window stays as it was."""
    app = make_app(size)
    app.stream_on = True
    app._show_running(SimpleNamespace(label=f"Enroll {LONG_NAME}"))
    app.root.update()
    strip, strip_h = app.strip, app.px(launcher.STRIP_PX)
    assert strip is not None and strip is not app.root and strip.winfo_ismapped()
    assert (strip.winfo_width(), strip.winfo_height()) == (size[0], strip_h)
    assert strip.winfo_rooty() == size[1] - strip_h
    assert app.root.winfo_ismapped()
    assert (app.root.winfo_width(), app.root.winfo_height()) == size
    assert cut_off(app) == []                   # strip, and the screen behind main.py's


def wait_for(app, condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.root.update()
        time.sleep(0.02)
    return condition()


def settled(app):
    """Wait out the startup geometry checks (re-asserts are allowed after them)."""
    assert wait_for(app, lambda: app._reasserts == 0)


def sleeper(seconds):
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])


def run_session(app, seconds=0.6, label="Pre-drive assessment"):
    """A real child that exits by itself, driven through the launcher's own poll loop."""
    session = launcher.Session(label, [])
    session.proc = sleeper(seconds)
    app.session = session
    app._show_running(session)
    assert wait_for(app, lambda: app.session is None, timeout=10)
    return session


def test_session_never_unmaps_or_reconfigures_the_full_window(make_app, monkeypatch):
    """
    The cause of the Pi bug: the full window was unmapped, re-typed as a dock
    and resized into the strip, then mapped again - and the window manager did
    not put it back. Now nothing touches it until the child has exited.
    """
    app = make_app((800, 480))
    settled(app)
    calls = []
    for name in ("withdraw", "iconify", "overrideredirect", "geometry", "attributes"):
        real = getattr(app.root, name)
        monkeypatch.setattr(app.root, name,
                            lambda *a, _n=name, _r=real: (calls.append((_n, a)), _r(*a))[1])
    typed = []
    real_type = launcher.Launcher._set_x11_type
    monkeypatch.setattr(launcher.Launcher, "_set_x11_type", staticmethod(
        lambda w, t: (typed.append((w, t, bool(w.winfo_ismapped()))), real_type(w, t))[1]))

    session = launcher.Session("Pre-drive assessment", [])
    session.proc = sleeper(30)
    app.session = session
    app._show_running(session)
    app.root.update()
    try:
        assert calls == [], "the full window must not be re-configured for a session"
        assert typed == [(app.strip, "dock", False)], "strip typed before its first map"
        assert int(app.strip.attributes("-topmost")) == 1
        assert app.root.winfo_ismapped()
        assert app._actual_geometry() == app._home
    finally:
        session.proc.kill()
        session.proc.wait()
    assert wait_for(app, lambda: app.session is None)
    assert app.strip is None
    assert not any(n in ("withdraw", "iconify") for n, _ in calls), \
        "the restore must not unmap the window either"


def misplace_after_restore(app, monkeypatch, wrong):
    """Make the 'window manager' put the window somewhere else right after the restore."""
    real = app._restore_full_window

    def restore(stage):
        real(stage)
        wrong()
    monkeypatch.setattr(app, "_restore_full_window", restore)


def test_full_window_restored_after_the_window_manager_misplaces_it(make_app, monkeypatch,
                                                                    caplog):
    """The Pi symptom - window left in the strip's place - is detected, re-asserted, logged."""
    caplog.set_level("INFO", logger="launcher")
    app = make_app((800, 480))
    settled(app)
    home = app._home
    assert home == (0, 0, 800, 480)
    menu_layout = (app.win_w, app.win_h, app.scale)
    misplace_after_restore(app, monkeypatch, lambda: app.root.geometry("800x66+0+414"))

    run_session(app)
    assert wait_for(app, lambda: app._actual_geometry() == home
                    and "re-fit complete" in caplog.text)
    assert (app.win_w, app.win_h, app.scale) == menu_layout
    assert app.last_result.endswith("finished OK")
    assert cut_off(app) == []
    log = caplog.text
    assert "Window [startup]" in log
    assert "Window [child exited: Pre-drive assessment, exit code 0]" in log
    assert "actual 800x66+0+414" in log and "re-asserting (1/3)" in log
    assert "Window [after Pre-drive assessment" in log and "re-fit complete" in log


def test_kiosk_fullscreen_restored_after_a_session(make_app, monkeypatch, caplog):
    """Kiosk mode: fullscreen is re-asserted (removed and re-added) when the WM drops it."""
    caplog.set_level("INFO", logger="launcher")
    monkeypatch.setattr(launcher, "detect_monitors", lambda: [])
    app = make_app(None)
    settled(app)
    home = app._home
    assert int(app.root.attributes("-fullscreen")) == 1
    x, y, w, h = home

    def wm_drops_fullscreen():
        app.root.attributes("-fullscreen", False)
        app.root.geometry(f"{w}x66+{x}+{y + h - 66}")
    misplace_after_restore(app, monkeypatch, wm_drops_fullscreen)

    run_session(app, label="Monitoring only (test)")
    assert wait_for(app, lambda: app._actual_geometry() == home
                    and int(app.root.attributes("-fullscreen")) == 1
                    and "re-fit complete" in caplog.text)
    assert (app.win_w, app.win_h) == (w, h)
    assert "re-asserting" in caplog.text


def test_child_exit_logs_geometry_three_times(make_app, caplog):
    """Startup, immediately after the child exits, and after the re-fit - with both sizes."""
    caplog.set_level("INFO", logger="launcher")
    app = make_app((800, 480))
    settled(app)
    run_session(app)
    assert wait_for(app, lambda: "Window [after Pre-drive assessment, +1500 ms]" in caplog.text)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Window [")]
    stages = [line.split("]")[0] for line in lines]
    assert stages[0] == "Window [startup"
    exited = stages.index("Window [child exited: Pre-drive assessment, exit code 0")
    assert "Window [after Pre-drive assessment, +150 ms" in stages[exited + 1:]
    for line in lines:
        assert "requested 800x480+0+0" in line and "actual " in line


def test_layout_follows_the_window_and_refits_after_a_session(make_app):
    """
    The window manager's size wins over the one asked for, and coming back
    from a session re-fits from scratch instead of keeping an earlier size.
    """
    app = make_app(None, windowed=True)
    panel = (launcher.config.DISPLAY_WIDTH, launcher.config.DISPLAY_HEIGHT)
    assert (app.win_w, app.win_h) == panel
    settled(app)                                # the startup size is recorded first
    app.root.geometry("1100x700")
    assert wait_for(app, lambda: (app.win_w, app.win_h) == (1100, 700))
    assert app.scale == launcher.layout_scale(1100, 700)
    assert cut_off(app) == []

    session = launcher.Session("Pre-drive assessment", [])
    session.proc = finished(0)
    app.session = session
    app._show_running(session)
    app._poll_session()                 # the child has exited -> back to the menu
    app.root.update()
    assert (app.win_w, app.win_h) == panel
    assert (app.root.winfo_width(), app.root.winfo_height()) == panel
    assert app.scale == launcher.layout_scale(*panel)
    assert "finished OK" in app.last_result
    assert cut_off(app) == []


def test_kiosk_sizes_to_its_monitor_not_the_x_screen(make_app, monkeypatch):
    app = make_app((800, 480))                  # window at 0,0, 800x480
    app.forced_size = None                      # as if started without the flag
    monkeypatch.setattr(launcher, "detect_monitors",
                        lambda: launcher.parse_xrandr_monitors(XRANDR))
    assert app._target_geometry() == (0, 0, 800, 480, "monitor HDMI-1")
    monkeypatch.setattr(launcher, "detect_monitors", lambda: [])
    _, _, w, h, _ = app._target_geometry()
    assert (w, h) == (app.root.winfo_screenwidth(), app.root.winfo_screenheight())
