"""
Display: the in-vehicle window never shows camera pixels during pre-drive
and monitoring (only enrollment has a live view, see test_enrollment.py),
frames are annotated only when someone sees them, and the data screen
renders every state.
"""

import dataclasses
import inspect
import time

import numpy as np
import pytest

from fakes import CAL, FakeAPI, FakeExtractor, MarkedCam, ScriptedIgnition, shows_camera
from modules.phase import Phase
from modules.session_display import (BG, FACE_NONE, FACE_OK, FACE_SEARCHING, FACE_UNRECOGNISED,
                                     ScreenState, render)


def test_frame_annotated_only_when_shown_as_video(m):
    m._display_mode, m._display_available = "data", True
    f = m.capture_frame()
    m.display_text(f, "X")
    m.draw_banner(f, "B")
    assert not f.any(), "data screen: the camera frame is never drawn on"
    assert m._screen.status[-1][0] == "X" and m._screen.phase == "B", "text goes to the data screen"

    m._display_mode = "video"
    f = m.capture_frame()
    m.display_text(f, "X")
    assert f.any(), "--show-video draws on the camera frame"

    m._display_available = False
    f = m.capture_frame()
    m.display_text(f, "X")
    m.draw_banner(f, "X")
    assert not f.any(), "--no-preview draws nothing"


@pytest.mark.parametrize("state", [
    ScreenState(),
    ScreenState(phase="PRE-DRIVE", level="ALERT", frs=0.2, face=FACE_OK, countdown=12.3,
                ear=0.28, mar=0.41, perclos=3.2, blink_per_min=12, yaw=3, pitch=-4,
                driver_name="A Very Long Driver Name That Must Be Shrunk To Fit", fps=20.0),
    ScreenState(level="DANGER", frs=0.8, alert_banner="MICROSLEEP 1.4s", face=FACE_OK),
    ScreenState(level="FAULT", face=FACE_NONE, face_detail="11.2s",
                status=[("NO FACE 11.2s  [ALERT]", (255, 0, 255))]),
    ScreenState(face=FACE_SEARCHING, status=[("IDENTIFYING 0/3", (255, 255, 255)),
                                             ("7s left", (160, 160, 160))]),
    ScreenState(face=FACE_UNRECOGNISED, level="ALERT", frs_note="learning EAR baseline",
                baseline_label="defaults"),
    ScreenState(status=[("STARTER LOCKED: " + "VERY LONG REASON " * 6, (0, 0, 255)),
                        ("awaiting operator override (request 41)", (0, 220, 255))]),
])
def test_data_screen_renders_and_keeps_strip_clear(state):
    img = render(state, 800, 480, reserve_bottom=66)
    assert img.shape == (480, 800, 3)
    assert (img[-66:] == np.array(BG, np.uint8)).all(), "launcher strip area left empty"


def test_data_screen_cannot_take_a_camera_image():
    """The pre-drive / monitoring screen has no way to receive camera pixels at all."""
    assert list(inspect.signature(render).parameters) == ["s", "width", "height",
                                                          "reserve_bottom"]
    assert not any(f.name == "image" for f in dataclasses.fields(ScreenState))


@pytest.mark.slow
def test_session_window_never_gets_a_camera_frame(m, rig, hp, monkeypatch):
    """Pre-drive pass -> monitoring with a no-face gap, data screen on: only canvases are shown."""
    shown, camera_on_panel = [], []
    monkeypatch.setattr(m, "show_frame", lambda img: (shown.append(img.shape),
                                                      camera_on_panel.append(shows_camera(img)),
                                                      -1)[2])
    m._camera = MarkedCam()
    m._display_mode, m._display_available, m._reserve_bottom = "data", True, 66
    monkeypatch.setattr(m, "identify_driver_bounded",
                        lambda *a, **k: {"driver_id": 6, "name": "Test Driver", "confidence": 0.6})
    monkeypatch.setattr(m, "load_calibration", lambda *a, **k: (CAL, CAL.provenance()))
    api = FakeAPI()
    m._rate = m.LoopRate()
    ign = m.ForcedIgnition(Phase.PREDRIVE, source="tests")
    am = rig(api, ign)
    from fakes import FakeRecognizer
    assert m.run_predrive_assessment(api, FakeExtractor(cost=0.01), FakeRecognizer(0.0), am, hp,
                                     ign, m._rate)
    n = {"i": 0}

    def face():
        n["i"] += 1
        return not 40 <= n["i"] < 90                    # a no-face gap

    m._ignition = ScriptedIgnition([(5.0, Phase.MONITORING), (99, Phase.PREDRIVE)])
    m.run_monitoring(api, FakeExtractor(cost=0.01, face=face), FakeRecognizer(0.0), am, hp,
                     m._ignition, m._rate)
    assert len(shown) > 20
    assert set(shown) == {(480, 800, 3)}, "only data-screen canvases, never a 640x480 frame"
    assert not any(camera_on_panel), "no camera pixels on the panel in pre-drive / monitoring"


def test_data_screen_redraw_is_throttled(m, monkeypatch):
    shown = []
    monkeypatch.setattr(m, "show_frame", lambda img: (shown.append(1), -1)[1])
    m._display_mode, m._display_available = "data", True
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:
        f = m.capture_frame()                            # 30 fps fake camera
        m.present(f)
    assert 8 <= len(shown) <= 12, f"~{m.config.DATA_SCREEN_HZ:.0f} redraws a second, got {len(shown)}"
