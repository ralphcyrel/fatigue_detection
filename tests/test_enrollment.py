"""
Enrollment: positioning guidance, early refusals, the result file the
launcher reads, the enrollment screen's live mirror view, and a full run on
the enrollment screen.
"""

import json
import sys
import time
import types

import numpy as np
import pytest

from fakes import (FakeAPI, FakeRect, MarkedCam, ScriptedIgnition, close_eyes, shows_camera,
                   synthetic_landmarks)
from modules.api import CALIBRATION_SOURCE_DEVICE, Calibration
from modules.calibration import CalibrationManager, ClosedEyeCapture
from modules.phase import Phase
from modules.session_display import BG, EnrollState, enroll_text_layout, render_enroll

W, H = 640, 480


@pytest.mark.parametrize("box, ok, text", [
    ((20, 120, 260, 360), False, "Move to your left"),      # image left = driver's right
    ((380, 120, 620, 360), False, "Move to your right"),
    ((200, 120, 440, 360), True, "Good - hold still"),
    ((280, 200, 360, 280), False, "Move closer to the camera"),
    ((50, 0, 600, 480), False, "Move back from the camera"),
    ((200, 0, 440, 200), False, "Move down (sit lower or tilt the camera up)"),
    (None, False, "Face the camera"),
])
def test_position_guidance(m, box, ok, text):
    assert m.position_guidance(box, W, H) == (ok, text)


def test_flipped_camera_swaps_left_and_right(m, monkeypatch):
    monkeypatch.setattr(m, "ENROLL_CAMERA_HFLIP", True)
    assert m.position_guidance((20, 120, 260, 360), W, H)[1] == "Move to your right"


class RosterAPI:
    token, base_url, last_enroll_error = "t", "http://backend/api", None

    def __init__(self, roster):
        self.roster = roster

    def get_drivers(self):
        return self.roster


def test_backend_unreachable_refused_before_capture(m):
    code = m.run_enrollment(RosterAPI(None), extractor=None, driver_id=3)
    assert code == 5 and m._enroll_report["outcome"] == "rejected"


def test_unknown_driver_id_refused_before_capture(m):
    code = m.run_enrollment(RosterAPI([{"id": 4, "full_name": "Ana"}]), extractor=None, driver_id=3)
    assert code == 2 and m._enroll_report["outcome"] == "bad_driver"


def test_result_file_when_stopped(m, tmp_path):
    out = tmp_path / "r.json"
    m.write_enroll_result(str(out), 3, 0, interrupted=True)
    result = json.loads(out.read_text())
    assert result["outcome"] == "stopped" and result["driver_id"] == 3


LEFT, RIGHT = (200, 30, 30), (30, 30, 200)         # BGR halves of a test frame


def two_halves():
    frame = np.empty((H, W, 3), np.uint8)
    frame[:, :W // 2], frame[:, W // 2:] = LEFT, RIGHT
    return frame


def view_colours(img):
    """Colour at the left and right of the enrollment screen's camera view (row 200)."""
    row = img[200, 18:346]
    cols = [i for i in range(len(row)) if tuple(row[i]) in (LEFT, RIGHT)]
    return tuple(row[cols[5]]), tuple(row[cols[-5]])


@pytest.mark.parametrize("mirror, expected", [(True, (RIGHT, LEFT)), (False, (LEFT, RIGHT))])
def test_enroll_screen_shows_the_camera_as_a_mirror_view(mirror, expected):
    s = EnrollState(step="Step 1 of 4 - Position", instruction="Move to your left",
                    guide=(0.35, 0.32, 0.65, 0.68), image=two_halves(), mirror=mirror)
    img = render_enroll(s, 800, 480, reserve_bottom=66)
    assert img.shape == (480, 800, 3)
    assert view_colours(img) == expected
    assert (img[-66:] == np.array(BG, np.uint8)).all(), "launcher strip area left empty"
    assert not (img[:, 360:] == np.array(LEFT, np.uint8)).all(axis=2).any(), \
        "camera pixels stay inside the view, clear of the guidance column"


def test_enroll_screen_countdown_replaces_the_view():
    """Eyes-shut countdown: the number, not the face (the driver cannot see it anyway)."""
    s = EnrollState(instruction="CLOSE YOUR EYES", countdown=2.0, image=two_halves())
    img = render_enroll(s, 800, 480, reserve_bottom=66)
    assert not (img == np.array(LEFT, np.uint8)).all(axis=2).any()


@pytest.mark.parametrize("instruction, detail, note", [
    ("Move up (sit higher or tilt the camera down)",
     "Sit as you will drive and look straight at the camera - bring the dot on your face "
     "into the dashed box.", ""),
    ("Look at the road", "Stay alert, look at the road as when driving, keep your mouth "
     "relaxed and do not talk.", ""),
    ("CLOSE YOUR EYES on the beep",
     'Keep them shut until the long beep (about 3 s). Operator: say "open" at the long beep.',
     "Attempt 2 did not pass: closed / open 0.81 is above 0.60 - the eyes did not close, or "
     "the landmarks do not follow the eyelids. Close the eyes fully this time."),
])
def test_enroll_text_is_never_dropped(instruction, detail, note):
    """Long guidance shrinks to fit the 800x480 screen above the progress bar - no line lost."""
    s = EnrollState(instruction=instruction, detail=detail, note=note, progress=0.5)
    # render_enroll at 800x480 with the launcher's 66 px strip: right column
    # from x 366 (424 px wide), first baseline 90, last above the progress bar.
    x0, max_w, y0, max_y = 366, 424, 90, 310
    lines = enroll_text_layout(s, x0, y0, max_w, max_y)
    for block, text in enumerate((instruction, detail, note)):
        drawn = " ".join(line for line, *_rest, b in lines if b == block)
        assert drawn.split() == text.split(), f"block {block} lost words"
    assert lines[-1][1] <= max_y
    import cv2
    for line, _y, scale, thick, font, _b in lines:
        assert cv2.getTextSize(line, font, scale, thick)[0][0] <= max_w


def test_positioning_shows_the_live_camera(m, monkeypatch):
    """Enrollment mode: the panel's canvases carry the camera image (and stay 800x480)."""
    shown = []
    monkeypatch.setattr(m, "show_frame", lambda img: (shown.append(img), -1)[1])
    monkeypatch.setattr(m, "ENROLL_POSITION_HOLD_S", 0.4)
    m._camera = MarkedCam()
    m._display_mode, m._display_available, m._reserve_bottom = "enroll", True, 66

    class InPosition:
        def extract(self, frame):
            return None, FakeRect(200, 120, 440, 360)

    assert m.run_positioning(InPosition()) is not None
    assert shown and all(img.shape == (480, 800, 3) for img in shown)
    assert all(shows_camera(img) for img in shown), "every enrollment redraw has the live view"
    assert all((img[-66:] == np.array(BG, np.uint8)).all() for img in shown)


def test_live_view_is_not_copied_per_frame(m):
    """The frame is held by reference (no per-frame cost) unless the stream will draw on it."""
    m._display_mode, m._display_available = "enroll", True
    frame = m.capture_frame()
    m.enroll_view(frame, "Step 1 of 4 - Position", "Face the camera")
    assert m._enroll.image is frame
    assert m._enroll.mirror is (not m.ENROLL_CAMERA_HFLIP)


def test_positioning_timeout_refuses(m, monkeypatch):
    class NoFace:
        def extract(self, frame):
            return None, None

    monkeypatch.setattr(m, "ENROLL_POSITION_TIMEOUT_S", 0.5)
    code = m.run_enrollment(RosterAPI([{"id": 3, "full_name": "Ana"}]), NoFace(), driver_id=3)
    assert code == 3 and m._enroll_report["outcome"] == "refused"


@pytest.mark.slow
def test_full_enrollment_on_the_enrollment_screen(m, rig, monkeypatch):
    """Positioning -> capture -> closed eyes -> calibration -> saved; only canvases shown."""
    fr = types.ModuleType("face_recognition")
    fr.face_locations = lambda rgb, model="hog": [(120, 440, 360, 200)]
    fr.face_encodings = lambda rgb, locs: [np.zeros(128)]
    monkeypatch.setitem(sys.modules, "face_recognition", fr)

    eyes = {"closed": False}

    class Capture(ClosedEyeCapture):
        def start(self, now):
            eyes["closed"] = True
            super().start(now)

        def done(self, now):
            finished = super().done(now)
            if finished:
                eyes["closed"] = False
            return finished

    class Calib(CalibrationManager):
        def __init__(self):
            super().__init__(duration=4)

        def write_files(self, *a, **k):          # never into logs/calibrations
            pass

    t0 = time.monotonic()

    class Extractor:
        rng = np.random.default_rng(0)

        def extract(self, frame):
            time.sleep(0.01)
            lm = synthetic_landmarks(self.rng)
            if eyes["closed"]:
                lm = close_eyes(lm)
            if time.monotonic() - t0 < 1.0:      # starts off to one side
                return lm, FakeRect(20, 120, 260, 360)
            return lm, FakeRect(200, 120, 440, 360)

    class API(RosterAPI):
        def save_driver_enrollment(self, *a, **k):
            return True

        def get_driver_calibration(self, did):
            return Calibration({"ear_baseline": 0.3, "ear_closed_baseline": 0.1}, did,
                               CALIBRATION_SOURCE_DEVICE, 1, "pi-01")

    shown, steps, live = [], set(), set()
    real_render = m.render_enroll

    def render(state, w, h, r):
        steps.add(state.step.split(" (")[0])
        img = real_render(state, w, h, r)
        if shows_camera(img):
            live.add(state.step.split(" (")[0])
        return img

    monkeypatch.setattr(m, "render_enroll", render)
    monkeypatch.setattr(m, "show_frame", lambda img: (shown.append(img.shape), -1)[1])
    m._camera = MarkedCam()
    monkeypatch.setattr(m, "ClosedEyeCapture", Capture)
    monkeypatch.setattr(m, "CalibrationManager", Calib)
    m._display_mode, m._display_available = "enroll", True
    m._rate = m.LoopRate()
    rig(FakeAPI(), ScriptedIgnition([(9e9, Phase.PREDRIVE)]))

    code = m.run_enrollment(API([{"id": 9, "full_name": "Maria Santos"}]), Extractor(),
                            driver_id=9)
    assert code == 0 and m._enroll_report["outcome"] == "saved"
    assert m._enroll_report["contrast"] < 0.6
    assert steps == {"Step 1 of 4 - Position", "Step 2 of 4 - Face capture",
                     "Step 3 of 4 - Closed eyes", "Step 4 of 4 - Calibration"}
    assert shown and set(shown) == {(480, 800, 3)}, "always the 800x480 enrollment screen"
    assert live == steps, "the live view is on the panel in every enrollment step"
