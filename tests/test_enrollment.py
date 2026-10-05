"""
Enrollment: positioning guidance, early refusals, the result file the
launcher reads, and a full run on the enrollment screen (no camera image).
"""

import json
import sys
import time
import types

import numpy as np
import pytest

from fakes import FakeAPI, FakeRect, ScriptedIgnition, close_eyes, synthetic_landmarks
from modules.api import CALIBRATION_SOURCE_DEVICE, Calibration
from modules.calibration import CalibrationManager, ClosedEyeCapture
from modules.phase import Phase

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

    shown, steps = [], set()
    real_render = m.render_enroll

    def render(state, w, h, r):
        steps.add(state.step.split(" (")[0])
        return real_render(state, w, h, r)

    monkeypatch.setattr(m, "render_enroll", render)
    monkeypatch.setattr(m, "show_frame", lambda img: (shown.append(img.shape), -1)[1])
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
    assert shown and set(shown) == {(480, 800, 3)}, "no camera frame on the panel"
