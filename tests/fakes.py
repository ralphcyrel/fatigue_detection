"""
Fake hardware and backend for the tests.

Nothing here touches the camera, GPIO or the network: frames are blank (or
synthetic landmarks), the backend is a recorder, and the ignition follows a
script. Costs are modelled with ``time.sleep`` so the loops run at a
realistic pace.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.api import CALIBRATION_SOURCE_DEVICE, Calibration
from modules.assessment import PredriveAssessment
from modules.phase import Phase


def synthetic_landmarks(rng: np.random.Generator) -> np.ndarray:
    """A plausible 68-point face (EAR 0.30, mouth closed) with a little jitter."""
    pts = np.zeros((68, 2), float)
    for i in range(17):                                   # jaw arc
        a = np.pi * i / 16
        pts[i] = (320 - 120 * np.cos(a), 200 + 200 * np.sin(a))
    pts[17:27] = [(220 + 22 * k if k < 5 else 310 + 22 * (k - 4), 170) for k in range(10)]
    pts[27:31] = [(320, 200 + 23 * k) for k in range(4)]  # nose bridge -> tip (30)
    pts[31:36] = [(300 + 10 * k, 280) for k in range(5)]
    for c, base in ((250, 36), (390, 42)):                # eyes, EAR = 0.30
        pts[base:base + 6] = [(c - 30, 200), (c - 10, 191), (c + 10, 191),
                              (c + 30, 200), (c + 10, 209), (c - 10, 209)]
    for k in range(12):                                   # outer lip
        a = 2 * np.pi * k / 12
        pts[48 + k] = (320 - 30 * np.cos(a), 330 - 8 * np.sin(a))
    for k in range(8):                                    # inner lip
        a = 2 * np.pi * k / 8
        pts[60 + k] = (320 - 20 * np.cos(a), 330 - 3 * np.sin(a))
    return (pts + rng.normal(0, 0.3, pts.shape)).round().astype(np.int32)


def close_eyes(landmarks: np.ndarray) -> np.ndarray:
    """Flatten both eyes of ``synthetic_landmarks`` (EAR ~0.07)."""
    lm = landmarks.copy()
    for base in (36, 42):
        lm[base + 1, 1] = lm[base + 2, 1] = 198
        lm[base + 4, 1] = lm[base + 5, 1] = 202
    return lm


class FakeRect:
    """Stands in for ``dlib.rectangle``."""

    def __init__(self, left: int, top: int, right: int, bottom: int) -> None:
        self._box = (left, top, right, bottom)

    def left(self) -> int:
        return self._box[0]

    def top(self) -> int:
        return self._box[1]

    def right(self) -> int:
        return self._box[2]

    def bottom(self) -> int:
        return self._box[3]


class FakeCam:
    """30 fps sensor delivering blank frames."""

    def read(self) -> np.ndarray:
        time.sleep(1 / 30)
        return np.zeros((480, 640, 3), np.uint8)

    def release(self) -> None:
        pass


# A BGR colour the screens never draw: finding it on a canvas means camera
# pixels reached the panel.
MARKER = (1, 250, 3)


class MarkedCam(FakeCam):
    """FakeCam whose frames are filled with ``MARKER``."""

    def read(self) -> np.ndarray:
        frame = super().read()
        frame[:] = MARKER
        return frame


def shows_camera(canvas: np.ndarray) -> bool:
    """Whether any pixel of ``canvas`` is ``MARKER``."""
    return bool((canvas == np.array(MARKER, np.uint8)).all(axis=2).any())


class FakeExtractor:
    """Landmark extraction with a fixed cost; ``face()`` decides each frame."""

    def __init__(self, cost: float = 0.015, face=lambda: True) -> None:
        self.cost, self.face, self.rng = cost, face, np.random.default_rng(0)

    def extract(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[FakeRect]]:
        time.sleep(self.cost)
        if not self.face():
            return None, None
        return synthetic_landmarks(self.rng), FakeRect(200, 120, 440, 360)


class FakeRecognizer:
    """Recognition with a fixed cost; matches ``driver_id`` (None = never)."""

    def __init__(self, cost: float = 0.6, driver_id: Optional[int] = 6) -> None:
        self.cost, self.driver_id, self.calls = cost, driver_id, 0
        self.last_face_count: Optional[int] = 1

    def identify(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        self.calls += 1
        time.sleep(self.cost)
        if self.driver_id is None:
            return None
        return {"driver_id": self.driver_id, "name": "Test Driver", "confidence": 0.6}

    def load_encodings(self) -> int:
        return 1


class FakeAPI:
    """Records every call by name; returns canned values."""

    def __init__(self, override_status: str = "approved") -> None:
        self.calls: List[str] = []
        self.override_status = override_status

    def __getattr__(self, name: str):
        def rec(*a, **k):
            self.calls.append(name)
            return {"request_override": 42, "post_assessment": 7,
                    "check_override_request": self.override_status,
                    "get_driver_calibration": None, "ping": False}.get(name, True)
        return rec


class ScriptedIgnition:
    """``[(seconds_from_first_read, phase), ...]``; the last phase holds."""

    mock = True

    def __init__(self, schedule: List[Tuple[float, Phase]]) -> None:
        self.schedule, self.t0 = schedule, None

    def phase(self, now: Optional[float] = None) -> Phase:
        t = time.monotonic()
        self.t0 = self.t0 or t
        for until, ph in self.schedule:
            if t - self.t0 < until:
                return ph
        return self.schedule[-1][1]

    def read(self, now: Optional[float] = None) -> bool:
        return self.phase() is Phase.MONITORING

    def set_mock_state(self, on: bool) -> None:
        pass

    def cleanup(self) -> None:
        pass


THRESHOLDS = {"ear_baseline": 0.30, "ear_threshold": 0.225, "perclos_baseline": 5.0,
              "blink_duration_baseline": 150.0, "blink_frequency_baseline": 15.0,
              "mar_baseline": 0.45, "yawn_threshold": 0.90}
CAL = Calibration(THRESHOLDS, 6, CALIBRATION_SOURCE_DEVICE, 1, "pi-01")


class ShortAssessment(PredriveAssessment):
    """A 4 s assessment that writes nothing to disk."""

    def __init__(self) -> None:
        super().__init__(duration_s=4.0, worst_window_s=2.0)

    def write_files(self, *a, **k) -> None:
        pass


def keys_after(n: int, key: int):
    """``show_frame`` replacement: -1 for ``n - 1`` frames, then ``key`` once."""
    state = {"n": 0}

    def show(frame: np.ndarray) -> int:
        state["n"] += 1
        return key if state["n"] == n else -1
    return show
