"""
Data screen and enrollment screen for the in-vehicle display (800x480 HDMI
touchscreen).

Group decision (2026-10-05): during pre-drive and monitoring the in-vehicle
display shows NO camera image. :func:`render` draws what replaces it - the
fatigue level, the live numbers next to the driver's baselines, the phase /
status text that used to be drawn on the video, and a plain-text face
indicator so a mis-aimed camera is still obvious without a picture. It
takes no image at all.

Enrollment is the one exception (2026-10-07): :func:`render_enroll` shows
the live camera as a mirror view, so the driver being enrolled can position
themselves, next to the step's guidance.

Display only. Nothing here feeds back into a metric, a threshold, the FRS,
the relay or the backend: :class:`ScreenState` is written by the main loop's
existing draw helpers and read by :func:`render`, nothing else.

The canvas is drawn with OpenCV (Hershey fonts) so it goes through the same
``cv2.imshow`` window as the old preview and needs no extra GUI toolkit.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Colours (BGR). Level colours match main.LEVEL_BGR.
LEVEL_BGR = {
    "ALERT": (0, 170, 0),
    "WARNING": (0, 200, 255),
    "DANGER": (0, 0, 230),
    "FAULT": (200, 0, 200),
}
# Text drawn on a level tile: dark on the light tiles, white on the dark ones.
LEVEL_TEXT_BGR = {"ALERT": (255, 255, 255), "WARNING": (0, 0, 0),
                  "DANGER": (255, 255, 255), "FAULT": (255, 255, 255)}
BG = (24, 24, 24)
PANEL = (44, 44, 44)
WHITE = (240, 240, 240)
GREY = (150, 150, 150)
DIM = (95, 95, 95)
GREEN = (0, 190, 0)
AMBER = (0, 190, 255)
RED = (40, 40, 230)

# Face indicator states (FaceStatus.kind).
FACE_OK = "ok"                    # face found, driver recognised (or not needed)
FACE_NONE = "none"                # no face in the frame
FACE_UNRECOGNISED = "unrecognised"  # face found, not matched to a driver
FACE_SEARCHING = "searching"      # face found, identification still running

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class ScreenState:
    """
    What the data screen shows. Filled by main.py's draw helpers each frame.

    Fields under "per frame" are cleared by :meth:`begin_frame` (they
    describe one loop iteration); the "session" fields persist until
    :meth:`reset_session` so the driver's name and baselines stay up while,
    for example, the operator-override screen runs without metrics.
    """

    # ---- session -----------------------------------------------------------
    driver_name: Optional[str] = None
    ear_base: Optional[float] = None
    mar_base: Optional[float] = None
    perclos_base: Optional[float] = None
    blink_base_per_min: Optional[float] = None
    # Column header: "baseline" for the driver's own calibration; "defaults"
    # when there is none (population values, self-seeded EAR).
    baseline_label: str = "baseline"

    # ---- per frame ---------------------------------------------------------
    phase: str = ""                                  # the old bottom banner
    phase_color: Tuple[int, int, int] = WHITE        # BGR
    status: List[Tuple[str, Tuple[int, int, int]]] = field(default_factory=list)
    level: Optional[str] = None
    frs: Optional[float] = None
    frs_note: str = ""                               # why there is no FRS
    face: Optional[str] = None                       # FACE_* or None = not checked
    face_detail: str = ""
    countdown: Optional[float] = None                # seconds, shown large
    ear: Optional[float] = None
    mar: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    perclos: Optional[float] = None
    blink_per_min: Optional[float] = None
    alert_banner: str = ""                           # e.g. MICROSLEEP 1.2s
    fps: Optional[float] = None

    _PER_FRAME = ("phase", "phase_color", "status", "level", "frs", "frs_note", "face",
                  "face_detail", "countdown", "ear", "mar", "yaw", "pitch", "perclos",
                  "blink_per_min", "alert_banner")

    def begin_frame(self) -> None:
        """Clear the per-frame fields (called once per captured frame)."""
        fresh = ScreenState()
        for name in self._PER_FRAME:
            setattr(self, name, getattr(fresh, name))

    def reset_session(self) -> None:
        """Forget the driver and baselines (new pre-drive / monitoring run)."""
        self.driver_name = None
        self.ear_base = self.mar_base = self.perclos_base = self.blink_base_per_min = None
        self.baseline_label = "baseline"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _text(img: np.ndarray, text: str, x: int, y: int, scale: float,
          color: Tuple[int, int, int], thick: int = 1, align: str = "left",
          font: int = FONT, max_w: Optional[int] = None) -> int:
    """Draw ``text`` with its baseline at ``y``; returns the drawn width.

    ``max_w`` shrinks the scale until the text fits (long status lines).
    """
    (w, _h), _ = cv2.getTextSize(text, font, scale, thick)
    if max_w is not None and w > max_w and w > 0:
        scale *= max_w / w
        (w, _h), _ = cv2.getTextSize(text, font, scale, thick)
    if align == "center":
        x -= w // 2
    elif align == "right":
        x -= w
    cv2.putText(img, text, (int(x), int(y)), font, scale, color, thick, cv2.LINE_AA)
    return w


def _fmt(value: Optional[float], spec: str) -> str:
    return "--" if value is None else format(value, spec)


def _face_line(s: ScreenState) -> Tuple[str, Tuple[int, int, int]]:
    if s.face is None:
        return "CAMERA: not checked in this step", GREY
    if s.face == FACE_NONE:
        return "NO FACE" + (f"  {s.face_detail}" if s.face_detail else "") + \
            "  - check camera aim", RED
    if s.face == FACE_UNRECOGNISED:
        return "FACE DETECTED - DRIVER NOT RECOGNISED", AMBER
    if s.face == FACE_SEARCHING:
        return "FACE DETECTED - identifying" + (f" {s.face_detail}" if s.face_detail else ""), AMBER
    return "FACE DETECTED", GREEN


_BACKGROUNDS: Dict[Tuple[int, int], np.ndarray] = {}


def _background(width: int, height: int) -> np.ndarray:
    """Blank canvas, built once per size (np.full with a colour is ~1 ms)."""
    bg = _BACKGROUNDS.get((width, height))
    if bg is None:
        bg = _BACKGROUNDS[(width, height)] = np.empty((height, width, 3), np.uint8)
        bg[:] = BG
    return bg


def render(s: ScreenState, width: int, height: int, reserve_bottom: int = 0) -> np.ndarray:
    """
    Draw the data screen for ``s`` on a ``width`` x ``height`` canvas.

    ``reserve_bottom`` px at the bottom are left empty: the touchscreen
    launcher's STOP strip sits there, above this window.

    Layout (800x480, 66 px reserved)::

        +- phase line ------------------------------- driver name ---+
        | +- level tile --------+  +- numbers -------- baseline --+ |
        | |      WARNING        |  | EAR      0.241      0.301    | |
        | |     FRS 0.452       |  | MAR ...                       | |
        | +---------------------+  +------------------------------+ |
        | FACE DETECTED                                    12.3 s   |
        | STARTER LOCKED: FATIGUE DETECTED                          |
        | awaiting operator override (request 41)                  |
        +-----------------------------------------------------------+
    """
    img = _background(width, height).copy()
    h = height - reserve_bottom
    pad = 10

    # ---- top bar: phase (left), driver (right) ------------------------------
    bar_h = 46
    cv2.rectangle(img, (0, 0), (width, bar_h), PANEL, -1)
    cv2.rectangle(img, (0, 0), (8, bar_h), s.phase_color, -1)
    driver = f"Driver: {s.driver_name}" if s.driver_name else "Driver: not identified"
    dw = _text(img, driver, width - pad, 31, 0.75, WHITE if s.driver_name else GREY,
               1, "right", max_w=width // 2 - pad)
    _text(img, s.phase or "-", 18, 31, 0.7, s.phase_color, 1,
          max_w=width - dw - 3 * pad - 18)

    # ---- bottom block: face indicator + countdown, status lines ------------
    status_h = 34 * min(len(s.status), 2) + 4
    face_y = h - status_h - 14                       # baseline of the face line
    status_top = face_y + 12

    face_text, face_color = _face_line(s)
    cv2.circle(img, (pad + 10, face_y - 9), 9, face_color, -1)
    cw = 0
    if s.countdown is not None:
        cw = _text(img, f"{max(s.countdown, 0.0):4.1f} s", width - pad, face_y + 4, 1.3,
                   WHITE, 2, "right", font=FONT_BOLD)
    _text(img, face_text, pad + 28, face_y, 0.75, face_color, 1,
          max_w=width - cw - 3 * pad - 28)

    y = status_top
    for i, (line, color) in enumerate(s.status[:2]):
        scale = 0.95 if i == 0 else 0.7
        y += 34 if i == 0 else 30
        _text(img, line, pad, y - 6, scale, color, 2 if i == 0 else 1, max_w=width - 2 * pad - 70)

    # ---- middle: level tile (left) and numbers (right) ---------------------
    top, bottom = bar_h + pad, face_y - 30
    tile_w = int(width * 0.42)
    if s.level is not None:
        tile_color = LEVEL_BGR.get(s.level, PANEL)
        text_color = LEVEL_TEXT_BGR.get(s.level, WHITE)
        cv2.rectangle(img, (pad, top), (pad + tile_w, bottom), tile_color, -1)
        cx = pad + tile_w // 2
        mid = (top + bottom) // 2
        _text(img, s.level, cx, mid + 10, 2.2, text_color, 5, "center", font=FONT_BOLD,
              max_w=tile_w - 24)
        frs = f"FRS {s.frs:.2f}" if s.frs is not None else (s.frs_note or "FRS --")
        _text(img, frs, cx, mid + 58, 0.95 if s.frs is not None else 0.6, text_color,
              2 if s.frs is not None else 1, "center", max_w=tile_w - 24)
        if s.alert_banner:
            _text(img, s.alert_banner, cx, top + 34, 0.8, text_color, 2, "center",
                  font=FONT_BOLD, max_w=tile_w - 24)
    else:
        # No metrics in this step (identifying, override wait, release):
        # the tile carries the first status line instead of a level.
        cv2.rectangle(img, (pad, top), (pad + tile_w, bottom), PANEL, -1)
        head, color = s.status[0] if s.status else ("-", GREY)
        _text(img, head.split(":")[0], pad + tile_w // 2, (top + bottom) // 2 + 14, 1.4,
              color, 3, "center", font=FONT_BOLD, max_w=tile_w - 24)

    x0 = pad + tile_w + 2 * pad
    cv2.rectangle(img, (x0 - pad // 2, top), (width - pad, bottom), PANEL, -1)
    col_val, col_base = x0 + 255, width - 2 * pad
    rows = [
        ("EAR", _fmt(s.ear, ".3f"), _fmt(s.ear_base, ".3f")),
        ("MAR", _fmt(s.mar, ".3f"), _fmt(s.mar_base, ".3f")),
        ("PERCLOS %", _fmt(s.perclos, ".1f"), _fmt(s.perclos_base, ".1f")),
        ("Blinks/min", _fmt(s.blink_per_min, ".0f"), _fmt(s.blink_base_per_min, ".0f")),
        ("Yaw", _fmt(s.yaw, "+.0f"), ""),
        ("Pitch", _fmt(s.pitch, "+.0f"), ""),
    ]
    if s.fps is not None:
        _text(img, f"{s.fps:.1f} fps", x0 + 6, top + 22, 0.45, DIM, 1)
    _text(img, "now", col_val, top + 22, 0.5, GREY, 1, "right")
    _text(img, s.baseline_label, col_base, top + 22, 0.5, GREY, 1, "right")
    row_h = max(24, (bottom - top - 30) // len(rows))
    scale = min(0.8, row_h / 40)
    for i, (label, now, base) in enumerate(rows):
        ry = top + 30 + row_h * (i + 1) - row_h // 4
        _text(img, label, x0 + 6, ry, scale * 0.85, GREY, 1)
        _text(img, now, col_val, ry, scale * 1.1, WHITE, 2, "right")
        if base:
            _text(img, base, col_base, ry, scale, GREY, 1, "right")
    return img


# ---------------------------------------------------------------------------
# Enrollment screen
# ---------------------------------------------------------------------------

@dataclass
class EnrollState:
    """
    What the enrollment screen shows. Rebuilt by main.py on every frame.

    ``box`` and ``guide`` are normalised (0..1) rectangles ``(x0, y0, x1,
    y1)`` in display orientation (already mirrored if guidance is mirrored):
    the face box the detector found and the zone it should sit in.

    ``image`` is the current camera frame (BGR, camera orientation; held by
    reference, not copied - :func:`render_enroll` only reads it, and only
    when the screen is redrawn). ``mirror`` flips it into display
    orientation so it matches ``box`` and ``guide``.
    """

    driver_name: Optional[str] = None
    step: str = ""
    instruction: str = ""
    instruction_color: Tuple[int, int, int] = WHITE
    detail: str = ""
    note: str = ""                                   # e.g. why the last attempt failed
    face_text: str = ""
    face_color: Tuple[int, int, int] = GREY
    box: Optional[Tuple[float, float, float, float]] = None
    box_ok: bool = False
    guide: Optional[Tuple[float, float, float, float]] = None
    progress: Optional[float] = None                 # 0..1
    progress_label: str = ""
    countdown: Optional[float] = None                # seconds, drawn huge
    countdown_label: str = ""
    fps: Optional[float] = None
    image: Optional[np.ndarray] = None
    mirror: bool = True


def _wrap(text: str, scale: float, thick: int, max_w: int, font: int = FONT) -> List[str]:
    """Greedy word wrap for cv2.putText."""
    lines: List[str] = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            trial = f"{line} {word}".strip()
            if line and cv2.getTextSize(trial, font, scale, thick)[0][0] > max_w:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


def _camera_view(img: np.ndarray, s: EnrollState, x0: int, y0: int, x1: int, y1: int) -> None:
    """
    The camera frame in a 4:3 box - the live mirror view when ``s.image``
    is set, else an empty outline - with the target zone and the face box.
    """
    w, h = x1 - x0, y1 - y0
    fw = min(w, int(h * 4 / 3))
    fh = int(fw * 3 / 4)
    fx, fy = x0 + (w - fw) // 2, y0 + (h - fh) // 2
    if s.image is not None and s.image.size:
        # Shrink first, then flip the small copy: ~1/4 of the pixels.
        view = cv2.resize(s.image, (fw, fh), interpolation=cv2.INTER_LINEAR)
        if view.ndim == 2:
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
        img[fy:fy + fh, fx:fx + fw] = cv2.flip(view, 1) if s.mirror else view
    else:
        cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (30, 30, 30), -1)
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), GREY, 1)

    def px(r: Tuple[float, float, float, float]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return ((fx + int(r[0] * fw), fy + int(r[1] * fh)),
                (fx + int(r[2] * fw), fy + int(r[3] * fh)))

    if s.guide is not None:
        (gx0, gy0), (gx1, gy1) = px(s.guide)
        for x in range(gx0, gx1, 12):                    # dashed target zone
            cv2.line(img, (x, gy0), (min(x + 6, gx1), gy0), DIM, 1)
            cv2.line(img, (x, gy1), (min(x + 6, gx1), gy1), DIM, 1)
        for y in range(gy0, gy1, 12):
            cv2.line(img, (gx0, y), (gx0, min(y + 6, gy1)), DIM, 1)
            cv2.line(img, (gx1, y), (gx1, min(y + 6, gy1)), DIM, 1)
    if s.box is not None:
        (bx0, by0), (bx1, by1) = px(s.box)
        color = GREEN if s.box_ok else AMBER
        cv2.rectangle(img, (bx0, by0), (bx1, by1), color, 3)
        # The face centre: this dot, not the whole box, belongs in the zone.
        cv2.circle(img, ((bx0 + bx1) // 2, (by0 + by1) // 2), 7, color, -1)
    if s.image is not None:
        caption = ("live camera, mirror view - face dot inside the dashed box"
                   if s.guide is not None else "live camera, mirror view")
    else:
        caption = ("outline only, no image - face dot inside the dashed box"
                   if s.guide is not None else "camera view - outline only, no image")
    _text(img, caption, fx + fw // 2, fy + fh + 18, 0.45, DIM, 1, "center", max_w=fw)


# Instruction / detail / note in the enrollment screen's right column:
# (scale, thickness, font, baseline-to-baseline px, extra px before the block).
_ENROLL_BLOCKS = ((1.1, 2, FONT_BOLD, 42, 0), (0.65, 1, FONT, 27, 6), (0.55, 1, FONT, 23, 4))


def enroll_text_layout(s: EnrollState, x0: int, y0: int, max_w: int, max_y: int
                       ) -> List[Tuple[str, int, float, int, int, int]]:
    """
    Every line of the instruction, detail and note as
    ``(line, baseline_y, scale, thickness, font, block)``.

    All of it, always: if it does not fit between ``y0`` (first baseline)
    and ``max_y`` (last baseline) the three blocks shrink together, down
    to half size - lines are never dropped.
    """
    texts = (s.instruction, s.detail, s.note)
    for k in np.arange(1.0, 0.45, -0.05):
        out: List[Tuple[str, int, float, int, int, int]] = []
        y = y0
        for block, (text, (scale, thick, font, line_h, gap)) in enumerate(
                zip(texts, _ENROLL_BLOCKS)):
            if not text:
                continue
            if out:
                y += int(gap * k)
            for line in _wrap(text, scale * k, thick, max_w, font):
                out.append((line, y, scale * k, thick, font, block))
                y += int(line_h * k)
        if not out or out[-1][1] <= max_y:
            return out
    return out


def render_enroll(s: EnrollState, width: int, height: int, reserve_bottom: int = 0) -> np.ndarray:
    """
    Draw the enrollment screen: step, the live mirror view with the target
    zone and face box (or a huge countdown), the instruction, a progress
    bar and the face indicator.
    """
    img = _background(width, height).copy()
    h = height - reserve_bottom
    pad = 10

    bar_h = 46
    cv2.rectangle(img, (0, 0), (width, bar_h), PANEL, -1)
    cv2.rectangle(img, (0, 0), (8, bar_h), AMBER, -1)
    driver = f"Enrolling: {s.driver_name}" if s.driver_name else "Enrolling"
    dw = _text(img, driver, width - pad, 31, 0.7, WHITE, 1, "right", max_w=width // 2 - pad)
    _text(img, s.step, 18, 31, 0.7, AMBER, 1, max_w=width - dw - 3 * pad - 18)

    # Bottom: face indicator.
    face_y = h - 14
    cv2.circle(img, (pad + 10, face_y - 9), 9, s.face_color, -1)
    _text(img, s.face_text, pad + 28, face_y, 0.7, s.face_color, 1, max_w=width - 2 * pad - 110)
    if s.fps is not None:
        _text(img, f"{s.fps:.1f} fps", width - pad, face_y, 0.45, DIM, 1, "right")

    top, bottom = bar_h + pad, face_y - 30
    tile_w = int(width * 0.42)
    cv2.rectangle(img, (pad, top), (pad + tile_w, bottom), PANEL, -1)
    if s.countdown is not None:
        cx, mid = pad + tile_w // 2, (top + bottom) // 2
        _text(img, f"{max(s.countdown, 0.0):.0f}" if s.countdown >= 1 else f"{s.countdown:.1f}",
              cx, mid + 45, 5.0, s.instruction_color, 10, "center", font=FONT_BOLD,
              max_w=tile_w - 30)
        if s.countdown_label:
            _text(img, s.countdown_label, cx, bottom - 16, 0.7, GREY, 1, "center",
                  max_w=tile_w - 20)
    elif s.image is not None or s.guide is not None or s.box is not None:
        _camera_view(img, s, pad + 8, top + 8, pad + tile_w - 8, bottom - 22)

    # Right: instruction, detail, note, progress.
    x0 = pad + tile_w + 2 * pad
    max_w = width - x0 - pad
    by = bottom - 26                                 # progress bar top
    text_bottom = (by - 34) if s.progress is not None else bottom - 6
    colors = (s.instruction_color, WHITE, AMBER)
    for line, y, scale, thick, font, block in enroll_text_layout(s, x0, top + 34, max_w,
                                                                 text_bottom):
        _text(img, line, x0, y, scale, colors[block], thick, font=font, max_w=max_w)
    if s.progress is not None:
        cv2.rectangle(img, (x0, by), (width - pad, by + 22), PANEL, -1)
        fill = int((width - pad - x0) * min(max(s.progress, 0.0), 1.0))
        if fill > 0:
            cv2.rectangle(img, (x0, by), (x0 + fill, by + 22), GREEN, -1)
        cv2.rectangle(img, (x0, by), (width - pad, by + 22), GREY, 1)
        if s.progress_label:
            _text(img, s.progress_label, x0, by - 8, 0.55, GREY, 1, max_w=max_w)
    return img
