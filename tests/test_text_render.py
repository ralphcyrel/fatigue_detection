"""
Names that are not ASCII (2026-10-19): Hershey drew "Acuña" as "Acu??a".
Such text is drawn with a TrueType font, at Hershey's size; with no font it
is folded to ASCII - never handed to cv2.putText as is.
"""

import cv2
import numpy as np
import pytest

from modules import text_render
from modules.session_display import EnrollState, ScreenState, render, render_enroll
from modules.text_render import ascii_fold, put_text, text_size

FONT = cv2.FONT_HERSHEY_DUPLEX


@pytest.fixture
def cv2_texts(monkeypatch):
    """Every string that reaches cv2.putText."""
    seen = []
    real = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a: (seen.append(text),
                                                               real(img, text, *a))[1])
    return seen


@pytest.fixture
def no_font(monkeypatch):
    monkeypatch.setattr(text_render, "_truetype", lambda font, scale, thick: None)


needs_font = pytest.mark.skipif(text_render._font_file(False) is None,
                                reason="no TrueType font on this machine")


def test_ascii_fold():
    assert ascii_fold("Acuña") == "Acuna"
    assert ascii_fold("José Peña-Muñoz") == "Jose Pena-Munoz"
    assert ascii_fold("Łukasz Straße") == "Lukasz Strasse"
    assert ascii_fold("王") == "?"


def test_ascii_text_is_drawn_by_cv2_exactly_as_before(cv2_texts):
    a, b = np.zeros((60, 300, 3), np.uint8), np.zeros((60, 300, 3), np.uint8)
    put_text(a, "Driver: Acuna", (5, 40), FONT, 0.75, (240, 240, 240), 1)
    cv2.putText(b, "Driver: Acuna", (5, 40), FONT, 0.75, (240, 240, 240), 1, cv2.LINE_AA)
    assert (a == b).all()
    assert text_size("Driver: Acuna", FONT, 0.75, 1) == \
        cv2.getTextSize("Driver: Acuna", FONT, 0.75, 1)[0]


@needs_font
def test_non_ascii_text_drawn_whole_at_hershey_size(cv2_texts):
    img = np.zeros((60, 300, 3), np.uint8)
    put_text(img, "Acuña", (5, 40), FONT, 0.75, (0, 0, 240), 1)
    assert cv2_texts == [], "cv2 would draw the ñ as ??"
    ys, xs = np.nonzero(img.any(axis=2))
    assert ys.size and (img[ys, xs][:, :2] == 0).all(), "drawn in the BGR colour given"
    # The capital A is as tall as Hershey's, so the layout does not move.
    cap = cv2.getTextSize("H", FONT, 0.75, 1)[0][1]
    a_rows = np.nonzero(img[:, :12].any(axis=(1, 2)))[0]
    assert abs((a_rows.max() - a_rows.min() + 1) - cap) <= 2
    w, h = text_size("Acuña", FONT, 0.75, 1)
    assert abs(w - (xs.max() - 5 + 1)) <= 3 and h == cap


def test_without_a_font_the_name_is_folded_not_question_marks(cv2_texts, no_font):
    img = np.zeros((60, 300, 3), np.uint8)
    put_text(img, "Acuña", (5, 40), FONT, 0.75, (240, 240, 240), 1)
    assert cv2_texts == ["Acuna"]
    assert text_size("Acuña", FONT, 0.75, 1) == cv2.getTextSize("Acuna", FONT, 0.75, 1)[0]


def test_text_at_the_canvas_edge_is_clipped_not_an_error():
    img = np.zeros((20, 40, 3), np.uint8)
    put_text(img, "Acuña Acuña Acuña", (-15, 10), FONT, 1.0, (255, 255, 255), 2)
    put_text(img, "Acuña", (500, 500), FONT, 1.0, (255, 255, 255), 2)


@pytest.mark.parametrize("use_font", [True, False])
def test_screens_never_hand_cv2_a_non_ascii_name(cv2_texts, monkeypatch, use_font):
    if not use_font:
        monkeypatch.setattr(text_render, "_truetype", lambda font, scale, thick: None)
    render(ScreenState(driver_name="Juan Acuña", level="ALERT", frs=0.1,
                       status=[("Welcome, Juan Acuña", (240, 240, 240))]), 800, 480, 66)
    render_enroll(EnrollState(driver_name="Juan Acuña", step="Face capture",
                              instruction="Look at the camera, Juan Acuña"), 800, 480, 66)
    assert all(t.isascii() for t in cv2_texts)
    assert any("Acuna" in t for t in cv2_texts) != use_font


def test_video_overlay_name(m, cv2_texts):
    """--show-video draws the driver's name on the camera frame the same way."""
    m._display_mode, m._display_available = "video", True
    frame = m.capture_frame()
    m.display_text(frame, "Welcome, Juan Acuña")
    m.draw_banner(frame, "Driver: Acuña")
    assert all(t.isascii() for t in cv2_texts)
    assert frame.any()
