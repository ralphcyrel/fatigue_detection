"""
Text on the OpenCV canvases, including names that are not ASCII.

``cv2.putText``'s Hershey fonts have ASCII only: every other byte of the
UTF-8 string is drawn as ``?``, so "Acuña" (two bytes for the ñ) showed as
"Acu??a" (2026-10-19). :func:`put_text` / :func:`text_size` are drop-in
replacements for ``cv2.putText`` / ``cv2.getTextSize``:

- ASCII text goes straight to cv2 - the same look and speed as before;
- any other text is drawn with a TrueType font through Pillow (a
  face_recognition dependency), sized to Hershey's cap height at the same
  scale so no layout moves;
- with no Pillow or no usable font the text is folded to ASCII ("Acuna"),
  never drawn as ``??``.
"""

import functools
import logging
import unicodedata
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:                       # pragma: no cover - Pillow comes with face_recognition
    Image = ImageDraw = ImageFont = None

logger = logging.getLogger(__name__)

# Regular / bold (Hershey thickness >= 2). DejaVu is on Raspberry Pi OS and
# is the launcher's Tk font; Arial is the Windows development box. Bare
# names are looked up in the system font directories by Pillow.
FONT_FILES = {
    False: ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf", "arial.ttf"),
    True: ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf",
           "C:/Windows/Fonts/arialbd.ttf", "arialbd.ttf"),
}
_REF_SIZE = 100   # px: the size a font's cap height is measured at

# Letters NFKD does not decompose into an ASCII letter + accent.
_FOLD = str.maketrans({"ß": "ss", "Æ": "AE", "æ": "ae", "Œ": "OE", "œ": "oe", "Ø": "O",
                       "ø": "o", "Ł": "L", "ł": "l", "Đ": "D", "đ": "d", "Þ": "Th",
                       "þ": "th", "ı": "i"})


def ascii_fold(text: str) -> str:
    """``"Acuña"`` -> ``"Acuna"``; what cannot be folded becomes one ``?`` per character."""
    decomposed = unicodedata.normalize("NFKD", text.translate(_FOLD))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.encode("ascii", "replace").decode("ascii")


@functools.lru_cache(maxsize=None)
def _font_file(bold: bool) -> Optional[Tuple[str, float]]:
    """(path, cap height / size) of the first font that loads, or None (logged once)."""
    if ImageFont is None:
        logger.warning("Pillow not installed: non-ASCII text is drawn folded to ASCII")
        return None
    for path in FONT_FILES[bold]:
        try:
            font = ImageFont.truetype(path, _REF_SIZE)
        except OSError:
            continue
        _l, top, _r, bottom = font.getbbox("H", anchor="ls")
        return path, (bottom - top) / _REF_SIZE
    logger.warning("No TrueType font found (%s): non-ASCII text is drawn folded to ASCII",
                   ", ".join(FONT_FILES[bold]))
    return None


@functools.lru_cache(maxsize=256)
def _truetype(font: int, scale: float, thick: int):
    """The Pillow font whose cap height matches Hershey ``font`` at ``scale``, or None."""
    found = _font_file(thick >= 2)
    if found is None:
        return None
    path, cap_ratio = found
    cap = cv2.getTextSize("H", font, scale, thick)[0][1]
    return ImageFont.truetype(path, max(1, int(round(cap / cap_ratio))))


def text_size(text: str, font: int, scale: float, thick: int = 1) -> Tuple[int, int]:
    """(width, height) of ``text`` as :func:`put_text` draws it - ``cv2.getTextSize(...)[0]``."""
    if text.isascii():
        return cv2.getTextSize(text, font, scale, thick)[0]
    tt = _truetype(font, scale, thick)
    if tt is None:
        return cv2.getTextSize(ascii_fold(text), font, scale, thick)[0]
    return (int(np.ceil(tt.getlength(text))), cv2.getTextSize("H", font, scale, thick)[0][1])


def put_text(img: np.ndarray, text: str, org: Tuple[int, int], font: int, scale: float,
             color: Tuple[int, int, int], thick: int = 1) -> None:
    """``cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)`` for any text."""
    x, y = int(org[0]), int(org[1])
    if text.isascii():
        cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)
        return
    tt = _truetype(font, scale, thick) if img.ndim == 3 and img.shape[2] == 3 else None
    if tt is None:
        cv2.putText(img, ascii_fold(text), (x, y), font, scale, color, thick, cv2.LINE_AA)
        return
    # Only the text's box goes through Pillow (BGR <-> RGB), clipped to the image.
    left, top, right, bottom = tt.getbbox(text, anchor="ls")
    x0, y0 = max(x + left, 0), max(y + top, 0)
    x1, y1 = min(x + right + 1, img.shape[1]), min(y + bottom + 1, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    canvas = Image.fromarray(np.ascontiguousarray(roi[..., ::-1]))
    ImageDraw.Draw(canvas).text((x - x0, y - y0), text, font=tt, anchor="ls",
                                fill=(int(color[2]), int(color[1]), int(color[0])))
    roi[:] = np.asarray(canvas)[..., ::-1]
