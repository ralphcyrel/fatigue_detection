"""
Module 1 — Facial landmark extraction.

Wraps dlib's HOG face detector and 68-point shape predictor into a single
`LandmarkExtractor` class tuned for the Raspberry Pi 4B:

* Face *detection* (the expensive step) runs on a downscaled copy of the frame.
* Landmark *prediction* (cheap, but accuracy-sensitive) runs on the
  full-resolution greyscale frame using the detected rectangle scaled back up.

Downstream modules (EAR, blink, PERCLOS) consume the returned (68, 2) landmark
array and slice it with `get_region()` using the index constants defined here.

Landmark numbering follows the iBUG 300-W 68-point convention used by dlib.
"""

from typing import Dict, Optional, Tuple

import cv2
import dlib
import numpy as np

# ---------------------------------------------------------------------------
# Landmark index ranges (start, stop) — half-open, so use as arr[start:stop].
# Note the naming is from the *subject's* perspective: LEFT_EYE (42–47) is the
# eye that appears on the right side of the image.
# ---------------------------------------------------------------------------
LEFT_EYE: Tuple[int, int] = (42, 48)
RIGHT_EYE: Tuple[int, int] = (36, 42)
MOUTH: Tuple[int, int] = (48, 68)
NOSE: Tuple[int, int] = (27, 36)
JAW: Tuple[int, int] = (0, 17)

# Maps the string names accepted by `get_region()` to their index ranges so
# callers don't need to import the constants individually.
REGIONS: Dict[str, Tuple[int, int]] = {
    "left_eye": LEFT_EYE,
    "right_eye": RIGHT_EYE,
    "mouth": MOUTH,
    "nose": NOSE,
    "jaw": JAW,
}

# Total number of points produced by the shape predictor.
NUM_LANDMARKS: int = 68


class LandmarkExtractor:
    """
    Detects the largest face in a frame and extracts its 68 facial landmarks.

    Typical usage::

        extractor = LandmarkExtractor(config.LANDMARK_MODEL, config.SCALE_FACTOR)
        landmarks, rect = extractor.extract(frame)
        if landmarks is not None:
            left_eye = extractor.get_region(landmarks, "left_eye")
    """

    def __init__(self, model_path: str, scale_factor: float = 0.5) -> None:
        """
        Load the dlib HOG face detector and the 68-point shape predictor.

        Args:
            model_path: Path to ``shape_predictor_68_face_landmarks.dat``.
            scale_factor: Factor by which frames are downscaled before face
                detection (0 < scale_factor <= 1). Smaller is faster but may
                miss small/distant faces. 0.5 is a good balance on the Pi 4B.

        Raises:
            ValueError: If ``scale_factor`` is outside (0, 1].
            RuntimeError: If dlib cannot load the predictor file (wrapped so
                the caller gets a readable message instead of a bare dlib error).
        """
        if not 0.0 < scale_factor <= 1.0:
            raise ValueError(f"scale_factor must be in (0, 1], got {scale_factor}")

        self.scale_factor: float = scale_factor

        # HOG + linear SVM detector. Slower than a CNN on GPU but the only
        # option that runs at usable frame rates on the Pi's CPU.
        self.detector = dlib.get_frontal_face_detector()

        # dlib raises a generic RuntimeError if the file is missing or corrupt;
        # re-raise with the path included so setup problems are obvious.
        try:
            self.predictor = dlib.shape_predictor(str(model_path))
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load landmark model from '{model_path}'. "
                "Download shape_predictor_68_face_landmarks.dat into data/ "
                "(see README.md)."
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[dlib.rectangle]]:
        """
        Detect the largest face in ``frame`` and return its landmarks.

        The pipeline is:
        1. Convert to greyscale (dlib works on single-channel images).
        2. Downscale the grey frame and run the HOG detector on it.
        3. Pick the largest detected rectangle (assumed to be the driver —
           passengers/reflections appear smaller because they are further away).
        4. Scale that rectangle back up to full resolution.
        5. Run the shape predictor on the *full-res* grey frame inside the
           scaled rectangle for maximum landmark accuracy.

        Args:
            frame: BGR image as produced by OpenCV / Picamera2.

        Returns:
            ``(landmarks, rect)`` where ``landmarks`` is an ``(68, 2)`` int32
            array of ``(x, y)`` pixel coordinates in full-resolution space and
            ``rect`` is the corresponding full-resolution ``dlib.rectangle``.
            Returns ``(None, None)`` when no face is detected.
        """
        if frame is None or frame.size == 0:
            return None, None

        # Greyscale once; both detection and prediction use it.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Only bother resizing when it actually reduces work.
        if self.scale_factor < 1.0:
            small = cv2.resize(
                gray,
                None,
                fx=self.scale_factor,
                fy=self.scale_factor,
                interpolation=cv2.INTER_AREA,  # best quality for downscaling
            )
        else:
            small = gray

        # Second argument is the number of upsampling passes. 0 = none, which
        # is fastest; the driver's face is large enough at 640x480 not to need it.
        faces = self.detector(small, 0)
        if len(faces) == 0:
            return None, None

        # Largest face by area — the driver is closest to the camera.
        largest_small = max(faces, key=lambda r: r.width() * r.height())

        # Map the rectangle back into full-resolution coordinates so the
        # predictor sees the un-downscaled pixels.
        rect = self._scale_rect(largest_small, 1.0 / self.scale_factor)

        # dlib returns a full_object_detection; convert to a plain numpy array
        # so downstream modules don't depend on dlib types.
        shape = self.predictor(gray, rect)
        landmarks = np.array(
            [(shape.part(i).x, shape.part(i).y) for i in range(NUM_LANDMARKS)],
            dtype=np.int32,
        )

        return landmarks, rect

    @staticmethod
    def get_region(landmarks: np.ndarray, region: str) -> np.ndarray:
        """
        Slice the landmark array to a named facial region.

        Args:
            landmarks: ``(68, 2)`` array returned by :meth:`extract`.
            region: One of ``"left_eye"``, ``"right_eye"``, ``"mouth"``,
                ``"nose"``, ``"jaw"`` (case-insensitive).

        Returns:
            A view of the landmark array containing only that region's points,
            e.g. ``(6, 2)`` for an eye.

        Raises:
            KeyError: If ``region`` is not a recognised region name.
        """
        key = region.lower()
        if key not in REGIONS:
            raise KeyError(
                f"Unknown region '{region}'. Valid regions: {sorted(REGIONS)}"
            )
        start, stop = REGIONS[key]
        return landmarks[start:stop]

    @staticmethod
    def draw_landmarks(
        frame: np.ndarray,
        landmarks: np.ndarray,
        rect: Optional[dlib.rectangle] = None,
    ) -> np.ndarray:
        """
        Draw landmark points (and optionally the face box) for debugging.

        The input frame is *not* modified — a copy is annotated and returned so
        the caller can keep feeding the clean frame to the detection pipeline.

        Args:
            frame: BGR image to annotate.
            landmarks: ``(68, 2)`` landmark array.
            rect: Optional full-resolution face rectangle to draw as a box.

        Returns:
            Annotated copy of ``frame``.
        """
        annotated = frame.copy()

        if rect is not None:
            cv2.rectangle(
                annotated,
                (rect.left(), rect.top()),
                (rect.right(), rect.bottom()),
                (0, 255, 0),  # green box
                2,
            )

        # Eyes are drawn in a distinct colour because they are what the rest
        # of the pipeline cares about; everything else is plain white.
        eye_indices = set(range(*LEFT_EYE)) | set(range(*RIGHT_EYE))
        for idx, (x, y) in enumerate(landmarks):
            color = (0, 255, 255) if idx in eye_indices else (255, 255, 255)
            cv2.circle(annotated, (int(x), int(y)), 1, color, -1)

        return annotated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scale_rect(rect: dlib.rectangle, factor: float) -> dlib.rectangle:
        """
        Multiply every coordinate of a dlib rectangle by ``factor``.

        Used to map a rectangle detected on the downscaled frame back to
        full-resolution coordinates (factor = 1 / scale_factor).

        Args:
            rect: Rectangle to scale.
            factor: Multiplier applied to left/top/right/bottom.

        Returns:
            A new ``dlib.rectangle`` with scaled integer coordinates.
        """
        return dlib.rectangle(
            left=int(round(rect.left() * factor)),
            top=int(round(rect.top() * factor)),
            right=int(round(rect.right() * factor)),
            bottom=int(round(rect.bottom() * factor)),
        )
