"""
Module 2 — Driver identification via face recognition.

`DriverRecognizer` answers the question "who is sitting in the driver's seat?"
by comparing the live camera frame against 128-d face encodings stored in the
Laravel backend. Knowing the driver lets later modules load that person's
calibrated EAR baseline and attribute fatigue events to the correct record.

The heavy lifting is done by the `face_recognition` library (a dlib wrapper):

* ``face_locations``  — find face bounding boxes.
* ``face_encodings``  — compute a 128-d embedding per face.
* ``compare_faces`` / ``face_distance`` — match embeddings against the DB.

Encodings are fetched through the API client (Module 9). The expected payload
shape from ``api_client.get_face_encodings()`` is a list of records::

    [{"driver_id": 1, "name": "Jane Doe", "face_encoding": [0.12, -0.05, ...]}, ...]

The vector may be a list of 128 floats or a JSON-encoded string of one
(Laravel often stores it as a JSON column); both are handled. The key is
``face_encoding`` on the current backend and ``encoding`` on older ones;
either is accepted (see ``ENCODING_KEYS``).
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import face_recognition
import numpy as np

logger = logging.getLogger(__name__)

# Length of the embedding vector produced by face_recognition / dlib's ResNet.
ENCODING_LENGTH: int = 128

# Record keys that may carry the vector, in order of preference. The Laravel
# backend serialises it as ``face_encoding``; ``encoding`` is the older name.
ENCODING_KEYS = ("face_encoding", "encoding")


class DriverRecognizer:
    """
    Identify the current driver by matching their face to stored encodings.

    Typical usage::

        recognizer = DriverRecognizer(api_client, tolerance=0.5)
        recognizer.load_encodings()
        match = recognizer.identify(frame)
        if match:
            print(match["driver_id"], match["name"], match["confidence"])
    """

    def __init__(self, api_client: Any, tolerance: float = 0.5) -> None:
        """
        Store the API client and matching tolerance; no network I/O happens here.

        Args:
            api_client: Instance of the Module 9 API client. Must expose
                ``get_face_encodings()``.
            tolerance: Maximum Euclidean distance between two encodings for
                them to be considered the same person. face_recognition's
                default is 0.6; 0.5 is stricter and reduces false positives,
                which matters because a misidentified driver would load the
                wrong calibration profile.
        """
        self.api_client = api_client
        self.tolerance: float = tolerance

        # driver_id -> 128-d encoding (populated by load_encodings()).
        self.encodings: Dict[int, np.ndarray] = {}

        # driver_id -> display name, kept separately so `encodings` stays a
        # clean {id: vector} mapping for matching.
        self.names: Dict[int, str] = {}

        # Cached parallel lists so identify() doesn't rebuild them per frame.
        # face_recognition's comparison helpers expect a list of encodings.
        self._known_ids: List[int] = []
        self._known_encodings: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Encoding management
    # ------------------------------------------------------------------

    def load_encodings(self) -> int:
        """
        Fetch every driver's face encoding from the API and cache it locally.

        Called once at startup and again (via :meth:`reload_encodings`) after
        a new driver is enrolled. Records with a missing or malformed encoding
        are skipped with a warning rather than aborting the whole load, so one
        bad DB row can't take the recogniser offline.

        Returns:
            Number of encodings successfully loaded.
        """
        records = self.api_client.get_face_encodings() or []

        encodings: Dict[int, np.ndarray] = {}
        names: Dict[int, str] = {}

        for record in records:
            try:
                driver_id = int(record["driver_id"])
                encoding = self._parse_encoding(self._encoding_field(record))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed face encoding record %r: %s", record, exc)
                continue

            encodings[driver_id] = encoding
            # Fall back to a generic label if the API didn't send a name.
            names[driver_id] = str(record.get("name", f"Driver {driver_id}"))

        # Swap in atomically so a frame processed mid-reload never sees a
        # half-populated cache.
        self.encodings = encodings
        self.names = names
        self._known_ids = list(encodings.keys())
        self._known_encodings = [encodings[i] for i in self._known_ids]

        logger.info("Loaded %d driver face encodings", len(self.encodings))
        return len(self.encodings)

    def reload_encodings(self) -> int:
        """
        Re-fetch encodings from the API.

        Thin alias for :meth:`load_encodings` that exists to make call sites
        self-documenting (e.g. after the enrollment endpoint reports a new
        driver was added).

        Returns:
            Number of encodings loaded.
        """
        logger.info("Reloading driver face encodings from API")
        return self.load_encodings()

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def identify(self, frame: np.ndarray) -> Optional[Dict[str, Union[int, str, float]]]:
        """
        Identify the driver in a BGR frame.

        Steps:
        1. Convert BGR -> RGB (face_recognition expects RGB).
        2. Locate faces with the HOG model (CNN is far too slow on the Pi).
        3. Compute a 128-d encoding for each face found.
        4. For each face, compute distances to every known encoding and take
           the closest. Keep the overall best match across all faces.
        5. Reject the match if its distance exceeds ``tolerance``.

        Args:
            frame: BGR image from the camera.

        Returns:
            ``{"driver_id": int, "name": str, "confidence": float}`` for the
            best match, where ``confidence`` is in [0, 1] (1 = identical
            encodings, 0 = at/over tolerance). ``None`` if no face was found,
            no encodings are loaded, or no face is within tolerance.
        """
        if frame is None or frame.size == 0 or not self._known_encodings:
            return None

        # OpenCV/Picamera2 give BGR; face_recognition (dlib) wants RGB.
        rgb = np.ascontiguousarray(frame[:, :, ::-1])

        # HOG detector — same trade-off as Module 1: fast enough on the Pi.
        locations = face_recognition.face_locations(rgb, model="hog")
        if not locations:
            return None

        # One 128-d vector per detected face.
        live_encodings = face_recognition.face_encodings(rgb, locations)

        best_id: Optional[int] = None
        best_distance: float = float("inf")

        for live in live_encodings:
            # Distance to every known driver; lower = more similar.
            distances = face_recognition.face_distance(self._known_encodings, live)

            # compare_faces gives a boolean per known encoding using the same
            # tolerance; we use it as the accept/reject gate and face_distance
            # to rank among the accepted candidates.
            matches = face_recognition.compare_faces(
                self._known_encodings, live, tolerance=self.tolerance
            )

            for idx, (is_match, dist) in enumerate(zip(matches, distances)):
                if is_match and dist < best_distance:
                    best_distance = float(dist)
                    best_id = self._known_ids[idx]

        if best_id is None:
            return None

        # Map distance to an intuitive confidence: 0 distance -> 1.0,
        # distance == tolerance -> 0.0. Linear is good enough for logging
        # and for a simple "is this match solid?" check upstream.
        confidence = max(0.0, min(1.0, 1.0 - best_distance / self.tolerance))

        return {
            "driver_id": best_id,
            "name": self.names.get(best_id, f"Driver {best_id}"),
            "confidence": round(confidence, 4),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encoding_field(record: Dict[str, Any]) -> Union[str, List[float], np.ndarray]:
        """
        Return the raw vector from an API record under whichever key it uses.

        Raises:
            KeyError: If none of ``ENCODING_KEYS`` is present.
        """
        for key in ENCODING_KEYS:
            if key in record:
                return record[key]
        raise KeyError(f"none of {ENCODING_KEYS} present")

    @staticmethod
    def _parse_encoding(raw: Union[str, List[float], np.ndarray]) -> np.ndarray:
        """
        Normalise an encoding from the API into a float64 numpy array.

        Accepts a JSON string (as stored in a Laravel JSON column), a Python
        list, or an existing ndarray.

        Args:
            raw: Encoding in any of the supported representations.

        Returns:
            ``(128,)`` float64 array.

        Raises:
            ValueError: If the decoded vector does not have 128 elements.
        """
        if isinstance(raw, str):
            raw = json.loads(raw)

        encoding = np.asarray(raw, dtype=np.float64).ravel()
        if encoding.shape[0] != ENCODING_LENGTH:
            raise ValueError(
                f"Expected {ENCODING_LENGTH}-d encoding, got {encoding.shape[0]}"
            )
        return encoding
