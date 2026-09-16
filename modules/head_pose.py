"""
Module 10 - Head pose estimation.

`HeadPoseEstimator` recovers the driver's 3D head orientation (pitch, yaw,
roll) from the 2D dlib landmarks using OpenCV's ``solvePnP``. No trained
model is required: six landmarks are matched against a generic 3D face model
(the widely used LearnOpenCV reference geometry, in millimetres) and PnP
solves for the rotation that projects the model onto the observed points.

Head pose is a fatigue cue that is independent of the eyes:

* **pitch** - nodding. A drowsy driver's head drops forward (chin to chest).
* **yaw**   - looking away from the road for an extended time.
* **roll**  - sideways tilt, typical of a driver slumping against the door.

Angle conventions (all in degrees, ``0`` = facing the camera squarely):

* ``pitch < 0``  -> head tilted **down** (forehead towards the camera).
* ``yaw < 0``    -> nose moved towards the **image's left** edge.
* ``roll < 0``   -> head tilted towards the **image's left** edge.

The thresholds in :class:`HeadPoseEstimator` flag a pose as an alert when any
one of the three angles exceeds its limit.

Implementation note: the reference model has +y pointing *up* (chin at
``y = -330``) while image coordinates have +y pointing *down*. A frontal face
therefore corresponds to a 180 deg rotation about x, not to the identity. The
code removes that fixed offset before extracting Euler angles so a frontal
face reads as ``(0, 0, 0)`` rather than ``(180, 0, 180)``.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Reference geometry
# ---------------------------------------------------------------------------

# dlib 68-point indices of the six landmarks used for PnP, in the same order
# as MODEL_POINTS below.
POSE_LANDMARK_INDICES: Tuple[int, ...] = (30, 8, 36, 45, 48, 54)

# Generic 3D face model (mm), origin at the nose tip, +x right, +y up, +z
# towards the viewer. Values are the standard ones from the LearnOpenCV head
# pose tutorial and are accurate enough for coarse pitch/yaw/roll.
MODEL_POINTS: np.ndarray = np.array(
    [
        [0.0, 0.0, 0.0],          # 30 - nose tip
        [0.0, -330.0, -65.0],     # 8  - chin
        [-225.0, 170.0, -135.0],  # 36 - left eye outer corner
        [225.0, 170.0, -135.0],   # 45 - right eye outer corner
        [-150.0, -150.0, -125.0], # 48 - left mouth corner
        [150.0, -150.0, -125.0],  # 54 - right mouth corner
    ],
    dtype=np.float64,
)

# Fixed rotation that maps the model frame (+y up, +z to viewer) onto the
# camera frame (+y down, +z away from camera): a 180 deg turn about x.
# It is its own inverse, so it is used both to build synthetic faces and to
# strip the offset from solvePnP's result.
_MODEL_TO_CAMERA: np.ndarray = np.diag([1.0, -1.0, -1.0])

# Length (mm) of the axes drawn by draw_pose_axes().
_AXIS_LENGTH_MM: float = 300.0


class HeadPoseEstimator:
    """
    Estimate pitch / yaw / roll of the head from 68-point landmarks.

    Typical usage::

        estimator = HeadPoseEstimator(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        pose = estimator.estimate(landmarks)          # landmarks: (68, 2)
        if pose and pose["alert"]:
            print(estimator.get_status_text(pose))   # e.g. "NODDING"

    Attributes:
        frame_width: Image width in pixels used to build the camera matrix.
        frame_height: Image height in pixels.
        camera_matrix: 3x3 pinhole intrinsics (focal length = frame width).
        dist_coeffs: Lens distortion, assumed zero.
    """

    # Angle limits (degrees) beyond which a pose is flagged.
    PITCH_THRESHOLD: float = -15.0  # head down (nodding)
    YAW_THRESHOLD: float = 30.0     # |yaw| - looking away
    ROLL_THRESHOLD: float = 20.0    # |roll| - tilting

    def __init__(self, frame_width: int = 640, frame_height: int = 480) -> None:
        """
        Build the pinhole camera model from the frame size.

        Using the frame width as the focal length is the usual approximation
        for an uncalibrated webcam (roughly a 53 deg horizontal field of view)
        and is accurate enough for threshold-based pose alerts.

        Args:
            frame_width: Width of the frames that will be passed to
                :meth:`estimate`, in pixels.
            frame_height: Height of those frames, in pixels.
        """
        self.frame_width: int = frame_width
        self.frame_height: int = frame_height

        focal_length = float(frame_width)
        center_x = frame_width / 2.0
        center_y = frame_height / 2.0
        self.camera_matrix: np.ndarray = np.array(
            [
                [focal_length, 0.0, center_x],
                [0.0, focal_length, center_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # No lens distortion assumed (4 coefficients: k1, k2, p1, p2).
        self.dist_coeffs: np.ndarray = np.zeros((4, 1), dtype=np.float64)

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate(self, landmarks: np.ndarray) -> Optional[Dict[str, object]]:
        """
        Solve for the head orientation.

        Args:
            landmarks: ``(68, 2)`` array of dlib landmarks in pixel
                coordinates.

        Returns:
            A dict::

                {
                  "pitch": float,            # deg, negative = head down
                  "yaw": float,              # deg, negative = towards image left
                  "roll": float,             # deg, negative = tilt to image left
                  "is_nodding": bool,        # pitch < PITCH_THRESHOLD
                  "is_looking_away": bool,   # |yaw| > YAW_THRESHOLD
                  "is_tilting": bool,        # |roll| > ROLL_THRESHOLD
                  "alert": bool,             # any of the three
                }

            or ``None`` if the landmarks are malformed or ``solvePnP`` fails.
        """
        lm = np.asarray(landmarks, dtype=np.float64)
        if lm.ndim != 2 or lm.shape[0] < max(POSE_LANDMARK_INDICES) + 1 or lm.shape[1] != 2:
            return None

        image_points = lm[list(POSE_LANDMARK_INDICES)]
        # Degenerate input (e.g. all six points coincide) makes PnP meaningless.
        if not np.isfinite(image_points).all() or np.ptp(image_points, axis=0).min() <= 0.0:
            return None

        try:
            success, rvec, _tvec = cv2.solvePnP(
                MODEL_POINTS,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None
        if not success:
            return None

        pitch, yaw, roll = self._euler_from_rvec(rvec)

        is_nodding = pitch < self.PITCH_THRESHOLD
        is_looking_away = abs(yaw) > self.YAW_THRESHOLD
        is_tilting = abs(roll) > self.ROLL_THRESHOLD

        return {
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll),
            "is_nodding": bool(is_nodding),
            "is_looking_away": bool(is_looking_away),
            "is_tilting": bool(is_tilting),
            "alert": bool(is_nodding or is_looking_away or is_tilting),
        }

    @staticmethod
    def _euler_from_rvec(rvec: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert a solvePnP rotation vector into (pitch, yaw, roll) degrees.

        Steps:
        1. ``cv2.Rodrigues`` -> 3x3 rotation matrix ``R`` (model -> camera).
        2. Right-multiply by ``_MODEL_TO_CAMERA`` to remove the fixed 180 deg
           flip between the model's +y-up frame and the camera's +y-down
           frame, leaving only the head's own rotation ``R_head``.
        3. ``cv2.RQDecomp3x3`` -> Euler angles (x, y, z) in degrees.
        4. Re-sign so that head-down pitch and image-left yaw/roll are
           negative (see module docstring).

        Args:
            rvec: ``(3, 1)`` Rodrigues rotation vector from ``solvePnP``.

        Returns:
            ``(pitch, yaw, roll)`` in degrees.
        """
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        r_head = rotation_matrix @ _MODEL_TO_CAMERA

        # RQDecomp3x3 returns the Euler angles (in degrees) as its first value.
        angles, *_ = cv2.RQDecomp3x3(r_head)
        x_rot, y_rot, z_rot = (float(a) for a in angles)

        # In the camera frame (+y down, +z forward) a positive rotation about
        # x brings the forehead towards the camera (head down), so negate to
        # get "negative = down". A positive y rotation swings the nose towards
        # image left already, but RQDecomp3x3 reports it with the opposite
        # sign, so negate for "negative = left". A positive z rotation moves
        # the top of the head towards image right, which is the sign we want
        # ("negative = tilted towards image left"), so roll is kept as-is.
        pitch = -x_rot
        yaw = -y_rot
        roll = z_rot
        return pitch, yaw, roll

    # ------------------------------------------------------------------
    # Visualisation / reporting
    # ------------------------------------------------------------------

    def draw_pose_axes(
        self, frame: np.ndarray, landmarks: np.ndarray, pose: Dict[str, object]
    ) -> np.ndarray:
        """
        Draw the head's X/Y/Z axes projected from the nose tip (debug aid).

        The axes are re-derived from ``pose`` (pitch/yaw/roll) so the drawing
        always matches the numbers being reported, and are projected with the
        same camera model used for estimation.

        Colours: red = X (roll axis), green = Y (pitch axis), blue = Z (yaw
        axis, points out of the face).

        Args:
            frame: BGR image to annotate. Not modified.
            landmarks: ``(68, 2)`` landmarks; only the nose tip (30) is used
                as the axis origin.
            pose: Dict returned by :meth:`estimate`.

        Returns:
            A copy of ``frame`` with the axes drawn.
        """
        annotated = frame.copy()
        lm = np.asarray(landmarks, dtype=np.float64)
        nose = tuple(int(v) for v in lm[30])

        # Rebuild R_head from the reported angles (inverse of _euler_from_rvec)
        # then re-apply the model->camera flip so we can reuse solvePnP's
        # projection conventions.
        pitch = -float(pose["pitch"])
        yaw = -float(pose["yaw"])
        roll = float(pose["roll"])
        r_head = self._rotation_from_euler_deg(pitch, yaw, roll)
        rotation = r_head @ _MODEL_TO_CAMERA
        rvec, _ = cv2.Rodrigues(rotation)

        # Place the model at a nominal depth in front of the camera and shift
        # it so the projected nose lands on the detected nose tip.
        tvec = np.array([[0.0], [0.0], [1000.0]], dtype=np.float64)
        nose_proj, _ = cv2.projectPoints(
            np.zeros((1, 3)), rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        offset = np.array(nose, dtype=np.float64) - nose_proj.reshape(2)

        axes_3d = np.array(
            [
                [_AXIS_LENGTH_MM, 0.0, 0.0],
                [0.0, _AXIS_LENGTH_MM, 0.0],
                [0.0, 0.0, _AXIS_LENGTH_MM],
            ],
            dtype=np.float64,
        )
        axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        axes_2d = axes_2d.reshape(-1, 2) + offset

        colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: red, green, blue
        for end, colour in zip(axes_2d, colours):
            cv2.line(annotated, nose, (int(end[0]), int(end[1])), colour, 2, cv2.LINE_AA)
        return annotated

    @staticmethod
    def _rotation_from_euler_deg(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
        """
        Compose a rotation matrix from Euler angles in the RQDecomp3x3 order.

        ``cv2.RQDecomp3x3`` factors ``R = Qx @ Qy @ Qz`` (x, y, z in degrees);
        this rebuilds ``R`` from those angles.

        Args:
            x_deg: Rotation about x, degrees.
            y_deg: Rotation about y, degrees.
            z_deg: Rotation about z, degrees.

        Returns:
            ``(3, 3)`` rotation matrix.
        """
        x, y, z = np.radians([x_deg, y_deg, z_deg])
        qx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        qy = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        qz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return qx @ qy @ qz

    def get_status_text(self, pose: Dict[str, object]) -> str:
        """
        Human-readable summary of a pose.

        Priority when several flags are set: nodding is the strongest
        drowsiness cue, then looking away, then tilting.

        Args:
            pose: Dict returned by :meth:`estimate`.

        Returns:
            ``"NODDING"``, ``"LOOKING AWAY"``, ``"TILTING"`` or ``"ALERT"``
            (meaning the head is in a normal, attentive position).
        """
        if pose["is_nodding"]:
            return "NODDING"
        if pose["is_looking_away"]:
            return "LOOKING AWAY"
        if pose["is_tilting"]:
            return "TILTING"
        return "ALERT"
