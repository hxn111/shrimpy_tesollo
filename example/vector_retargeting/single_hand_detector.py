"""
single_hand_detector.py — rewritten for MediaPipe 0.10.x (Tasks API).

MediaPipe 0.10.18+ removed mp.solutions entirely; the new Tasks API is used here.
A model file (hand_landmarker.task) is required and will be auto-downloaded on
first run to the same directory as this script.
"""
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

# ---------------------------------------------------------------------------
# Model auto-download
# ---------------------------------------------------------------------------
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        print(f"[SingleHandDetector] Downloading hand landmarker model to:\n  {_MODEL_PATH}")
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print("[SingleHandDetector] Download complete.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download hand landmarker model: {e}\n"
                f"Please download it manually from:\n  {_MODEL_URL}\n"
                f"and place it at:\n  {_MODEL_PATH}"
            )
    return _MODEL_PATH


# ---------------------------------------------------------------------------
# Hand skeleton connections (for drawing)
# ---------------------------------------------------------------------------
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),                # palm cross
]

# ---------------------------------------------------------------------------
# Coordinate transforms (unchanged from original)
# ---------------------------------------------------------------------------
OPERATOR2MANO_RIGHT = np.array(
    [[0, 0, -1],
     [-1, 0, 0],
     [0, 1, 0]]
)

OPERATOR2MANO_LEFT = np.array(
    [[0, 0, -1],
     [1, 0, 0],
     [0, -1, 0]]
)


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------
class SingleHandDetector:
    def __init__(
        self,
        hand_type: str = "Right",
        min_detection_confidence: float = 0.8,
        min_tracking_confidence: float = 0.8,
        selfie: bool = False,
    ):
        model_path = _ensure_model()

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(                                                                                                                                                                                                                                                                
                model_asset_path=model_path,                                                                                                                                                                                                                                                                   
                delegate=mp_python.BaseOptions.Delegate.GPU
            ), 
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hand_detector = mp_vision.HandLandmarker.create_from_options(options)
        self.selfie = selfie
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )

        # NOTE: Tasks API reports handedness from the *person's* perspective
        # (not mirrored), unlike the legacy Solutions API.
        # For a standard (non-selfie) camera: person's Right hand → label "Right".
        # Legacy code inverted this; Tasks API does NOT need inversion.
        self.detected_hand_type = hand_type

    @staticmethod
    def draw_skeleton_on_image(image, keypoint_2d, style: str = "white"):
        """Draw hand skeleton on image. keypoint_2d is a list of NormalizedLandmark."""
        h, w = image.shape[:2]

        # Accept both Tasks API (list of NormalizedLandmark) and legacy format
        try:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in keypoint_2d]
        except TypeError:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in keypoint_2d.landmark]

        dot_color = (255, 48, 48) if style == "white" else (0, 255, 0)

        for a, b in _HAND_CONNECTIONS:
            cv2.line(image, pts[a], pts[b], (255, 255, 255), 2)
        for pt in pts:
            cv2.circle(image, pt, 4, dot_color, -1)

        return image

    def detect(self, rgb: np.ndarray):
        """
        Detect one hand in an RGB image.

        Returns
        -------
        num_box        : int   — number of detected hands (0 if none)
        joint_pos      : np.ndarray (21, 3) in MANO frame, or None
        keypoint_2d    : list of NormalizedLandmark, or None
        wrist_rot      : np.ndarray (3, 3), or None
        """
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
        result = self.hand_detector.detect(mp_image)

        if not result.hand_world_landmarks:
            return 0, None, None, None

        desired_hand_num = -1
        for i, handedness in enumerate(result.handedness):
            label = handedness[0].category_name  # "Left" or "Right"
            if label == self.detected_hand_type:
                desired_hand_num = i
                break

        if desired_hand_num < 0:
            return 0, None, None, None

        keypoint_3d = result.hand_world_landmarks[desired_hand_num]
        keypoint_2d = result.hand_landmarks[desired_hand_num]
        num_box = len(result.hand_landmarks)

        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano

        return num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot

    @staticmethod
    def parse_keypoint_3d(keypoint_3d) -> np.ndarray:
        """Parse Tasks API Landmark list → (21, 3) numpy array."""
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d[i].x
            keypoint[i][1] = keypoint_3d[i].y
            keypoint[i][2] = keypoint_3d[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(keypoint_2d, img_size) -> np.ndarray:
        """Parse Tasks API NormalizedLandmark list → pixel coordinates (21, 2)."""
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d[i].x
            keypoint[i][1] = keypoint_2d[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from 3D key points.
        Uses wrist (0), index MCP (5), middle MCP (9).
        Returns a (3, 3) rotation matrix in MANO convention.
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        x_vector = points[0] - points[2]

        points = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(points)
        normal = v[2, :]

        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1

        return np.stack([x, normal, z], axis=1)