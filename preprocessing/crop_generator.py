"""Crop generator for face and hand regions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class CropConfig:
    crop_size: int = 224
    face_padding: float = 0.2
    hand_padding: float = 0.3


class CropGenerator:
    """Generate crops for face and hands based on landmarks."""

    def __init__(self, config: CropConfig) -> None:
        self.config = config
        self.left_eye_indices = [69, 168, 156, 118, 54]
        self.right_eye_indices = [168, 299, 347, 336, 301]
        self.mouth_indices = [164, 212, 432, 18]

    def generate(self, frames: np.ndarray, landmarks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate crops for all frames.

        Args:
            frames: [T, H, W, C] RGB frames.
            landmarks: Dict of landmark arrays.

        Returns:
            Dict with face, left_hand, right_hand crops of shape [T, 224, 224, 3].
        """
        face_crops = []
        left_crops = []
        right_crops = []

        for idx, frame in enumerate(frames):
            face_crops.append(self._face_crop(frame, landmarks["face"][idx]))
            left_crops.append(self._hand_crop(frame, landmarks["left_hand"][idx]))
            right_crops.append(self._hand_crop(frame, landmarks["right_hand"][idx]))

        return {
            "face": np.stack(face_crops),
            "left_hand": np.stack(left_crops),
            "right_hand": np.stack(right_crops),
        }

    def _face_crop(self, frame: np.ndarray, face_landmarks: np.ndarray) -> np.ndarray:
        if np.allclose(face_landmarks, 0.0):
            return np.zeros((self.config.crop_size, self.config.crop_size, 3), dtype=np.uint8)
        h, w, _ = frame.shape
        bbox = self._bbox_from_landmarks(face_landmarks, w, h, self.config.face_padding)
        crop = self._safe_crop(frame, bbox)

        eyes_mouth = self._eyes_mouth_mask(face_landmarks, w, h, bbox)
        blurred = cv2.GaussianBlur(crop, (15, 15), 0)
        if eyes_mouth is not None:
            x0, y0, x1, y1 = eyes_mouth
            blurred[y0:y1, x0:x1] = crop[y0:y1, x0:x1]
        resized = cv2.resize(blurred, (self.config.crop_size, self.config.crop_size), interpolation=cv2.INTER_AREA)
        return resized

    def _hand_crop(self, frame: np.ndarray, hand_landmarks: np.ndarray) -> np.ndarray:
        if np.allclose(hand_landmarks, 0.0):
            return np.zeros((self.config.crop_size, self.config.crop_size, 3), dtype=np.uint8)
        h, w, _ = frame.shape
        bbox = self._bbox_from_landmarks(hand_landmarks, w, h, self.config.hand_padding)
        crop = self._safe_crop(frame, bbox)
        resized = cv2.resize(crop, (self.config.crop_size, self.config.crop_size), interpolation=cv2.INTER_AREA)
        return resized

    def _bbox_from_landmarks(
        self, landmarks: np.ndarray, width: int, height: int, padding: float
    ) -> Tuple[int, int, int, int]:
        xs = landmarks[:, 0] * width
        ys = landmarks[:, 1] * height
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        pad_x = (x_max - x_min) * padding
        pad_y = (y_max - y_min) * padding
        x0 = int(max(x_min - pad_x, 0))
        y0 = int(max(y_min - pad_y, 0))
        x1 = int(min(x_max + pad_x, width - 1))
        y1 = int(min(y_max + pad_y, height - 1))
        return x0, y0, x1, y1

    def _eyes_mouth_mask(
        self, face_landmarks: np.ndarray, width: int, height: int, face_bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int] | None:
        if face_landmarks.shape[0] < max(self.mouth_indices) + 1:
            return None
        indices = self.left_eye_indices + self.right_eye_indices + self.mouth_indices
        xs = face_landmarks[indices, 0] * width
        ys = face_landmarks[indices, 1] * height
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x0, y0, x1, y1 = face_bbox
        x_min = int(max(x_min - x0, 0))
        x_max = int(min(x_max - x0, x1 - x0))
        y_min = int(max(y_min - y0, 0))
        y_max = int(min(y_max - y0, y1 - y0))
        if x_max <= x_min or y_max <= y_min:
            return None
        return x_min, y_min, x_max, y_max

    @staticmethod
    def _safe_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return frame[y0:y1, x0:x1]
