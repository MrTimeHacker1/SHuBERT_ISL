"""MediaPipe-based landmark extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import mediapipe as mp


@dataclass
class MediaPipeConfig:
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class MediaPipeExtractor:
    """Extracts face, hand, and upper-body landmarks with MediaPipe Holistic."""

    def __init__(self, config: MediaPipeConfig) -> None:
        self.config = config
        self.holistic = mp.solutions.holistic.Holistic(
            model_complexity=config.model_complexity,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            refine_face_landmarks=True,
        )

    def process(self, frames: np.ndarray) -> Dict[str, np.ndarray]:
        """Process frames and return interpolated landmarks.

        Args:
            frames: [T, H, W, C] RGB frames.

        Returns:
            Dict with keys: face, left_hand, right_hand, body.
        """
        face_list = []
        left_list = []
        right_list = []
        body_list = []

        for frame in frames:
            results = self.holistic.process(frame)
            face_list.append(self._landmarks_to_array(results.face_landmarks, 468))
            left_list.append(self._landmarks_to_array(results.left_hand_landmarks, 21))
            right_list.append(self._landmarks_to_array(results.right_hand_landmarks, 21))
            body_list.append(self._pose_to_upper_body(results.pose_landmarks))

        face = self._interpolate(np.stack(face_list))
        left = self._interpolate(np.stack(left_list))
        right = self._interpolate(np.stack(right_list))
        body = self._interpolate(np.stack(body_list))

        return {"face": face, "left_hand": left, "right_hand": right, "body": body}

    @staticmethod
    def _landmarks_to_array(landmarks, count: int) -> np.ndarray:
        if landmarks is None:
            return np.full((count, 3), np.nan, dtype=np.float32)
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
        if coords.shape[0] != count:
            padded = np.full((count, 3), np.nan, dtype=np.float32)
            padded[: coords.shape[0]] = coords
            return padded
        return coords

    @staticmethod
    def _pose_to_upper_body(landmarks) -> np.ndarray:
        indices = [0, 11, 12, 13, 14, 15, 16]
        if landmarks is None:
            return np.full((len(indices), 3), np.nan, dtype=np.float32)
        coords = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y, landmarks.landmark[i].z] for i in indices], dtype=np.float32)
        return coords

    @staticmethod
    def _interpolate(values: np.ndarray) -> np.ndarray:
        """Interpolate NaNs along time dimension for each landmark coordinate."""
        output = values.copy()
        time_dim = output.shape[0]
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                series = output[:, i, j]
                nan_mask = np.isnan(series)
                if nan_mask.all():
                    output[:, i, j] = 0.0
                    continue
                valid_idx = np.where(~nan_mask)[0]
                output[nan_mask, i, j] = np.interp(np.where(nan_mask)[0], valid_idx, series[valid_idx])
        return output
