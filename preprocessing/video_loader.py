"""Video loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def read_video(path: Path) -> Tuple[List[np.ndarray], float]:
    """Read video frames with OpenCV.

    Args:
        path: Path to a video file.

    Returns:
        Tuple of (frames, fps).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames, fps


def resize_frames(frames: List[np.ndarray], size: Tuple[int, int]) -> List[np.ndarray]:
    """Resize frames to a fixed size.

    Args:
        frames: List of RGB frames.
        size: (width, height).

    Returns:
        Resized frames.
    """
    width, height = size
    return [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames]
