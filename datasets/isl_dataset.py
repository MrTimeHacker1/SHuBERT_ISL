"""ISL video dataset loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from preprocessing.video_loader import read_video, resize_frames
from utils.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class DatasetConfig:
    root: Path
    num_frames: int = 64
    frame_size: Tuple[int, int] = (256, 256)
    recursive: bool = True
    extensions: Tuple[str, ...] = (".mp4", ".MP4")


class ISLVideoDataset:
    """Dataset that loads ISL videos and returns uniformly sampled frames."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.video_paths = self._collect_videos(config.root)

    def _collect_videos(self, root: Path) -> List[Path]:
        if self.config.recursive:
            paths = [p for p in root.rglob("*") if p.suffix in self.config.extensions]
        else:
            paths = [p for p in root.iterdir() if p.suffix in self.config.extensions]
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Path]:
        path = self.video_paths[index]
        frames, _ = read_video(path)
        if not frames:
            logger.warning("Failed to load video: %s", path)
            frames = [np.zeros((self.config.frame_size[1], self.config.frame_size[0], 3), dtype=np.uint8)]

        frames = resize_frames(frames, self.config.frame_size)
        sampled = self._uniform_sample(frames, self.config.num_frames)
        return sampled, path

    def _uniform_sample(self, frames: List[np.ndarray], num_frames: int) -> np.ndarray:
        total = len(frames)
        if total == 0:
            return np.zeros((num_frames, self.config.frame_size[1], self.config.frame_size[0], 3), dtype=np.uint8)

        if total >= num_frames:
            indices = np.linspace(0, total - 1, num_frames).astype(int)
            sampled = [frames[i] for i in indices]
        else:
            sampled = frames.copy()
            pad_count = num_frames - total
            sampled.extend([frames[-1]] * pad_count)

        return np.stack(sampled, axis=0)


class ISLFeatureDataset:
    """Dataset that loads cached features and cluster labels."""

    def __init__(self, feature_dir: Path, cluster_dir: Path) -> None:
        self.feature_paths = sorted(feature_dir.glob("*.npz"))
        self.cluster_dir = cluster_dir
        self._models = None

    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, dict]:
        feature_path = self.feature_paths[index]
        data = np.load(feature_path)
        features = np.stack(
            [data["face"], data["left_hand"], data["right_hand"], data["body"]],
            axis=1,
        )
        labels = self._assign_clusters(data)
        return features, labels

    def _assign_clusters(self, data: np.lib.npyio.NpzFile) -> dict:
        if self._models is None:
            import joblib

            self._models = {
                "face": joblib.load(self.cluster_dir / "face_kmeans.pkl"),
                "left": joblib.load(self.cluster_dir / "left_hand_kmeans.pkl"),
                "right": joblib.load(self.cluster_dir / "right_hand_kmeans.pkl"),
                "body": joblib.load(self.cluster_dir / "body_kmeans.pkl"),
            }
        labels = {
            "face": self._models["face"].predict(data["face"]),
            "left": self._models["left"].predict(data["left_hand"]),
            "right": self._models["right"].predict(data["right_hand"]),
            "body": self._models["body"].predict(data["body"]),
        }
        return labels
