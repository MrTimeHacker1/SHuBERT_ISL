"""Clustering utilities for SHuBERT."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def load_feature_file(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {
        "face": data["face"],
        "left_hand": data["left_hand"],
        "right_hand": data["right_hand"],
        "body": data["body"],
    }


def iter_feature_files(feature_dir: Path) -> Iterable[Path]:
    return sorted(feature_dir.glob("*.npz"))


def save_kmeans(model: MiniBatchKMeans, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_kmeans(path: Path) -> MiniBatchKMeans:
    return joblib.load(path)


def build_kmeans(
    features: np.ndarray,
    k: int,
    batch_size: int,
    init_centers: np.ndarray | None = None,
) -> MiniBatchKMeans:
    if init_centers is None:
        init = "k-means++"
    else:
        init = init_centers
    model = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        init=init,
        n_init=1 if init_centers is not None else 10,
    )
    model.fit(features)
    return model


def partial_fit_kmeans(
    model: MiniBatchKMeans, features: np.ndarray, batch_size: int
) -> MiniBatchKMeans:
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size]
        model.partial_fit(batch)
    return model


def assign_clusters(model: MiniBatchKMeans, features: np.ndarray) -> np.ndarray:
    return model.predict(features)


def flatten_features(feature_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "face": feature_dict["face"].reshape(-1, feature_dict["face"].shape[-1]),
        "left_hand": feature_dict["left_hand"].reshape(-1, feature_dict["left_hand"].shape[-1]),
        "right_hand": feature_dict["right_hand"].reshape(-1, feature_dict["right_hand"].shape[-1]),
        "body": feature_dict["body"].reshape(-1, feature_dict["body"].shape[-1]),
    }


def load_asl_centers(asl_dir: Path) -> Dict[str, np.ndarray]:
    centers = {}
    for part in ["face", "left_hand", "right_hand", "body"]:
        path = asl_dir / f"{part}_centers.npy"
        centers[part] = np.load(path)
    return centers
