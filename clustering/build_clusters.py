"""Build or adapt k-means clusters for ISL features."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

from clustering.cluster_utils import (
    assign_clusters,
    iter_feature_files,
    load_feature_file,
    load_kmeans,
    save_kmeans,
    build_kmeans,
    partial_fit_kmeans,
    load_asl_centers,
)


def _load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: Path) -> None:
    cfg = _load_config(config_path)
    feature_dir = Path(cfg["features"]["cache_dir"])
    cluster_dir = Path(cfg["clustering"]["cache_dir"])
    asl_dir = Path(cfg["clustering"]["asl_cluster_dir"])
    mode = cfg["clustering"]["mode"]
    k = int(cfg["clustering"]["k"])
    batch_size = int(cfg["clustering"]["batch_size"])

    cluster_dir.mkdir(parents=True, exist_ok=True)

    if mode == "reuse":
        for part in ["face", "left_hand", "right_hand", "body"]:
            src = asl_dir / f"{part}_kmeans.pkl"
            if not src.exists():
                raise FileNotFoundError(f"Missing ASL cluster model: {src}")
            model = load_kmeans(src)
            save_kmeans(model, cluster_dir / f"{part}_kmeans.pkl")
        return

    if mode != "adapt":
        raise ValueError("Invalid clustering mode. Use 'reuse' or 'adapt'.")

    centers = None
    if asl_dir.exists():
        try:
            centers = load_asl_centers(asl_dir)
        except Exception:
            centers = None

    models = {}
    for part in ["face", "left_hand", "right_hand", "body"]:
        init_centers = centers[part] if centers is not None else None
        models[part] = build_kmeans(
            features=_load_seed_features(feature_dir, part),
            k=k,
            batch_size=batch_size,
            init_centers=init_centers,
        )

    for feature_path in iter_feature_files(feature_dir):
        data = load_feature_file(feature_path)
        for part, model in models.items():
            partial_fit_kmeans(model, data[part], batch_size)

    for part, model in models.items():
        save_kmeans(model, cluster_dir / f"{part}_kmeans.pkl")


def _load_seed_features(feature_dir: Path, part: str) -> "np.ndarray":
    for feature_path in iter_feature_files(feature_dir):
        data = load_feature_file(feature_path)
        if data[part].shape[0] > 0:
            return data[part]
    return np.zeros((1, 256), dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()
    main(args.config)
