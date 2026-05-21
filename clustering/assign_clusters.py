"""Assign clusters to feature sequences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import yaml

from clustering.cluster_utils import load_feature_file, load_kmeans, assign_clusters


def _load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def assign_for_feature_file(feature_path: Path, config: Dict, output_dir: Path) -> Path:
    cluster_dir = Path(config["clustering"]["cache_dir"])
    models = {
        "face": load_kmeans(cluster_dir / "face_kmeans.pkl"),
        "left": load_kmeans(cluster_dir / "left_hand_kmeans.pkl"),
        "right": load_kmeans(cluster_dir / "right_hand_kmeans.pkl"),
        "body": load_kmeans(cluster_dir / "body_kmeans.pkl"),
    }

    feats = load_feature_file(feature_path)
    face_labels = assign_clusters(models["face"], feats["face"])
    left_labels = assign_clusters(models["left"], feats["left_hand"])
    right_labels = assign_clusters(models["right"], feats["right_hand"])
    body_labels = assign_clusters(models["body"], feats["body"])

    cluster_map = {}
    for idx in range(face_labels.shape[0]):
        cluster_map[f"frame_{idx}"] = {
            "face": int(face_labels[idx]),
            "left": int(left_labels[idx]),
            "right": int(right_labels[idx]),
            "body": int(body_labels[idx]),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{feature_path.stem}_cluster_map.json"
    with open(output_path, "w") as f:
        json.dump(cluster_map, f, indent=2)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--feature_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("cache/cluster_maps"))
    args = parser.parse_args()

    cfg = _load_config(args.config)
    assign_for_feature_file(args.feature_file, cfg, args.output_dir)
