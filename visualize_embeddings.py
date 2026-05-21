"""Visualize embeddings with t-SNE, UMAP, and temporal plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from utils.visualization import plot_tsne, plot_umap, plot_temporal_trajectory


def load_cluster_labels(cluster_map_path: Path) -> Optional[np.ndarray]:
    if not cluster_map_path.exists():
        return None
    with open(cluster_map_path, "r") as f:
        data = json.load(f)
    labels = [entry["face"] for entry in data.values()]
    return np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=Path("outputs/embeddings.npy"))
    parser.add_argument("--cluster_map", type=Path, default=Path("outputs/cluster_map.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    labels = load_cluster_labels(args.cluster_map)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_tsne(embeddings, labels, args.output_dir / "tsne.png")
    plot_umap(embeddings, labels, args.output_dir / "umap.png")
    plot_temporal_trajectory(embeddings, args.output_dir / "trajectory.png")


if __name__ == "__main__":
    main()
