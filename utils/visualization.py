"""Embedding visualization helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import umap


def plot_tsne(embeddings: np.ndarray, labels: Optional[np.ndarray], output_path: Path) -> None:
    """Generate a t-SNE plot.

    Args:
        embeddings: [N, D] array.
        labels: Optional [N] labels for coloring.
        output_path: Path to save the plot.
    """
    tsne = TSNE(n_components=2, init="pca", random_state=42)
    reduced = tsne.fit_transform(embeddings)

    _scatter_plot(reduced, labels, output_path, title="t-SNE")


def plot_umap(embeddings: np.ndarray, labels: Optional[np.ndarray], output_path: Path) -> None:
    """Generate a UMAP plot.

    Args:
        embeddings: [N, D] array.
        labels: Optional [N] labels for coloring.
        output_path: Path to save the plot.
    """
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    _scatter_plot(reduced, labels, output_path, title="UMAP")


def plot_temporal_trajectory(embeddings: np.ndarray, output_path: Path) -> None:
    """Plot temporal trajectory over first two principal components.

    Args:
        embeddings: [T, D] array.
        output_path: Path to save the plot.
    """
    if embeddings.shape[0] < 2:
        return
    coords = embeddings[:, :2]
    plt.figure(figsize=(8, 6))
    plt.plot(coords[:, 0], coords[:, 1], marker="o", linewidth=1)
    plt.title("Temporal Embedding Trajectory")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _scatter_plot(points: np.ndarray, labels: Optional[np.ndarray], output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    if labels is None:
        plt.scatter(points[:, 0], points[:, 1], s=5)
    else:
        plt.scatter(points[:, 0], points[:, 1], s=5, c=labels, cmap="tab20")
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
