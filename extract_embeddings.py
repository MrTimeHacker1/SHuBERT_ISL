"""Extract contextual embeddings and cluster assignments for a video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

from preprocessing.video_loader import read_video
from preprocessing.mediapipe_extractor import MediaPipeConfig, MediaPipeExtractor
from preprocessing.crop_generator import CropConfig, CropGenerator
from preprocessing.feature_extractor import FeatureConfig, FeatureExtractor
from clustering.cluster_utils import load_kmeans, assign_clusters
from models.shubert_model import SHuBERTConfig, SHuBERTModel
from models.load_pretrained import load_pretrained
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    cfg = load_config(args.config)

    frames, _ = read_video(args.video)
    if not frames:
        raise RuntimeError("Failed to load video")
    frames_np = np.stack(frames)

    extractor = MediaPipeExtractor(MediaPipeConfig(**cfg["preprocessing"]["mediapipe"]))
    landmarks = extractor.process(frames_np)
    crops = CropGenerator(CropConfig(crop_size=int(cfg["preprocessing"]["crop_size"]))).generate(frames_np, landmarks)

    feature_cfg = FeatureConfig(
        dino_model=cfg["features"]["dino_model"],
        dino_dim=int(cfg["features"]["dino_dim"]),
        projection_dim=int(cfg["features"]["projection_dim"]),
        normalize=bool(cfg["features"]["normalize"]),
        cache_dir=Path(cfg["features"]["cache_dir"]),
    )
    feature_extractor = FeatureExtractor(feature_cfg)
    features = feature_extractor.extract(crops, landmarks["body"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = SHuBERTConfig(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        num_heads=int(cfg["model"]["num_heads"]),
        ff_dim=int(cfg["model"]["ff_dim"]),
        num_clusters=int(cfg["model"]["num_clusters"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    model = SHuBERTModel(model_cfg).to(device)
    weights_path = args.checkpoint if args.checkpoint is not None else Path(cfg["model"]["weights_path"])
    model = load_pretrained(model, weights_path, int(cfg["model"]["freeze_layers"]))
    model.eval()

    stream_features = np.stack(
        [features["face"], features["left_hand"], features["right_hand"], features["body"]], axis=1
    )
    with torch.no_grad():
        encoded, _ = model(torch.from_numpy(stream_features).unsqueeze(0).to(device))
    embeddings = encoded.squeeze(0).cpu().numpy()

    cluster_dir = Path(cfg["clustering"]["cache_dir"])
    models = {
        "face": load_kmeans(cluster_dir / "face_kmeans.pkl"),
        "left": load_kmeans(cluster_dir / "left_hand_kmeans.pkl"),
        "right": load_kmeans(cluster_dir / "right_hand_kmeans.pkl"),
        "body": load_kmeans(cluster_dir / "body_kmeans.pkl"),
    }
    face_labels = assign_clusters(models["face"], features["face"])
    left_labels = assign_clusters(models["left"], features["left_hand"])
    right_labels = assign_clusters(models["right"], features["right_hand"])
    body_labels = assign_clusters(models["body"], features["body"])

    cluster_map = {
        f"frame_{idx}": {
            "face": int(face_labels[idx]),
            "left": int(left_labels[idx]),
            "right": int(right_labels[idx]),
            "body": int(body_labels[idx]),
        }
        for idx in range(embeddings.shape[0])
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "embeddings.npy", embeddings)
    with open(args.output_dir / "cluster_map.json", "w") as f:
        json.dump(cluster_map, f, indent=2)

    logger.info("Saved embeddings and cluster map to %s", args.output_dir)


if __name__ == "__main__":
    main()
