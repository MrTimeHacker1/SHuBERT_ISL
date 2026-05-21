"""Train SHuBERT on ISL videos with self-supervised objectives."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets.isl_dataset import DatasetConfig, ISLVideoDataset, ISLFeatureDataset
from preprocessing.mediapipe_extractor import MediaPipeConfig, MediaPipeExtractor
from preprocessing.crop_generator import CropConfig, CropGenerator
from preprocessing.feature_extractor import FeatureConfig, FeatureExtractor
from clustering.build_clusters import main as build_clusters
from models.shubert_model import SHuBERTConfig, SHuBERTModel
from models.masking import MaskConfig, StreamMasker
from models.load_pretrained import load_pretrained
from training.trainer import Trainer, default_collate
from training.callbacks import load_checkpoint
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cache_features(cfg: Dict) -> None:
    dataset_cfg = DatasetConfig(
        root=Path(cfg["dataset"]["root"]),
        num_frames=int(cfg["dataset"]["num_frames"]),
        frame_size=tuple(cfg["dataset"]["frame_size"]),
        recursive=bool(cfg["dataset"]["recursive"]),
        extensions=tuple(cfg["dataset"]["extensions"]),
    )
    dataset = ISLVideoDataset(dataset_cfg)
    mediapipe_cfg = MediaPipeConfig(**cfg["preprocessing"]["mediapipe"])
    extractor = MediaPipeExtractor(mediapipe_cfg)
    cropper = CropGenerator(CropConfig(crop_size=int(cfg["preprocessing"]["crop_size"])))
    feature_cfg = FeatureConfig(
        dino_model=cfg["features"]["dino_model"],
        dino_dim=int(cfg["features"]["dino_dim"]),
        projection_dim=int(cfg["features"]["projection_dim"]),
        normalize=bool(cfg["features"]["normalize"]),
        cache_dir=Path(cfg["features"]["cache_dir"]),
    )
    feature_extractor = FeatureExtractor(feature_cfg)

    output_dir = Path(cfg["features"]["cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for frames, path in dataset:
        output_path = output_dir / f"{path.stem}.npz"
        if output_path.exists():
            continue
        landmarks = extractor.process(frames)
        crops = cropper.generate(frames, landmarks)
        features = feature_extractor.extract(crops, landmarks["body"])
        npz_data = {
            "face": features["face"],
            "left_hand": features["left_hand"],
            "right_hand": features["right_hand"],
            "body": features["body"],
            "combined": features["combined"],
        }
        np.savez(output_path, **npz_data)
        logger.info("Cached features: %s", output_path)


def train(cfg: Dict, config_path: Path) -> None:
    cache_features(cfg)

    build_clusters(config_path)

    feature_dir = Path(cfg["features"]["cache_dir"])
    cluster_dir = Path(cfg["clustering"]["cache_dir"])
    dataset = ISLFeatureDataset(feature_dir, cluster_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        collate_fn=default_collate,
    )

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
    model = load_pretrained(model, Path(cfg["model"]["weights_path"]), int(cfg["model"]["freeze_layers"]))

    mask_cfg = MaskConfig(
        span_length=int(cfg["masking"]["span_length"]),
        mask_prob=float(cfg["masking"]["mask_prob"]),
    )
    masker = StreamMasker(mask_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))

    trainer = Trainer(
        model=model,
        masker=masker,
        optimizer=optimizer,
        device=device,
        log_dir=Path(cfg["training"]["log_dir"]),
        checkpoint_dir=Path(cfg["training"]["checkpoint_dir"]),
        mixed_precision=bool(cfg["training"]["mixed_precision"]),
        grad_accum_steps=int(cfg["training"]["grad_accum_steps"]),
        early_stopping_patience=int(cfg["training"]["early_stopping_patience"]),
    )

    resume_from = cfg["training"]["resume_from"]
    if resume_from:
        load_checkpoint(model, optimizer, Path(resume_from))

    trainer.train(dataloader, epochs=int(cfg["training"]["epochs"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--dataset_root", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root is not None:
        cfg["dataset"]["root"] = str(args.dataset_root)

    train(cfg, args.config)


if __name__ == "__main__":
    main()
