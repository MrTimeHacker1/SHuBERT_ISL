"""Utilities to load pretrained SHuBERT weights."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from models.shubert_model import SHuBERTModel
from utils.logger import setup_logger


logger = setup_logger(__name__)


def load_pretrained(model: SHuBERTModel, weights_path: Path, freeze_layers: int) -> SHuBERTModel:
    """Load pretrained weights into the model.

    Args:
        model: SHuBERT model.
        weights_path: Path to pretrained weights.
        freeze_layers: Number of initial layers to freeze.

    Returns:
        Model with loaded weights.
    """
    if not weights_path.exists():
        logger.warning("Weights not found at %s. Training from scratch.", weights_path)
        return model

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.info("Missing keys: %s", missing)
    if unexpected:
        logger.info("Unexpected keys: %s", unexpected)

    model.freeze_layers(freeze_layers)
    return model
