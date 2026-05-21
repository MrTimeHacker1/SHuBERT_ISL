"""Metrics utilities."""
from __future__ import annotations

import torch


def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute accuracy over masked positions.

    Args:
        logits: [B, T, K] tensor.
        targets: [B, T] tensor.
        mask: [B, T] boolean tensor where True indicates masked positions.

    Returns:
        Accuracy as float.
    """
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds == targets) & mask
    return correct.sum().item() / mask.sum().item()
