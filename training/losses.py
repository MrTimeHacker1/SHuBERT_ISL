"""Loss functions for SHuBERT training."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


STREAM_ORDER = ["face", "left", "right", "body"]


def ssl_loss(
    logits: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked cross-entropy loss for each stream.

    Args:
        logits: Dict of [B, T, 256] logits.
        targets: Dict of [B, T] cluster labels.
        mask: [B, T, 4] boolean mask.

    Returns:
        Total loss.
    """
    total = 0.0
    for idx, key in enumerate(STREAM_ORDER):
        stream_mask = mask[:, :, idx]
        if stream_mask.sum() == 0:
            continue
        stream_logits = logits[key]
        stream_targets = targets[key]
        loss = F.cross_entropy(
            stream_logits[stream_mask],
            stream_targets[stream_mask],
            reduction="mean",
        )
        total += loss
    return total
