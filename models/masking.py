"""Masking utilities for SHuBERT-style SSL."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class MaskConfig:
    span_length: int = 3
    mask_prob: float = 0.15


class StreamMasker(nn.Module):
    """Applies independent random masking per stream."""

    def __init__(self, config: MaskConfig, stream_dim: int = 256, num_streams: int = 4) -> None:
        super().__init__()
        self.config = config
        self.num_streams = num_streams
        self.stream_dim = stream_dim
        self.mask_embeddings = nn.Parameter(torch.randn(num_streams, stream_dim))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask features.

        Args:
            features: [B, T, S, D] tensor.

        Returns:
            masked_features: Tensor with masked spans.
            mask: Boolean mask [B, T, S] where True indicates masked.
        """
        batch_size, time_steps, streams, _ = features.shape
        mask = torch.zeros((batch_size, time_steps, streams), dtype=torch.bool, device=features.device)
        for s in range(streams):
            mask[:, :, s] = self._generate_mask(time_steps, batch_size, features.device)

        masked = features.clone()
        for s in range(streams):
            mask_indices = mask[:, :, s].unsqueeze(-1)
            masked[:, :, s, :] = torch.where(
                mask_indices,
                self.mask_embeddings[s].view(1, 1, -1),
                masked[:, :, s, :],
            )
        return masked, mask

    def _generate_mask(self, time_steps: int, batch_size: int, device: torch.device) -> torch.Tensor:
        span = self.config.span_length
        num_spans = max(1, int(time_steps * self.config.mask_prob / span))
        mask = torch.zeros((batch_size, time_steps), dtype=torch.bool, device=device)
        for b in range(batch_size):
            start_positions = torch.randint(0, max(1, time_steps - span + 1), (num_spans,), device=device)
            for start in start_positions:
                end = min(time_steps, start + span)
                mask[b, start:end] = True
        return mask
