"""SHuBERT model implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass
class SHuBERTConfig:
    input_dim: int = 1024
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ff_dim: int = 3072
    dropout: float = 0.1


class SHuBERTModel(nn.Module):
    """Transformer-based SHuBERT model for multi-stream SSL."""

    def __init__(self, config: SHuBERTConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.input_dim, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.dropout = nn.Dropout(config.dropout)

        self.heads = nn.ModuleDict(
            {
                "face": nn.Linear(config.hidden_dim, 256),
                "left": nn.Linear(config.hidden_dim, 256),
                "right": nn.Linear(config.hidden_dim, 256),
                "body": nn.Linear(config.hidden_dim, 256),
            }
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            features: [B, T, 4, 256] tensor.

        Returns:
            contextual_embeddings: [B, T, 768].
            logits: Dict of stream logits [B, T, 256].
        """
        batch, time_steps, streams, dim = features.shape
        flattened = features.reshape(batch, time_steps, streams * dim)
        projected = self.projection(flattened)
        projected = self.dropout(projected)
        encoded = self.encoder(projected)
        logits = {key: head(encoded) for key, head in self.heads.items()}
        return encoded, logits

    def freeze_layers(self, num_layers: int) -> None:
        """Freeze the first N transformer layers."""
        for idx, layer in enumerate(self.encoder.layers):
            if idx < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
