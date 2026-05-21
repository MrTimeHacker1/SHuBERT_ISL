"""Training loop for SHuBERT SSL adaptation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.masking import StreamMasker
from training.losses import ssl_loss
from training.callbacks import EarlyStopping, save_checkpoint
from utils.logger import setup_logger


logger = setup_logger(__name__)


def default_collate(batch: list[Tuple[np.ndarray, dict]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    features = torch.tensor([item[0] for item in batch], dtype=torch.float32)
    labels = {
        "face": torch.tensor([item[1]["face"] for item in batch], dtype=torch.long),
        "left": torch.tensor([item[1]["left"] for item in batch], dtype=torch.long),
        "right": torch.tensor([item[1]["right"] for item in batch], dtype=torch.long),
        "body": torch.tensor([item[1]["body"] for item in batch], dtype=torch.long),
    }
    return features, labels


class Trainer:
    """Trainer for self-supervised SHuBERT adaptation."""

    def __init__(
        self,
        model: nn.Module,
        masker: StreamMasker,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_dir: Path,
        checkpoint_dir: Path,
        mixed_precision: bool,
        grad_accum_steps: int,
        early_stopping_patience: int,
    ) -> None:
        self.model = model
        self.masker = masker
        self.optimizer = optimizer
        self.device = device
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.checkpoint_dir = checkpoint_dir
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        self.grad_accum_steps = grad_accum_steps
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)

    def train(self, dataloader: DataLoader, epochs: int) -> None:
        global_step = 0
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            self.model.train()
            for step, (features, labels) in enumerate(dataloader):
                features = features.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}

                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    masked_features, mask = self.masker(features)
                    _, logits = self.model(masked_features)
                    loss = ssl_loss(logits, labels, mask) / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item() * self.grad_accum_steps
                if global_step % 10 == 0:
                    self.writer.add_scalar("train/loss", loss.item() * self.grad_accum_steps, global_step)
                global_step += 1

            avg_loss = epoch_loss / max(1, len(dataloader))
            logger.info("Epoch %d | loss=%.4f", epoch, avg_loss)
            self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)

            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                avg_loss,
                self.checkpoint_dir / f"epoch_{epoch}.pt",
            )

            if self.early_stopping.step(avg_loss):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        self.writer.close()
