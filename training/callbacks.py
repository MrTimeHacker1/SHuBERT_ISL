"""Training callbacks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class EarlyStopping:
    patience: int
    best_loss: float = float("inf")
    counter: int = 0

    def step(self, loss: float) -> bool:
        """Update early stopping state.

        Returns:
            True if training should stop.
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
) -> Optional[int]:
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    optimizer.load_state_dict(state["optimizer"])
    return int(state.get("epoch", 0))
