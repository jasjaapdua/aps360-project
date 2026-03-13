"""
trainer.py

Reusable training utilities for language models.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.file_utils import ensure_dir


class Trainer:
    """Generic trainer handling train/validation loops and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: float = 5.0,
        checkpoint_dir: str | Path = "checkpoints",
        early_stopping_patience: int = 3,
        early_stopping_delta: float = 0.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta

        self.model.to(device)
        self.best_val_loss = math.inf
        self.no_improve_epochs = 0

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="train", leave=False):
            inputs, targets = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            logits, _ = self.model(inputs)
            logits_last = logits[:, -1, :]
            loss = self.criterion(logits_last, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
        if self.scheduler:
            self.scheduler.step()
        return total_loss / len(self.train_loader)

    def validate_epoch(self) -> float:
        if self.val_loader is None:
            return math.inf
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="val", leave=False):
                inputs, targets = [b.to(self.device) for b in batch]
                logits, _ = self.model(inputs)
                logits_last = logits[:, -1, :]
                loss = self.criterion(logits_last, targets)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "best") -> Path:
        path = self.checkpoint_dir / f"model_{tag}_epoch{epoch}.pt"
        torch.save({"epoch": epoch, "model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict(), "val_loss": val_loss}, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint.get("optimizer_state", {}))

    def fit(self, num_epochs: int) -> Dict[str, float]:
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss + self.early_stopping_delta < self.best_val_loss:
                self.best_val_loss = val_loss
                self.no_improve_epochs = 0
                self.save_checkpoint(epoch, val_loss, tag="best")
            else:
                self.no_improve_epochs += 1

            if self.no_improve_epochs >= self.early_stopping_patience:
                break
        return history
