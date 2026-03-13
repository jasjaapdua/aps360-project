"""
train_transformer.py

Training helper for Transformer-based lyric model.
"""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.dataset import LanguageModelingDataset, collate_batch
from src.models.transformer_model import TransformerLanguageModel
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def train_transformer_model(train_sequences, val_sequences, vocab_size: int, pad_token_id: int, cfg: dict) -> Trainer:
    """Train a Transformer language model with given hyperparameters."""
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cpu"))

    train_ds = LanguageModelingDataset(train_sequences)
    val_ds = LanguageModelingDataset(val_sequences)
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 16), shuffle=True, collate_fn=lambda b: collate_batch(b, pad_token_id))
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 16), shuffle=False, collate_fn=lambda b: collate_batch(b, pad_token_id))

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 8),
        num_layers=cfg.get("num_layers", 4),
        dim_feedforward=cfg.get("dim_feedforward", 1024),
        dropout=cfg.get("dropout", 0.1),
        pad_token_id=pad_token_id,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 5e-4))

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints"),
        early_stopping_patience=cfg.get("patience", 2),
        early_stopping_delta=cfg.get("min_delta", 0.0),
    )

    trainer.fit(cfg.get("num_epochs", 10))
    return trainer
