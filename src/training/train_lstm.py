"""
train_lstm.py

Model-specific training loop for the LSTM lyric generator.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.dataset import LanguageModelingDataset, collate_batch
from src.models.lstm_model import LyricsLSTM
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def train_lstm_model(train_sequences, val_sequences, vocab_size: int, pad_token_id: int, cfg: dict) -> Trainer:
    """Configure and train the LSTM language model.

    Args:
        train_sequences: List of (context, target) pairs for training.
        val_sequences: List of (context, target) pairs for validation.
        vocab_size: Size of vocabulary.
        pad_token_id: Padding token id for collate.
        cfg: Training configuration dictionary.

    Returns:
        Trainer instance after training.
    """
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cpu"))

    train_ds = LanguageModelingDataset(train_sequences)
    val_ds = LanguageModelingDataset(val_sequences)
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 32), shuffle=True, collate_fn=lambda b: collate_batch(b, pad_token_id))
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 32), shuffle=False, collate_fn=lambda b: collate_batch(b, pad_token_id))

    model = LyricsLSTM(
        vocab_size=vocab_size,
        embedding_dim=cfg.get("embedding_dim", 256),
        hidden_dim=cfg.get("hidden_dim", 512),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.3),
        pad_token_id=pad_token_id,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("learning_rate", 1e-3), weight_decay=cfg.get("weight_decay", 0.0))

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_grad_norm=cfg.get("max_grad_norm", 5.0),
        checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints"),
        early_stopping_patience=cfg.get("patience", 3),
        early_stopping_delta=cfg.get("min_delta", 0.0),
    )

    trainer.fit(cfg.get("num_epochs", 10))
    return trainer
