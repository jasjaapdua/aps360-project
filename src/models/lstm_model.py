"""
lstm_model.py

Defines the LSTM-based language model used for lyric generation.

This model learns to predict the next token in a lyric sequence.
Architecture:

Embedding -> LSTM -> Linear -> Softmax

Used during training and generation.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class LyricsLSTM(nn.Module):
    """Two-layer LSTM language model for next-token prediction."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3, pad_token_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embeds = self.embedding(input_ids)
        outputs, hidden = self.lstm(embeds, hidden)
        outputs = self.dropout(outputs)
        logits = self.fc(outputs)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states with zeros."""
        weight = next(self.parameters())
        num_layers = self.lstm.num_layers
        hidden_size = self.lstm.hidden_size
        h0 = weight.new_zeros((num_layers, batch_size, hidden_size), device=device)
        c0 = weight.new_zeros((num_layers, batch_size, hidden_size), device=device)
        return h0, c0
