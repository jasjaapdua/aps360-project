"""
transformer_model.py

Lightweight GPT-style Transformer encoder-decoder for lyric generation.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TransformerLanguageModel(nn.Module):
    """Transformer-based language model using nn.TransformerEncoder."""

    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1, pad_token_id: int = 0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        encoded = self.encoder(x, mask=attn_mask)
        logits = self.fc_out(encoded)
        return logits
