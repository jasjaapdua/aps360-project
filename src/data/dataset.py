"""
dataset.py

PyTorch dataset wrappers for lyric modeling.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset):
    """Dataset of fixed-length context-target pairs for language modeling."""

    def __init__(self, sequences: Sequence[Tuple[Sequence[int], int]]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context, target = self.sequences[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function that pads contexts to equal length within a batch."""
    contexts, targets = zip(*batch)
    max_len = max(ctx.size(0) for ctx in contexts)
    padded = torch.full((len(contexts), max_len), pad_token_id, dtype=torch.long)
    for i, ctx in enumerate(contexts):
        padded[i, : ctx.size(0)] = ctx
    return padded, torch.stack(targets)
