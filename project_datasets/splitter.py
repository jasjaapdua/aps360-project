"""
splitter.py

Helpers for deterministic train/val/test splits at the song level.
"""

import random


def split_items(items, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Split tokenized song sequences into train/val/test lists.
    """
    total = len(items)
    if total == 0:
        return [], [], []

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value")

    train_ratio = train_ratio / ratio_sum
    val_ratio = val_ratio / ratio_sum
    test_ratio = test_ratio / ratio_sum

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Ensure each split has at least one sample when feasible.
    if total >= 3:
        if not train_idx:
            train_idx.append(test_idx.pop() if test_idx else val_idx.pop())
        if not val_idx:
            val_idx.append(test_idx.pop() if test_idx else train_idx.pop())
        if not test_idx:
            test_idx.append(val_idx.pop() if val_idx else train_idx.pop())

    train = [items[i] for i in train_idx]
    val = [items[i] for i in val_idx]
    test = [items[i] for i in test_idx]

    return train, val, test


def split_token_sequences(token_sequences, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Backward-compatible wrapper.
    """
    return split_items(token_sequences, train_ratio, val_ratio, test_ratio, seed=seed)
