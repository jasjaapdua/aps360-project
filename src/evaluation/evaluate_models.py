"""
evaluate_models.py

Helper to compute evaluation metrics for generated lyrics.
"""

from __future__ import annotations

from typing import List, Tuple

from src.evaluation.metrics import bleu_score, distinct_n, perplexity


def evaluate_generation(predictions: List[str], references: List[str]) -> dict:
    """Compute BLEU and diversity metrics for generated samples."""
    return {
        "bleu": bleu_score(predictions, references),
        "distinct_1": distinct_n(predictions, n=1),
        "distinct_2": distinct_n(predictions, n=2),
    }


def summarize_losses(train_loss: float, val_loss: float) -> dict:
    """Return perplexities for training and validation losses."""
    return {"train_perplexity": perplexity(train_loss), "val_perplexity": perplexity(val_loss)}
