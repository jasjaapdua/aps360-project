"""
metrics.py

Evaluation metrics for lyric generation models.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, List

import sacrebleu


def perplexity(loss: float) -> float:
    """Compute perplexity from average cross-entropy loss."""
    return float(math.exp(loss))


def bleu_score(predictions: List[str], references: List[str]) -> float:
    """Compute corpus BLEU using SacreBLEU."""
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return float(bleu.score)


def distinct_n(sequences: Iterable[str], n: int = 2) -> float:
    """Compute distinct-n diversity metric."""
    total_ngrams = 0
    unique_ngrams = set()
    for seq in sequences:
        tokens = seq.split()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            unique_ngrams.add(ngram)
            total_ngrams += 1
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams
