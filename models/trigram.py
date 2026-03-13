"""Trigram baseline model for lightweight lyric generation.

This baseline is useful for quick sanity checks before training the LSTM.
It models `P(next | previous_two_tokens)` using empirical counts and random
sampling from observed continuations.
"""

from __future__ import annotations

import random
from collections import defaultdict

from preprocessing.build_vocabulary import BOS_TOKEN, EOS_TOKEN


def train_trigram(corpus: list[list[str]]) -> dict[tuple[str, str], list[str]]:
    """Build a trigram continuation table from tokenized lines.

    Args:
        corpus: Tokenized lyric lines.

    Returns:
        Mapping from token bigram `(w1, w2)` to a list of observed next tokens.
        Repetitions are preserved so random sampling approximates frequency.
    """
    table: dict[tuple[str, str], list[str]] = defaultdict(list)

    for line in corpus:
        seq = [BOS_TOKEN, BOS_TOKEN, *line, EOS_TOKEN]
        for i in range(len(seq) - 2):
            key = (seq[i], seq[i + 1])
            nxt = seq[i + 2]
            table[key].append(nxt)

    return dict(table)


def generate_trigram(
    model: dict[tuple[str, str], list[str]],
    prompt: str = "",
    max_tokens: int = 30,
) -> str:
    """Generate text from a trigram table.

    Args:
        model: Trigram continuation mapping created by :func:`train_trigram`.
        prompt: Optional seed text. The final two prompt tokens define context.
        max_tokens: Maximum number of generated tokens.

    Returns:
        Generated text string.
    """
    seed = [tok for tok in prompt.lower().split() if tok]
    if len(seed) >= 2:
        w1, w2 = seed[-2], seed[-1]
        generated = seed.copy()
    elif len(seed) == 1:
        w1, w2 = BOS_TOKEN, seed[-1]
        generated = seed.copy()
    else:
        w1, w2 = BOS_TOKEN, BOS_TOKEN
        generated: list[str] = []

    for _ in range(max_tokens):
        choices = model.get((w1, w2))
        if not choices:
            break
        nxt = random.choice(choices)
        if nxt == EOS_TOKEN:
            break
        generated.append(nxt)
        w1, w2 = w2, nxt

    return " ".join(generated).strip()


def evaluate_trigram_accuracy(
    model: dict[tuple[str, str], list[str]],
    corpus: list[list[str]],
) -> float:
    """Evaluate trigram next-token accuracy on a tokenized validation corpus.

    Prediction uses majority vote: for each bigram context `(w1, w2)`, the model
    predicts the most frequent observed continuation in training data.

    Args:
        model: Trigram continuation table from :func:`train_trigram`.
        corpus: Tokenized validation lines.

    Returns:
        Fraction of correctly predicted next tokens in `[0.0, 1.0]`.
        Returns `0.0` when no comparable trigram contexts are available.
    """
    total = 0
    correct = 0

    for line in corpus:
        seq = [BOS_TOKEN, BOS_TOKEN, *line, EOS_TOKEN]
        for i in range(len(seq) - 2):
            context = (seq[i], seq[i + 1])
            target = seq[i + 2]
            candidates = model.get(context)
            if not candidates:
                continue
            prediction = max(set(candidates), key=candidates.count)
            total += 1
            if prediction == target:
                correct += 1

    if total == 0:
        return 0.0
    return correct / total
