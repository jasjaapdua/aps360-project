"""
ngram_model.py

Frequency-based trigram language model for baseline lyric generation.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple


class TrigramLanguageModel:
    """Simple trigram language model with Laplace smoothing."""

    def __init__(self, vocab: List[str], alpha: float = 1.0):
        self.vocab = vocab
        self.alpha = alpha
        self.trigram_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self.bigram_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.vocab_set = set(vocab)

    def fit(self, corpus: List[List[str]]) -> None:
        """Fit frequency tables from tokenized corpus."""
        for tokens in corpus:
            padded = ["<bos>", "<bos>"] + tokens + ["<eos>"]
            for i in range(len(padded) - 2):
                trigram = (padded[i], padded[i + 1], padded[i + 2])
                bigram = (padded[i], padded[i + 1])
                self.trigram_counts[trigram] += 1
                self.bigram_counts[bigram] += 1

    def predict_next_word(self, prev_two: Tuple[str, str]) -> str:
        """Predict next word via maximum likelihood with smoothing."""
        candidates = []
        total = self.bigram_counts.get(prev_two, 0)
        for word in self.vocab:
            trigram = (prev_two[0], prev_two[1], word)
            count = self.trigram_counts.get(trigram, 0)
            prob = (count + self.alpha) / (total + self.alpha * len(self.vocab))
            candidates.append((prob, word))
        candidates.sort(reverse=True)
        return candidates[0][1]

    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        """Sample text from the model using temperature-less greedy decoding."""
        tokens = prompt.split()
        if len(tokens) < 2:
            tokens = ["<bos>", tokens[0] if tokens else "<bos>"]
        for _ in range(max_length):
            prev_two = (tokens[-2], tokens[-1])
            next_word = self.predict_next_word(prev_two)
            if next_word == "<eos>":
                break
            tokens.append(next_word)
        return " ".join(tokens[2:])
