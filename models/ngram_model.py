"""
ngram_model.py

Simple n-gram language model baseline.
"""

import random
from collections import defaultdict


class NGramLanguageModel:
    """
    N-gram language model for baseline lyric generation.
    """

    def __init__(self, n=2):
        self.n = n
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)

    def train(self, token_sequences):
        """
        Train the model on tokenized sequences.
        """

        for seq in token_sequences:

            if len(seq) < self.n:
                continue

            for i in range(len(seq) - self.n + 1):

                context = tuple(seq[i:i+self.n-1])
                target = seq[i+self.n-1]

                self.ngram_counts[context][target] += 1
                self.context_counts[context] += 1

    def predict_next(self, context):
        """
        Sample next token given context.
        """

        context = tuple(context)

        if context not in self.ngram_counts:
            return None

        targets = self.ngram_counts[context]

        words = list(targets.keys())
        counts = list(targets.values())

        return random.choices(words, weights=counts)[0]

    def generate(self, seed_tokens, max_length=50):
        """
        Generate token sequence from seed.
        """

        tokens = seed_tokens.copy()

        for _ in range(max_length):

            context = tokens[-(self.n-1):]

            next_token = self.predict_next(context)

            if next_token is None:
                break

            tokens.append(next_token)

        return tokens
