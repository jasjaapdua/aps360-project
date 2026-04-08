"""
tokenizer.py

Word-level tokenizer with a lightweight local vocabulary.
"""

import re
import json


class Tokenizer:
    """
    Class-based tokenizer for lyric generation.
    """

    def __init__(self):
        self.vocab = {"<unk>": 0}
        self.id_to_token = ["<unk>"]

    def _tokenize(self, text):
        if not isinstance(text, str):
            return []
        # Mirrors simple English tokenization without external binary deps.
        return re.findall(r"[a-zA-Z0-9']+", text.lower())

    def build_vocab(self, texts):
        """
        Build vocabulary from list of lyric strings.
        """
        for text in texts:
            for token in self._tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = len(self.id_to_token)
                    self.id_to_token.append(token)

    def encode(self, text):
        """
        Convert text → token IDs
        """
        return [self.vocab.get(token, 0) for token in self._tokenize(text)]

    def encode_batch(self, texts):
        """
        Encode list of texts.
        """
        return [self.encode(text) for text in texts]

    def decode(self, token_ids):
        """
        Convert token IDs → text
        """
        words = [
            self.id_to_token[i] if 0 <= i < len(self.id_to_token) else "<unk>"
            for i in token_ids
        ]
        return " ".join(words)

    def save(self, path):
        """
        Persist tokenizer vocabulary.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"id_to_token": self.id_to_token}, f)

    def load(self, path):
        """
        Load tokenizer vocabulary.
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.id_to_token = payload["id_to_token"]
        self.vocab = {token: idx for idx, token in enumerate(self.id_to_token)}

    @property
    def vocab_size(self):
        return len(self.id_to_token)
