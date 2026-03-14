"""
tokenizer.py

Word-level tokenizer using torchtext utilities.
"""

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class Tokenizer:
    """
    Class-based tokenizer for lyric generation.
    """

    def __init__(self):
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None

    def _yield_tokens(self, texts):
        """
        Generator used to build vocabulary.
        """
        for text in texts:
            yield self.tokenizer(text)

    def build_vocab(self, texts):
        """
        Build vocabulary from list of lyric strings.
        """
        self.vocab = build_vocab_from_iterator(
            self._yield_tokens(texts), specials=["<unk>"]
        )

        self.vocab.set_default_index(self.vocab["<unk>"])

    def encode(self, text):
        """
        Convert text → token IDs
        """
        tokens = self.tokenizer(text)
        return self.vocab(tokens)

    def encode_batch(self, texts):
        """
        Encode list of texts.
        """
        return [self.encode(text) for text in texts]

    def decode(self, token_ids):
        """
        Convert token IDs → text
        """
        words = [self.vocab.lookup_token(i) for i in token_ids]
        return " ".join(words)

    @property
    def vocab_size(self):
        return len(self.vocab)
