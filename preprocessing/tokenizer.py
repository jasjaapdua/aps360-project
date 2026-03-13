"""Word-level tokenizer utilities used by model training and generation."""

from __future__ import annotations

import re

from preprocessing.build_vocabulary import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN


_WORD_RE = re.compile(r"[a-z0-9']+|[.,!?-]")


class LyricsTokenizer:
    """Convert lyric text between tokens, token ids, and decoded strings.

    The tokenizer uses a simple regex that captures:
    - alphanumeric words plus apostrophes (`don't`),
    - select punctuation tokens (`. , ! ? -`).
    """

    def __init__(self, stoi: dict[str, int]):
        """Initialize the tokenizer from a vocabulary mapping.

        Args:
            stoi: Token-to-index mapping that must include `<unk>`, `<bos>`, and `<eos>`.
        """
        self.stoi = stoi
        self.itos = {idx: token for token, idx in stoi.items()}

        self.unk_id = self.stoi[UNK_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]

    def tokenize(self, text: str) -> list[str]:
        """Tokenize raw text into lowercase word/punctuation tokens.

        Args:
            text: Input text.

        Returns:
            List of token strings.
        """
        return _WORD_RE.findall(text.lower())

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Tokenize and map text to token ids.

        Args:
            text: Raw input text.
            add_special_tokens: Whether to wrap output with `<bos>` and `<eos>`.

        Returns:
            List of integer token ids.
        """
        ids = [self.stoi.get(tok, self.unk_id) for tok in self.tokenize(text)]
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def encode_tokens(self, tokens: list[str], add_special_tokens: bool = True) -> list[int]:
        """Map a pre-tokenized list of tokens to ids.

        Args:
            tokens: Tokenized text (already split into token strings).
            add_special_tokens: Whether to add sequence boundary tokens.

        Returns:
            List of integer token ids.
        """
        ids = [self.stoi.get(tok, self.unk_id) for tok in tokens]
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert token ids back into a readable lyric string.

        Args:
            ids: Sequence of token ids.
            skip_special_tokens: If `True`, hide tokens like `<bos>` and `<eos>`.

        Returns:
            Decoded text with simple punctuation spacing cleanup.
        """
        tokens: list[str] = []
        for idx in ids:
            tok = self.itos.get(int(idx), UNK_TOKEN)
            if skip_special_tokens and tok.startswith("<") and tok.endswith(">"):
                continue
            tokens.append(tok)

        text = " ".join(tokens)
        for punct in [".", ",", "!", "?", "-"]:
            text = text.replace(f" {punct}", punct)
        return text.strip()
