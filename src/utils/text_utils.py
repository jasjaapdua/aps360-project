"""
text_utils.py

Text preprocessing helpers for lyric data.
"""

from __future__ import annotations

import re
from typing import Iterable, List

SECTION_PATTERN = re.compile(r"\[(verse|chorus|bridge|intro|outro|hook|pre-chorus)[^\]]*\]", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_lyrics(text: str, lowercase: bool = True, remove_markup: bool = True) -> str:
    """Clean raw lyric text.

    Removes section markers, punctuation noise, and normalizes whitespace.

    Args:
        text: Raw lyric string.
        lowercase: Whether to lowercase the text.
        remove_markup: Whether to strip section markers like ``[chorus]``.

    Returns:
        Cleaned lyric string.
    """
    if remove_markup:
        text = SECTION_PATTERN.sub(" ", text)
    text = text.replace("\r", " ")
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text.lower() if lowercase else text


def build_vocabulary(sequences: Iterable[List[str]], max_size: int, specials: List[str]) -> List[str]:
    """Build a vocabulary list ordered by frequency.

    Args:
        sequences: Iterable of token lists.
        max_size: Maximum vocabulary size.
        specials: Special tokens to prepend to the vocabulary.

    Returns:
        Ordered list of vocabulary tokens including specials.
    """
    freq: dict[str, int] = {}
    for seq in sequences:
        for tok in seq:
            freq[tok] = freq.get(tok, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    vocab = specials + [tok for tok, _ in sorted_tokens if tok not in specials][: max_size - len(specials)]
    return vocab


def encode_sequence(tokens: List[str], token_to_id: dict[str, int], bos_token: str | None = None, eos_token: str | None = None) -> List[int]:
    """Encode tokens into ids with optional BOS/EOS wrappers."""
    encoded = []
    if bos_token:
        encoded.append(token_to_id[bos_token])
    encoded.extend(token_to_id.get(tok, token_to_id.get("<unk>", 0)) for tok in tokens)
    if eos_token:
        encoded.append(token_to_id[eos_token])
    return encoded


def create_training_sequences(encoded: List[int], seq_len: int) -> List[tuple[list[int], int]]:
    """Slice encoded ids into input/target pairs for language modeling."""
    sequences: List[tuple[list[int], int]] = []
    for i in range(len(encoded) - seq_len):
        context = encoded[i : i + seq_len]
        target = encoded[i + seq_len]
        sequences.append((context, target))
    return sequences
