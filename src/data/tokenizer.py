"""
tokenizer.py

Tokenizer wrapper for lyric text.
"""

from __future__ import annotations

import nltk
from typing import List

# Ensure NLTK punkt tokenizer is available at runtime
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def tokenize(text: str) -> List[str]:
    """Tokenize a lyric string into word tokens.

    Args:
        text: Cleaned lyric text.

    Returns:
        List of token strings.
    """
    return nltk.word_tokenize(text)
