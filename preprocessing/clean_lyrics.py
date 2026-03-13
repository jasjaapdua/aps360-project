"""Utilities for loading, normalizing, and token-splitting lyric text.

The preprocessing layer keeps the project intentionally simple:
- Input is plain `.txt` lyric files.
- Cleaning is regex-based and language-agnostic for English-style lyrics.
- Output is a corpus represented as `list[list[str]]` where each inner list is one line.

This format is shared by the trigram baseline and the LSTM training pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

_ALLOWED = re.compile(r"[^a-z0-9\s'.,!?-]+")
_MULTI_WS = re.compile(r"\s+")


def clean_text(text: str, lowercase: bool = True) -> str:
    """Normalize lyric text while preserving line boundaries.

    Args:
        text: Raw text that may include mixed casing, special symbols, and extra spaces.
        lowercase: Whether to lowercase the full input before cleanup.

    Returns:
        A cleaned string where:
        - unsupported characters are removed,
        - repeated whitespace is collapsed,
        - empty lines are dropped,
        - non-empty lines are joined with newline characters.
    """
    if lowercase:
        text = text.lower()

    lines = []
    for raw_line in text.splitlines():
        line = _ALLOWED.sub(" ", raw_line)
        line = _MULTI_WS.sub(" ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def read_lyrics_files(paths: Iterable[str | Path]) -> str:
    """Read and concatenate lyric text from one or more `.txt` files.

    Args:
        paths: Candidate file paths. Non-existent paths, directories, and
            non-`.txt` files are ignored.

    Returns:
        A single string containing all file contents separated by newlines.
        Returns an empty string when no readable text files are found.
    """
    chunks: List[str] = []
    for path in paths:
        p = Path(path)
        if p.is_file() and p.suffix.lower() == ".txt":
            chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def load_corpus(paths: Iterable[str | Path]) -> list[list[str]]:
    """Load lyric files and return a tokenized line-by-line corpus.

    This is the main entrypoint used by training code. It combines:
    file reading -> text cleaning -> line splitting -> whitespace tokenization.

    Args:
        paths: Paths to lyric files.

    Returns:
        A list of token lists. Each element corresponds to one cleaned lyric line.
        Lines with fewer than two tokens are dropped to avoid degenerate examples.
    """
    raw_text = read_lyrics_files(paths)
    cleaned = clean_text(raw_text)

    corpus: list[list[str]] = []
    for line in cleaned.splitlines():
        tokens = line.split()
        if len(tokens) >= 2:
            corpus.append(tokens)
    return corpus
