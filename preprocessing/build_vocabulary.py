"""Vocabulary construction and persistence helpers.

The vocabulary is word-level and includes reserved special tokens:
- `<pad>` for sequence padding (future extension, not currently used in batching),
- `<unk>` for out-of-vocabulary tokens,
- `<bos>` and `<eos>` for sequence boundaries.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def build_vocabulary(
    corpus: Iterable[Iterable[str]],
    min_freq: int = 1,
    max_size: int | None = None,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build token-to-id and id-to-token mappings from a tokenized corpus.

    Args:
        corpus: Iterable of token iterables (for example `list[list[str]]`).
        min_freq: Minimum count for a token to be included.
        max_size: Maximum total vocabulary size including special tokens.
            If `None`, no size limit is applied.

    Returns:
        A tuple `(stoi, itos)` where:
        - `stoi` maps token strings to integer ids,
        - `itos` maps integer ids back to token strings.

    Notes:
        Tokens are sorted by descending frequency, then alphabetically to make
        vocabulary generation deterministic.
    """
    counter = Counter(token for line in corpus for token in line)

    items = [item for item in counter.items() if item[1] >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))

    if max_size is not None and max_size > len(SPECIAL_TOKENS):
        max_main = max_size - len(SPECIAL_TOKENS)
        items = items[:max_main]

    itos = list(SPECIAL_TOKENS)
    itos.extend(token for token, _ in items)
    stoi = {token: idx for idx, token in enumerate(itos)}

    return stoi, {idx: token for idx, token in enumerate(itos)}


def save_vocabulary(stoi: dict[str, int], path: str | Path) -> None:
    """Persist a string-to-index vocabulary mapping as JSON.

    Args:
        stoi: Token-to-index mapping.
        path: Destination file path. Parent directories are created if needed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stoi, indent=2, sort_keys=True), encoding="utf-8")


def load_vocabulary(path: str | Path) -> dict[str, int]:
    """Load a previously saved vocabulary JSON file.

    Args:
        path: Path to a JSON file produced by :func:`save_vocabulary`.

    Returns:
        The token-to-index mapping.
    """
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))
