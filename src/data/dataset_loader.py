"""
dataset_loader.py

Utilities for downloading and caching lyric datasets using HuggingFace datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from datasets import load_dataset, DatasetDict


def load_lyrics_dataset(name: str, split: Optional[str] = None, cache_dir: str | Path = "data/raw") -> DatasetDict:
    """Load a HuggingFace dataset by name.

    Args:
        name: HuggingFace dataset identifier, e.g., ``"spotify/podbaby-lyrics"``.
        split: Specific split to load. If None, full DatasetDict is returned.
        cache_dir: Local cache directory for downloaded data.

    Returns:
        DatasetDict or Dataset for the requested split.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(name, cache_dir=str(cache_path))
    return dataset[split] if split else dataset
