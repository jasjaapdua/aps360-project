"""Dataset ingestion helpers for local, Hugging Face, and Kaggle sources.

This module centralizes dataset loading so the CLI can switch among sources
without changing training code. All loaders return a common corpus format:
`list[list[str]]` where each inner list is a tokenized lyric line.
"""

from __future__ import annotations

import csv
import importlib
import random
import sys
from pathlib import Path
from typing import Iterable

from preprocessing.clean_lyrics import clean_text, load_corpus


def _texts_to_corpus(texts: Iterable[str]) -> list[list[str]]:
    """Convert raw text snippets into the standard tokenized corpus format.

    Args:
        texts: Iterable of raw text samples (each may contain multiple lines).

    Returns:
        Tokenized corpus where each element is a lyric line represented as tokens.
    """
    corpus: list[list[str]] = []
    for text in texts:
        cleaned = clean_text(text)
        for line in cleaned.splitlines():
            tokens = line.split()
            if len(tokens) >= 2:
                corpus.append(tokens)
    return corpus


def _import_hf_datasets_module():
    """Import Hugging Face `datasets` while avoiding local path shadowing.

    Returns:
        Imported `datasets` module from site-packages.

    Raises:
        ImportError: If Hugging Face `datasets` is unavailable.
    """
    cwd = str(Path.cwd().resolve())
    removed_positions: list[tuple[int, str]] = []

    for idx in reversed(range(len(sys.path))):
        entry = sys.path[idx]
        entry_norm = str(Path(entry or ".").resolve())
        if entry in ("", ".") or entry_norm == cwd:
            removed_positions.append((idx, sys.path.pop(idx)))

    try:
        module = importlib.import_module("datasets")
    finally:
        for idx, entry in sorted(removed_positions, key=lambda x: x[0]):
            sys.path.insert(idx, entry)

    if not hasattr(module, "load_dataset"):
        raise ImportError(
            "Imported module named 'datasets' does not expose 'load_dataset'. "
            "Rename the local 'datasets/' folder or verify Hugging Face 'datasets' is installed."
        )
    return module


def load_local_corpus(input_arg: str) -> tuple[list[list[str]], str]:
    """Load corpus from local text files.

    Args:
        input_arg: File path, directory, or glob pattern for `.txt` files.

    Returns:
        Tuple `(corpus, source_info)`.
    """
    path = Path(input_arg)
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(path.glob("*.txt"))
    else:
        parent = path.parent if str(path.parent) != "" else Path(".")
        paths = sorted(parent.glob(path.name))

    if not paths:
        raise FileNotFoundError(f"No .txt files found for input pattern/path: {input_arg}")

    return load_corpus(paths), f"local files ({len(paths)} file(s))"


def load_huggingface_corpus(
    dataset_name: str,
    text_column: str,
    split: str = "train",
    config_name: str | None = None,
    hf_token: str | None = None,
    max_samples: int | None = None,
) -> tuple[list[list[str]], str]:
    """Load a lyric corpus from a Hugging Face dataset.

    Args:
        dataset_name: Dataset identifier, e.g. `neelshah18/lyrics-dataset`.
        text_column: Column containing lyric text.
        split: Dataset split to load.
        config_name: Optional dataset config name/subset.
        hf_token: Optional Hugging Face access token for gated/private datasets.
        max_samples: Optional cap on number of rows consumed.

    Returns:
        Tuple `(corpus, source_info)`.
    """
    hf_datasets = _import_hf_datasets_module()

    dataset_kwargs = {"path": dataset_name, "split": split}
    if config_name:
        dataset_kwargs["name"] = config_name
    if hf_token:
        dataset_kwargs["token"] = hf_token

    try:
        ds = hf_datasets.load_dataset(**dataset_kwargs)
    except RuntimeError as exc:
        message = str(exc)
        if "Dataset scripts are no longer supported" in message:
            raise RuntimeError(
                "The selected Hugging Face dataset requires a dataset script, but your "
                "installed 'datasets' package does not support script-based loading. "
                "Use a file-based dataset (for example, 'neelshah18/song-lyrics-dataset') "
                "or install a compatible version: pip install 'datasets<4'."
            ) from exc
        raise

    texts: list[str] = []
    for row in ds:
        value = row.get(text_column)
        if isinstance(value, str) and value.strip():
            texts.append(value)
            if max_samples is not None and len(texts) >= max_samples:
                break

    if not texts:
        raise ValueError(
            f"No non-empty string values found in column '{text_column}' "
            f"for Hugging Face dataset '{dataset_name}' split '{split}'."
        )

    source_info = f"huggingface:{dataset_name} split={split} rows={len(texts)}"
    return _texts_to_corpus(texts), source_info


def load_kaggle_corpus(
    dataset_ref: str,
    text_column: str,
    file_pattern: str = "*.csv",
    max_samples: int | None = None,
) -> tuple[list[list[str]], str]:
    """Load a lyric corpus from a Kaggle dataset downloaded via `kagglehub`.

    Args:
        dataset_ref: Kaggle dataset reference, e.g. `username/dataset-name`.
        text_column: CSV column containing lyric text.
        file_pattern: Glob pattern for files to parse inside downloaded dataset.
        max_samples: Optional cap on number of rows consumed.

    Returns:
        Tuple `(corpus, source_info)`.
    """
    kagglehub = importlib.import_module("kagglehub")
    dataset_dir = Path(kagglehub.dataset_download(dataset_ref))

    files = sorted(dataset_dir.rglob(file_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching pattern '{file_pattern}' found in Kaggle dataset '{dataset_ref}'."
        )

    texts: list[str] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or text_column not in reader.fieldnames:
                continue
            for row in reader:
                value = row.get(text_column)
                if isinstance(value, str) and value.strip():
                    texts.append(value)
                    if max_samples is not None and len(texts) >= max_samples:
                        break
        if max_samples is not None and len(texts) >= max_samples:
            break

    if not texts:
        raise ValueError(
            f"No non-empty string values found in column '{text_column}' "
            f"for Kaggle dataset '{dataset_ref}'."
        )

    source_info = f"kaggle:{dataset_ref} files={len(files)} rows={len(texts)}"
    return _texts_to_corpus(texts), source_info


def split_corpus(
    corpus: list[list[str]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[list[str]], list[list[str]]]:
    """Split corpus into train/validation sets.

    Args:
        corpus: Full tokenized corpus.
        val_ratio: Fraction of examples for validation in `[0.0, 1.0)`.
        seed: Random seed for deterministic shuffling.

    Returns:
        Tuple `(train_corpus, val_corpus)`.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0).")

    if len(corpus) < 2 or val_ratio == 0.0:
        return corpus, []

    indices = list(range(len(corpus)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = int(len(corpus) * val_ratio)
    val_size = max(1, val_size) if val_ratio > 0 else 0
    val_size = min(val_size, len(corpus) - 1)

    val_idx = set(indices[:val_size])
    train_corpus = [line for i, line in enumerate(corpus) if i not in val_idx]
    val_corpus = [line for i, line in enumerate(corpus) if i in val_idx]
    return train_corpus, val_corpus
