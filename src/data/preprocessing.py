"""
preprocessing.py

Dataset preprocessing pipeline for lyric data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from src.data.tokenizer import tokenize
from src.utils.text_utils import build_vocabulary, clean_lyrics, create_training_sequences, encode_sequence


@dataclass
class PreprocessConfig:
    text_column: str
    artist_column: str | None
    language: str | None
    language_column: str = "language"
    min_tokens: int
    val_split: float
    test_split: float
    max_vocab_size: int
    pad_token: str
    unk_token: str
    bos_token: str
    eos_token: str
    lowercase: bool = True
    remove_markup: bool = True


@dataclass
class PreprocessArtifacts:
    vocab: List[str]
    token_to_id: dict[str, int]


def filter_language(dataset: Dataset, cfg: PreprocessConfig) -> Dataset:
    """Filter dataset to the requested language if the column exists."""
    if cfg.language and cfg.language_column in dataset.column_names:
        dataset = dataset.filter(lambda ex: ex[cfg.language_column] == cfg.language)
    return dataset


def clean_and_tokenize(dataset: Dataset, cfg: PreprocessConfig) -> Dataset:
    """Apply text cleaning and tokenization."""

    def _process(example: dict) -> dict:
        text = clean_lyrics(example[cfg.text_column], lowercase=cfg.lowercase, remove_markup=cfg.remove_markup)
        tokens = tokenize(text)
        example["tokens"] = tokens
        example["num_tokens"] = len(tokens)
        return example

    dataset = dataset.map(_process, remove_columns=[col for col in [cfg.text_column] if col in dataset.column_names])
    dataset = dataset.filter(lambda ex: ex["num_tokens"] >= cfg.min_tokens)
    return dataset


def split_dataset(dataset: Dataset, cfg: PreprocessConfig) -> DatasetDict:
    """Split dataset at the song level into train/val/test."""
    train_val, test = train_test_split(dataset, test_size=cfg.test_split, shuffle=True, random_state=42)
    train, val = train_test_split(train_val, test_size=cfg.val_split / (1 - cfg.test_split), shuffle=True, random_state=42)
    return DatasetDict({"train": Dataset.from_dict(train), "validation": Dataset.from_dict(val), "test": Dataset.from_dict(test)})


def build_vocab_from_dataset(dataset: DatasetDict, cfg: PreprocessConfig) -> PreprocessArtifacts:
    """Construct vocabulary from training split."""
    specials = [cfg.pad_token, cfg.unk_token, cfg.bos_token, cfg.eos_token]
    vocab = build_vocabulary(dataset["train"]["tokens"], max_size=cfg.max_vocab_size, specials=specials)
    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
    return PreprocessArtifacts(vocab=vocab, token_to_id=token_to_id)


def encode_dataset(dataset: DatasetDict, artifacts: PreprocessArtifacts, cfg: PreprocessConfig) -> DatasetDict:
    """Encode tokens to integer ids and build training sequences."""

    def _encode(example: dict) -> dict:
        encoded = encode_sequence(example["tokens"], artifacts.token_to_id, cfg.bos_token, cfg.eos_token)
        example["input_ids"] = encoded
        return example

    encoded_ds = dataset.map(_encode)
    return encoded_ds


def prepare_training_sequences(dataset: Dataset, seq_len: int) -> List[Tuple[List[int], int]]:
    """Generate (context, target) pairs for LM training from encoded ids."""
    sequences: List[Tuple[List[int], int]] = []
    for encoded in dataset["input_ids"]:
        sequences.extend(create_training_sequences(encoded, seq_len))
    return sequences
