"""
Preprocess lyric dataset and store tokenized splits.
"""

from __future__ import annotations

from pathlib import Path

from datasets import DatasetDict

from src.data.dataset_loader import load_lyrics_dataset
from src.data.preprocessing import PreprocessArtifacts, PreprocessConfig, build_vocab_from_dataset, clean_and_tokenize, encode_dataset, filter_language, split_dataset
from src.utils.file_utils import ensure_dir, load_yaml, save_json


PROCESSED_DIR = Path("data/processed")


def main() -> None:
    cfg = load_yaml("configs/dataset_config.yaml")
    preprocess_cfg = PreprocessConfig(**cfg)

    raw_ds = load_lyrics_dataset(cfg["huggingface_dataset"])
    dataset = raw_ds
    train_split = dataset["train"] if isinstance(dataset, DatasetDict) else dataset

    filtered = filter_language(train_split, preprocess_cfg)
    tokenized = clean_and_tokenize(filtered, preprocess_cfg)
    splits = split_dataset(tokenized, preprocess_cfg)

    artifacts = build_vocab_from_dataset(splits, preprocess_cfg)
    encoded = encode_dataset(splits, artifacts, preprocess_cfg)

    ensure_dir(PROCESSED_DIR)
    encoded.save_to_disk(str(PROCESSED_DIR / "encoded"))
    save_json({"vocab": artifacts.vocab}, PROCESSED_DIR / "vocab.json")
    print("Preprocessing complete. Encoded dataset stored in data/processed/encoded")


if __name__ == "__main__":
    main()
