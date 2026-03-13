"""
Download lyric datasets using HuggingFace datasets.
"""

from __future__ import annotations

from src.data.dataset_loader import load_lyrics_dataset
from src.utils.file_utils import load_yaml


def main() -> None:
    cfg = load_yaml("configs/dataset_config.yaml")
    name = cfg.get("huggingface_dataset")
    load_lyrics_dataset(name)
    print(f"Dataset '{name}' downloaded to data/raw")


if __name__ == "__main__":
    main()
