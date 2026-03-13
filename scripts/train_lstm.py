"""
Train the LSTM lyric generator from processed data.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk

from src.data.preprocessing import prepare_training_sequences
from src.training.train_lstm import train_lstm_model
from src.utils.file_utils import load_yaml


def main() -> None:
    dataset_cfg = load_yaml("configs/dataset_config.yaml")
    train_cfg = load_yaml("configs/training_config.yaml")
    model_cfg = load_yaml("configs/model_config.yaml")

    processed_path = Path("data/processed/encoded")
    if not processed_path.exists():
        raise FileNotFoundError("Processed data not found. Run scripts/preprocess_data.py first.")

    ds = load_from_disk(processed_path)
    vocab = json.loads(Path("data/processed/vocab.json").read_text())["vocab"]
    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}

    seq_len = model_cfg.get("max_seq_len", 50)
    train_sequences = prepare_training_sequences(ds["train"], seq_len)
    val_sequences = prepare_training_sequences(ds["validation"], seq_len)

    trainer = train_lstm_model(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        vocab_size=len(vocab),
        pad_token_id=token_to_id[dataset_cfg["pad_token"]],
        cfg={**train_cfg, **model_cfg},
    )

    print("Training finished. Best model stored in checkpoints directory.")


if __name__ == "__main__":
    main()
