"""
Train trigram baseline model.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk

from src.models.ngram_model import TrigramLanguageModel


def main() -> None:
    processed_path = Path("data/processed/encoded")
    if not processed_path.exists():
        raise FileNotFoundError("Run scripts/preprocess_data.py first to create processed data.")

    ds = load_from_disk(processed_path)
    vocab = json.loads(Path("data/processed/vocab.json").read_text())['vocab']
    model = TrigramLanguageModel(vocab=vocab)
    model.fit(ds['train']["tokens"])
    Path("checkpoints").mkdir(exist_ok=True)
    out_path = Path("checkpoints/trigram_model.json")
    out_path.write_text(json.dumps({"vocab": vocab, "trigram_counts": {"|".join(k): v for k, v in model.trigram_counts.items()}}, indent=2))
    print(f"Saved baseline model to {out_path}")


if __name__ == "__main__":
    main()
