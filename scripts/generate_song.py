"""
Generate lyrics from a trained model checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.generation.generate_lyrics import generate
from src.models.lstm_model import LyricsLSTM


def load_model(checkpoint_path: Path, vocab: list[str], device: torch.device) -> LyricsLSTM:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint["model_state"] if isinstance(checkpoint, dict) else checkpoint
    model = LyricsLSTM(vocab_size=len(vocab), embedding_dim=256, hidden_dim=512)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lyrics from a trained model.")
    parser.add_argument("--prompt", type=str, default="love is a fire", help="Prompt text")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model_best_epoch1.pt"), help="Path to checkpoint")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    vocab = json.loads(Path("data/processed/vocab.json").read_text())["vocab"]
    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, vocab, device)
    text = generate(model, token_to_id, vocab, args.prompt, max_length=args.max_length, temperature=args.temperature, top_k=args.top_k, device=device)
    print("\n" + text)


if __name__ == "__main__":
    main()
