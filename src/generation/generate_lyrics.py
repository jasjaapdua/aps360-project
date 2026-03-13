"""
generate_lyrics.py

Utilities for sampling lyrics from trained models.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import List, Optional


def _sample_logits(logits: torch.Tensor, temperature: float, top_k: Optional[int]) -> int:
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        top_k = max(1, top_k)
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, 1)
        return indices[choice].item()
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate(
    model: nn.Module,
    token_to_id: dict[str, int],
    id_to_token: List[str],
    prompt: str,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: str | torch.device = "cpu",
) -> str:
    """Generate lyrics conditioned on a text prompt."""
    model.eval()
    tokens = prompt.split()
    input_ids = [token_to_id.get(tok, token_to_id.get("<unk>", 0)) for tok in tokens]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        hidden = None
        for _ in range(max_length):
            logits, hidden = model(input_tensor, hidden) if hasattr(model, "init_hidden") else (model(input_tensor), None)
            next_token_logits = logits[:, -1, :].squeeze(0)
            next_id = _sample_logits(next_token_logits, temperature, top_k)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_id]], device=device)], dim=1)
            if id_to_token[next_id] == "<eos>":
                break
    generated_tokens = [id_to_token[idx] for idx in input_tensor[0].tolist()]
    return " ".join(generated_tokens)
