"""
evaluation.py

Held-out and external-data evaluation helpers.
"""

import json
import math
import os
import random
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import config
from training.generate import generate_text


def _safe_exp(x):
    # Guard against overflow in extreme-loss edge cases.
    return math.exp(x) if x < 700 else float("inf")


def evaluate_lstm(model, dataset):
    """
    Evaluate LSTM on a SequenceDataset.
    Returns dict with loss, perplexity, and accuracy.
    """
    if len(dataset) == 0:
        return {
            "num_samples": 0,
            "loss": None,
            "perplexity": None,
            "accuracy": None,
        }

    device = torch.device(config.device)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(config.num_workers, 0),
        pin_memory=(config.device == "cuda"),
    )

    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == y).sum().item()

            batch_size = y.size(0)
            total_samples += batch_size
            total_loss += loss.item()

    avg_loss = total_loss / max(total_samples, 1)
    return {
        "num_samples": total_samples,
        "loss": avg_loss,
        "perplexity": _safe_exp(avg_loss),
        "accuracy": total_correct / max(total_samples, 1),
    }


def evaluate_ngram(model, token_sequences, vocab_size):
    """
    Evaluate n-gram on token sequences using Laplace-smoothed NLL.
    """
    total_nll = 0.0
    total_tokens = 0
    total_correct = 0

    n = model.n
    if n < 2:
        raise ValueError("Only n-gram models with n >= 2 are supported")

    for seq in token_sequences:
        if len(seq) < n:
            continue

        for i in range(len(seq) - n + 1):
            context = tuple(seq[i:i + n - 1])
            target = seq[i + n - 1]

            context_targets = model.ngram_counts.get(context)
            target_count = context_targets.get(target, 0) if context_targets else 0
            context_count = model.context_counts.get(context, 0)

            # Add-one smoothing so unseen contexts/tokens are finite.
            prob = (target_count + 1) / (context_count + vocab_size)
            total_nll += -math.log(prob)
            total_tokens += 1

            if context_targets:
                pred = max(context_targets, key=context_targets.get)
                total_correct += int(pred == target)

    if total_tokens == 0:
        return {
            "num_samples": 0,
            "loss": None,
            "perplexity": None,
            "accuracy": None,
        }

    avg_loss = total_nll / total_tokens
    return {
        "num_samples": total_tokens,
        "loss": avg_loss,
        "perplexity": _safe_exp(avg_loss),
        "accuracy": total_correct / total_tokens,
    }


def build_qualitative_samples(
    tokenizer,
    lstm_model,
    ngram_model,
    test_token_sequences,
    num_samples,
    max_length,
):
    """
    Build prompt/output samples from held-out test data.
    """
    seed_len = max(config.seq_length, 5)
    candidates = [seq for seq in test_token_sequences if len(seq) > seed_len + 1]
    if not candidates:
        return []

    rng = random.Random(config.random_seed)
    picked = rng.sample(candidates, k=min(num_samples, len(candidates)))

    outputs = []
    for seq in picked:
        prompt_tokens = seq[:seed_len]
        prompt_text = tokenizer.decode(prompt_tokens)

        lstm_output = generate_text(
            lstm_model,
            tokenizer,
            prompt_text,
            max_length=max_length,
        )

        baseline_tokens = ngram_model.generate(prompt_tokens, max_length=max_length)
        baseline_output = tokenizer.decode(baseline_tokens)

        outputs.append(
            {
                "prompt": prompt_text,
                "baseline_output": baseline_output,
                "lstm_output": lstm_output,
            }
        )

    return outputs


def save_json_report(payload, file_path):
    """
    Save JSON report artifact.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_qualitative_text(samples, file_path):
    """
    Save qualitative outputs in a report-friendly text format.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples, start=1):
            f.write(f"Sample {idx}\n")
            f.write(f"Prompt: {sample['prompt']}\n")
            f.write(f"Baseline: {sample['baseline_output']}\n")
            f.write(f"LSTM: {sample['lstm_output']}\n")
            f.write("\n")
