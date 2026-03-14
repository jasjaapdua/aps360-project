"""Plot helpers for training metrics using matplotlib."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

# Safe default for headless environments (CI, remote servers).
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _epochs(n: int) -> list[int]:
    return list(range(1, n + 1))


def _to_perplexity(losses: Sequence[float]) -> list[float]:
    return [float("inf") if loss > 50 else math.exp(loss) for loss in losses]


def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    output_path: str | Path | None = None,
    title: str = "Training Loss",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot train/validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(_epochs(len(train_losses)), train_losses, marker="o", label="Train Loss")

    if val_losses:
        ax.plot(_epochs(len(val_losses)), val_losses, marker="o", label="Val Loss")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)

    return fig, ax


def plot_perplexity_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    output_path: str | Path | None = None,
    title: str = "Training Perplexity",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot train/validation perplexity over epochs from loss values."""
    train_ppl = _to_perplexity(train_losses)
    val_ppl = _to_perplexity(val_losses) if val_losses else None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(_epochs(len(train_ppl)), train_ppl, marker="o", label="Train Perplexity")

    if val_ppl:
        ax.plot(_epochs(len(val_ppl)), val_ppl, marker="o", label="Val Perplexity")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)

    return fig, ax


def plot_training_dashboard(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    output_path: str | Path | None = None,
    title: str = "Training Dashboard",
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create a two-panel figure: loss (left) and perplexity (right)."""
    train_ppl = _to_perplexity(train_losses)
    val_ppl = _to_perplexity(val_losses) if val_losses else None

    fig, (ax_loss, ax_ppl) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(_epochs(len(train_losses)), train_losses, marker="o", label="Train Loss")
    if val_losses:
        ax_loss.plot(_epochs(len(val_losses)), val_losses, marker="o", label="Val Loss")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_ppl.plot(_epochs(len(train_ppl)), train_ppl, marker="o", label="Train Perplexity")
    if val_ppl:
        ax_ppl.plot(_epochs(len(val_ppl)), val_ppl, marker="o", label="Val Perplexity")
    ax_ppl.set_title("Perplexity")
    ax_ppl.set_xlabel("Epoch")
    ax_ppl.set_ylabel("Perplexity")
    ax_ppl.grid(alpha=0.3)
    ax_ppl.legend()

    fig.suptitle(title)
    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)

    return fig, (ax_loss, ax_ppl)
