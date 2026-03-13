"""
training_plots.py

Plot training and validation loss curves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def plot_losses(history: Dict[str, list], out_path: str | Path = "training_curve.png") -> Path:
    """Plot loss curves and save to disk."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.get("train_loss", []), label="train")
    plt.plot(history.get("val_loss", []), label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
