"""
persistence.py

Utilities for saving and loading trained models.
"""

import os
import torch


def save_model(model, path):
    """
    Save PyTorch model weights.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(model.state_dict(), path)

    print(f"Model saved to {path}")


def load_model(model, path, device="cpu"):
    """
    Load model weights into an existing model instance.
    """

    state_dict = torch.load(path, map_location=device)

    model.load_state_dict(state_dict)

    print(f"Model loaded from {path}")

    return model
