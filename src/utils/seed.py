"""
seed.py

Utility for reproducible experiments.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False, benchmark: bool = False) -> None:
    """Set RNG seeds for python, numpy and torch.

    Args:
        seed: Random seed value.
        deterministic: If True, force deterministic CuDNN.
        benchmark: If True, enable CuDNN benchmarking (ignored when deterministic).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark and not deterministic
