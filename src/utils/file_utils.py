"""
file_utils.py

Lightweight helpers for filesystem interactions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Path object to the directory.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save dictionary as pretty-printed JSON."""
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load JSON file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
