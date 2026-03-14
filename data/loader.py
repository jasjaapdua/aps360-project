"""
loader.py

Dataset loader for lyrics generation project.

Supports:
- HuggingFace datasets
- Kaggle datasets

Returns a list of lyric strings.
"""

import site
import sys
from pathlib import Path
import pandas as pd
import kagglehub
from config import config


def _resolve_hf_load_dataset():
    """
    Resolve Hugging Face's datasets.load_dataset even when a local
    `datasets/` package exists in the project.
    """
    original_sys_path = list(sys.path)
    project_root = str(Path(__file__).resolve().parents[1])
    cwd_entry = ""

    # Remove project path entries so local datasets/ doesn't shadow HF datasets.
    filtered_sys_path = [
        p
        for p in original_sys_path
        if p not in (project_root, cwd_entry)
    ]

    # Prefer site-packages first.
    for site_dir in reversed(site.getsitepackages()):
        if site_dir not in filtered_sys_path:
            filtered_sys_path.insert(0, site_dir)

    try:
        sys.path = filtered_sys_path
        from datasets import load_dataset as hf_load_dataset

        return hf_load_dataset
    finally:
        sys.path = original_sys_path


load_dataset = _resolve_hf_load_dataset()


class LyricsDatasetLoader:
    """
    Loader class responsible for retrieving lyric datasets
    from supported sources.
    """

    def __init__(self, dataset_source=None, dataset_id=None):
        """
        Initialize loader.

        Parameters
        ----------
        dataset_source : str
            Source of dataset ("huggingface" or "kaggle")
        dataset_id : str
            Dataset identifier
        """

        self.dataset_source = dataset_source or config.dataset_source
        self.dataset_id = dataset_id or config.dataset_id

        if not self.dataset_id:
            raise ValueError("dataset_id must be provided")

    def load_huggingface_dataset(self):
        """
        Load lyrics dataset from HuggingFace.
        """
        auth_mode = "authenticated" if config.hf_token else "unauthenticated"
        print(f"HF Hub mode: {auth_mode}")

        try:
            dataset = load_dataset(
                self.dataset_id,
                split="train",
                token=config.hf_token,
                streaming=config.hf_streaming
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Hugging Face dataset "
                f"'{self.dataset_id}'. If this is a Kaggle dataset slug, "
                "set DATASET_SOURCE=kaggle in .env."
            ) from exc

        lyrics = []
        max_songs = config.hf_max_songs if config.hf_max_songs > 0 else None
        progress_every = max(config.hf_progress_every, 1)

        for idx, row in enumerate(dataset, start=1):
            text = row.get("lyrics")

            if isinstance(text, str) and text.strip():
                lyrics.append(text)

            if idx % progress_every == 0:
                print(f"Processed {idx} rows; kept {len(lyrics)} lyrics")

            if max_songs and len(lyrics) >= max_songs:
                print(f"Reached HF_MAX_SONGS={max_songs}; stopping early")
                break

        return lyrics

    def load_kaggle_dataset(self):
        """
        Load lyrics dataset from Kaggle.
        """

        path = kagglehub.dataset_download(self.dataset_id)

        df = pd.read_csv(f"{path}/song_lyrics.csv")

        return df["lyrics"].dropna().tolist()

    def load_dataset(self):
        """
        Main dataset loader.
        """

        if self.dataset_source == "huggingface":
            return self.load_huggingface_dataset()

        if self.dataset_source == "kaggle":
            return self.load_kaggle_dataset()

        raise ValueError(f"Unsupported dataset source: {self.dataset_source}")
