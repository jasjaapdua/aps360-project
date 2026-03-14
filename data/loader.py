"""
loader.py

Dataset loader for lyrics generation project.

Supports:
- HuggingFace datasets
- Kaggle datasets

Returns a list of lyric strings.
"""

from datasets import load_dataset
import pandas as pd
import kagglehub
from config import config


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

        dataset = load_dataset(
            self.dataset_id, 
            split="train",
            token=config.hf_token,
            streaming=config.hf_streaming
        )

        lyrics = []

        for row in dataset:
            text = row.get("lyrics")

            if isinstance(text, str) and text.strip():
                lyrics.append(text)

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
