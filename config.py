"""
config.py

Singleton configuration loader that reads settings from a .env file.
Environment variables remain UPPERCASE, but code uses snake_case.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Load configuration values from .env and environment variables.
        Runs only once.
        """
        # ----------------------
        # dataset
        # ----------------------
        load_dotenv()
        self.dataset_path = os.getenv("DATASET_PATH", "data/lyrics.txt")
        self.hf_token = os.getenv("HF_TOKEN", None)
        self.dataset_source = os.getenv("DATASET_SOURCE", "huggingface")
        self.dataset_id = os.getenv("DATASET_ID")
        self.hf_token = os.getenv("HF_TOKEN")
        self.hf_streaming = (os.getenv("HF_STREAMING").lower() or "false") == "true"

        # ----------------------
        # training parameters
        # ----------------------
        self.seq_length = int(os.getenv("SEQ_LENGTH", 20))
        self.batch_size = int(os.getenv("BATCH_SIZE", 64))
        self.epochs = int(os.getenv("EPOCHS", 20))
        self.learning_rate = float(os.getenv("LEARNING_RATE", 0.001))

        # ----------------------
        # model parameters
        # ----------------------
        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", 128))
        self.hidden_size = int(os.getenv("HIDDEN_SIZE", 256))
        self.num_layers = int(os.getenv("NUM_LAYERS", 2))

        # ----------------------
        # generation parameters
        # ----------------------
        self.max_generation_length = int(os.getenv("MAX_GENERATION_LENGTH", 100))
        self.temperature = float(os.getenv("TEMPERATURE", 0.8))

        # ----------------------
        # device
        # ----------------------
        self.device = os.getenv("DEVICE", "cpu")


# singleton instance
config = Config()
