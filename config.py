"""
config.py

Singleton configuration loader that reads settings from a .env file.
Environment variables remain UPPERCASE, but code uses snake_case.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env"


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
        load_dotenv(dotenv_path=_ENV_PATH)
        self.dataset_path = os.getenv("DATASET_PATH", "data/lyrics.txt")
        self.hf_token = os.getenv("HF_TOKEN", None)
        self.dataset_source = os.getenv("DATASET_SOURCE", "huggingface")
        self.dataset_id = os.getenv("DATASET_ID")
        self.hf_max_songs = int(os.getenv("HF_MAX_SONGS", "20000"))
        self.hf_progress_every = int(os.getenv("HF_PROGRESS_EVERY", "1000"))
        self.hf_streaming = os.getenv("HF_STREAMING", "false").lower() in ("true", "1", "yes")

        # ----------------------
        # training parameters
        # ----------------------
        self.seq_length = int(os.getenv("SEQ_LENGTH", 20))
        self.batch_size = int(os.getenv("BATCH_SIZE", 64))
        self.epochs = int(os.getenv("EPOCHS", 20))
        self.learning_rate = float(os.getenv("LEARNING_RATE", 0.001))
        self.max_train_samples = int(os.getenv("MAX_TRAIN_SAMPLES", "100000"))
        self.max_steps_per_epoch = int(os.getenv("MAX_STEPS_PER_EPOCH", "1000"))
        self.num_workers = int(os.getenv("NUM_WORKERS", "0"))
        self.random_seed = int(os.getenv("RANDOM_SEED", "42"))
        self.train_ratio = float(os.getenv("TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.getenv("VAL_RATIO", "0.1"))
        self.test_ratio = float(os.getenv("TEST_RATIO", "0.1"))

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
        self.qual_num_samples = int(os.getenv("QUAL_NUM_SAMPLES", "10"))
        self.qual_max_length = int(os.getenv("QUAL_MAX_LENGTH", "50"))
        self.report_dir = os.getenv("REPORT_DIR", "reports")

        # ----------------------
        # external/new-data evaluation
        # ----------------------
        self.new_dataset_source = os.getenv("NEW_DATASET_SOURCE")
        self.new_dataset_id = os.getenv("NEW_DATASET_ID")
        self.new_dataset_max_songs = int(os.getenv("NEW_DATASET_MAX_SONGS", "5000"))

        # ----------------------
        # device
        # ----------------------
        self.device = os.getenv("DEVICE", "cpu")


# singleton instance
config = Config()
