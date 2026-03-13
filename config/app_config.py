"""Singleton application configuration loaded from environment variables.

Configuration loading follows this precedence:
1. Environment values from an `.env` file (if provided).
2. Existing process environment variables.
3. In-code defaults defined in dataclass fields.

The CLI uses this module as the primary source of truth so training and
inference commands can run with minimal arguments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths for datasets and model artifacts."""

    local_input: str = "data/sample_lyrics.txt"
    vocab_out: str = "artifacts/vocab.json"
    model_out: str = "artifacts/lyrics_lstm.pt"


@dataclass(frozen=True)
class DataConfig:
    """Dataset source and ingestion settings."""

    data_source: str = "local"
    use_sample_if_missing: bool = True

    hf_dataset: Optional[str] = None
    hf_config: Optional[str] = None
    hf_split: str = "train"
    hf_token: Optional[str] = None

    kaggle_dataset: Optional[str] = None
    kaggle_file_pattern: str = "*.csv"

    text_column: str = "text"
    max_samples: Optional[int] = None
    val_ratio: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    """Training and model hyperparameters."""

    min_freq: int = 1
    max_vocab: int = 5000

    seq_len: int = 12
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-3
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2

    seed: int = 42
    cpu: bool = False
    preview_prompt: str = "we sing"


@dataclass(frozen=True)
class GenerationConfig:
    """Generation-time decoding parameters."""

    prompt: str = "midnight"
    max_new_tokens: int = 40
    temperature: float = 0.9
    top_k: int = 20
    cpu: bool = False


@dataclass(frozen=True)
class LoggingConfig:
    """Runtime logging settings for CLI command execution."""

    log_file: str = "logs/cli_runs.log"
    level: str = "INFO"


@dataclass(frozen=True)
class AppConfig:
    """Root config object used by the CLI and runtime pipeline."""

    paths: PathsConfig
    data: DataConfig
    train: TrainingConfig
    generation: GenerationConfig
    logging: LoggingConfig


class ConfigSingleton:
    """Singleton manager that caches one resolved :class:`AppConfig` instance."""

    _instance: Optional[AppConfig] = None
    _loaded_env_file: Optional[str] = None

    @classmethod
    def get(cls, env_file: str = ".env", force_reload: bool = False) -> AppConfig:
        """Return the singleton config instance.

        Args:
            env_file: Path to the `.env` file to load first.
            force_reload: If `True`, rebuild config even if cached.

        Returns:
            Resolved :class:`AppConfig` instance.
        """
        should_reload = force_reload or cls._instance is None or cls._loaded_env_file != env_file
        if should_reload:
            cls._instance = _build_config(env_file)
            cls._loaded_env_file = env_file
        return cls._instance


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a string environment value, treating empty values as missing."""
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """Read an integer environment value with optional default."""
    value = _env_str(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    """Read a float environment value with required default fallback."""
    value = _env_str(name)
    if value is None:
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment value.

    Truthy values: `1`, `true`, `yes`, `on`
    Falsy values: `0`, `false`, `no`, `off`
    """
    value = _env_str(name)
    if value is None:
        return default

    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value}")


def _build_config(env_file: str) -> AppConfig:
    """Build an application config from `.env` + process environment."""
    env_path = Path(env_file)
    load_dotenv(dotenv_path=env_path if env_path.exists() else None, override=False)

    paths = PathsConfig(
        local_input=_env_str("LYRICS_LOCAL_INPUT", PathsConfig.local_input) or PathsConfig.local_input,
        vocab_out=_env_str("LYRICS_VOCAB_OUT", PathsConfig.vocab_out) or PathsConfig.vocab_out,
        model_out=_env_str("LYRICS_MODEL_OUT", PathsConfig.model_out) or PathsConfig.model_out,
    )

    data = DataConfig(
        data_source=_env_str("LYRICS_DATA_SOURCE", DataConfig.data_source) or DataConfig.data_source,
        use_sample_if_missing=_env_bool(
            "LYRICS_USE_SAMPLE_IF_MISSING", DataConfig.use_sample_if_missing
        ),
        hf_dataset=_env_str("LYRICS_HF_DATASET", None),
        hf_config=_env_str("LYRICS_HF_CONFIG", None),
        hf_split=_env_str("LYRICS_HF_SPLIT", DataConfig.hf_split) or DataConfig.hf_split,
        hf_token=_env_str("HF_TOKEN", None) or _env_str("LYRICS_HF_TOKEN", None),
        kaggle_dataset=_env_str("LYRICS_KAGGLE_DATASET", None),
        kaggle_file_pattern=(
            _env_str("LYRICS_KAGGLE_FILE_PATTERN", DataConfig.kaggle_file_pattern)
            or DataConfig.kaggle_file_pattern
        ),
        text_column=_env_str("LYRICS_TEXT_COLUMN", DataConfig.text_column) or DataConfig.text_column,
        max_samples=_env_int("LYRICS_MAX_SAMPLES", None),
        val_ratio=_env_float("LYRICS_VAL_RATIO", DataConfig.val_ratio),
    )

    train = TrainingConfig(
        min_freq=_env_int("LYRICS_MIN_FREQ", TrainingConfig.min_freq) or TrainingConfig.min_freq,
        max_vocab=_env_int("LYRICS_MAX_VOCAB", TrainingConfig.max_vocab) or TrainingConfig.max_vocab,
        seq_len=_env_int("LYRICS_SEQ_LEN", TrainingConfig.seq_len) or TrainingConfig.seq_len,
        batch_size=_env_int("LYRICS_BATCH_SIZE", TrainingConfig.batch_size) or TrainingConfig.batch_size,
        epochs=_env_int("LYRICS_EPOCHS", TrainingConfig.epochs) or TrainingConfig.epochs,
        lr=_env_float("LYRICS_LR", TrainingConfig.lr),
        embedding_dim=(
            _env_int("LYRICS_EMBEDDING_DIM", TrainingConfig.embedding_dim)
            or TrainingConfig.embedding_dim
        ),
        hidden_dim=_env_int("LYRICS_HIDDEN_DIM", TrainingConfig.hidden_dim) or TrainingConfig.hidden_dim,
        num_layers=_env_int("LYRICS_NUM_LAYERS", TrainingConfig.num_layers) or TrainingConfig.num_layers,
        dropout=_env_float("LYRICS_DROPOUT", TrainingConfig.dropout),
        seed=_env_int("LYRICS_SEED", TrainingConfig.seed) or TrainingConfig.seed,
        cpu=_env_bool("LYRICS_CPU", TrainingConfig.cpu),
        preview_prompt=(
            _env_str("LYRICS_PREVIEW_PROMPT", TrainingConfig.preview_prompt)
            or TrainingConfig.preview_prompt
        ),
    )

    generation = GenerationConfig(
        prompt=_env_str("LYRICS_PROMPT", GenerationConfig.prompt) or GenerationConfig.prompt,
        max_new_tokens=(
            _env_int("LYRICS_MAX_NEW_TOKENS", GenerationConfig.max_new_tokens)
            or GenerationConfig.max_new_tokens
        ),
        temperature=_env_float("LYRICS_TEMPERATURE", GenerationConfig.temperature),
        top_k=_env_int("LYRICS_TOP_K", GenerationConfig.top_k) or GenerationConfig.top_k,
        cpu=_env_bool("LYRICS_GENERATE_CPU", GenerationConfig.cpu),
    )

    logging = LoggingConfig(
        log_file=_env_str("LYRICS_LOG_FILE", LoggingConfig.log_file) or LoggingConfig.log_file,
        level=_env_str("LYRICS_LOG_LEVEL", LoggingConfig.level) or LoggingConfig.level,
    )

    return AppConfig(paths=paths, data=data, train=train, generation=generation, logging=logging)
