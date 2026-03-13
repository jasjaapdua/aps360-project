"""Public model API exposing neural and statistical lyric generators."""

from models.lstm import (
    LyricsLSTM,
    TrainConfig,
    evaluate_lstm_loss,
    generate_text,
    load_checkpoint,
    save_checkpoint,
    train_lstm_model,
)
from models.trigram import evaluate_trigram_accuracy, generate_trigram, train_trigram

__all__ = [
    "LyricsLSTM",
    "TrainConfig",
    "train_lstm_model",
    "evaluate_lstm_loss",
    "generate_text",
    "save_checkpoint",
    "load_checkpoint",
    "train_trigram",
    "generate_trigram",
    "evaluate_trigram_accuracy",
]
