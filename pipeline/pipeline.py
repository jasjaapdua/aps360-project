"""
pipeline.py

Main pipeline orchestrating the entire lyrics generation workflow.
"""

import os
from config import config

from data.loader import LyricsDatasetLoader
from preprocessing.text_cleaner import clean_lyrics
from preprocessing.tokenizer import Tokenizer

from project_datasets.sequence_dataset import SequenceDataset
from project_datasets.splitter import split_items

from models.ngram_model import NGramLanguageModel
from models.lstm_model import LSTMLanguageModel

from training.trainer import Trainer
from training.generate import generate_text
from training.persistence import load_model
from training.evaluation import (
    evaluate_lstm,
    evaluate_ngram,
    build_qualitative_samples,
    save_json_report,
    save_qualitative_text,
)


class LyricsGenerationPipeline:
    """
    End-to-end pipeline for lyric generation.
    """

    def __init__(self):
        self.loader = LyricsDatasetLoader()

        self.tokenizer = Tokenizer()

        self.ngram_model = None
        self.lstm_model = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.lyrics = None
        self.cleaned_lyrics = None
        self.train_lyrics = None
        self.val_lyrics = None
        self.test_lyrics = None
        self.train_sequences = None
        self.val_sequences = None
        self.test_sequences = None

        self.model_checkpoint_path = "checkpoints/lstm_model.pt"
        self.tokenizer_path = "checkpoints/tokenizer_vocab.json"

    def load_data(self):
        """
        Load dataset.
        """
        print("Loading dataset...")
        self.lyrics = self.loader.load_dataset()
        print(f"Loaded {len(self.lyrics)} songs")

    def clean_data(self):
        """
        Clean lyric text.
        """
        print("Cleaning lyrics...")
        self.cleaned_lyrics = clean_lyrics(self.lyrics)

    def tokenize(self):
        """
        Build vocabulary from train split and encode all splits.
        """
        print("Building tokenizer vocabulary...")
        self.tokenizer.build_vocab(self.train_lyrics)
        print("Encoding split lyrics...")
        self.train_sequences = self.tokenizer.encode_batch(self.train_lyrics)
        self.val_sequences = self.tokenizer.encode_batch(self.val_lyrics)
        self.test_sequences = self.tokenizer.encode_batch(self.test_lyrics)

    def split_data(self):
        """
        Split cleaned songs into train/val/test partitions.
        """
        print("Splitting cleaned songs...")
        self.train_lyrics, self.val_lyrics, self.test_lyrics = split_items(
            self.cleaned_lyrics,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.random_seed,
        )
        print(
            "Split sizes (songs) | "
            f"train={len(self.train_lyrics)} "
            f"val={len(self.val_lyrics)} "
            f"test={len(self.test_lyrics)}"
        )

    def build_dataset(self):
        """
        Build sequence datasets for LSTM.
        """
        print("Creating sequence datasets...")
        self.train_dataset = SequenceDataset(self.train_sequences, config.seq_length)
        self.val_dataset = SequenceDataset(self.val_sequences, config.seq_length)
        self.test_dataset = SequenceDataset(self.test_sequences, config.seq_length)
        print(
            "Split sizes (samples) | "
            f"train={len(self.train_dataset)} "
            f"val={len(self.val_dataset)} "
            f"test={len(self.test_dataset)}"
        )

    def train_baseline(self):
        """
        Train N-gram baseline model.
        """
        print("Training baseline n-gram model...")
        self.ngram_model = NGramLanguageModel(n=2)
        self.ngram_model.train(self.train_sequences)

    def train_lstm(self):
        """
        Train LSTM model.
        """
        print("Training LSTM model...")
        self.lstm_model = LSTMLanguageModel(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        )
        trainer = Trainer(self.lstm_model, self.train_dataset)
        trainer.train()
        os.makedirs(os.path.dirname(self.tokenizer_path), exist_ok=True)
        self.tokenizer.save(self.tokenizer_path)

    def _load_lstm_and_tokenizer(self):
        """
        Load tokenizer and LSTM checkpoint from disk.
        """
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {self.tokenizer_path}")
        if not os.path.exists(self.model_checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_checkpoint_path}")

        self.tokenizer.load(self.tokenizer_path)
        self.lstm_model = LSTMLanguageModel(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        )
        load_model(self.lstm_model, self.model_checkpoint_path, device=config.device)
        self.lstm_model.to(config.device)

    def generate(self, seed_text):
        """
        Generate lyrics from trained model.
        """
        if self.lstm_model is None:
            try:
                self._load_lstm_and_tokenizer()
            except FileNotFoundError:
                print("Checkpoint/tokenizer not found. Training model first...")
                self.load_data()
                self.clean_data()
                self.split_data()
                self.tokenize()
                self.build_dataset()
                self.train_lstm()
        print("Generating lyrics...\n")
        result = generate_text(self.lstm_model, self.tokenizer, seed_text)
        print(result)
        return result

    def _evaluate_on_new_data(self):
        """
        Optional external/new-data evaluation if configured.
        """
        if not config.new_dataset_source or not config.new_dataset_id:
            return None

        print("Evaluating on external unseen dataset...")
        external_loader = LyricsDatasetLoader(
            dataset_source=config.new_dataset_source,
            dataset_id=config.new_dataset_id,
        )

        original_max_songs = config.hf_max_songs
        try:
            config.hf_max_songs = config.new_dataset_max_songs
            lyrics = external_loader.load_dataset()
        finally:
            config.hf_max_songs = original_max_songs

        cleaned = clean_lyrics(lyrics)
        token_sequences = self.tokenizer.encode_batch(cleaned)
        ext_dataset = SequenceDataset(token_sequences, config.seq_length)

        return {
            "dataset_source": config.new_dataset_source,
            "dataset_id": config.new_dataset_id,
            "num_songs": len(token_sequences),
            "num_samples": len(ext_dataset),
            "baseline": evaluate_ngram(self.ngram_model, token_sequences, self.tokenizer.vocab_size),
            "lstm": evaluate_lstm(self.lstm_model, ext_dataset),
        }

    def evaluate(self):
        """
        Train models and produce held-out evaluation artifacts.
        """
        self.load_data()
        self.clean_data()
        self.split_data()
        self.tokenize()
        self.build_dataset()
        self.train_baseline()
        self.train_lstm()

        report = {
            "config": {
                "seq_length": config.seq_length,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "train_ratio": config.train_ratio,
                "val_ratio": config.val_ratio,
                "test_ratio": config.test_ratio,
                "random_seed": config.random_seed,
            },
            "split_song_counts": {
                "train": len(self.train_sequences),
                "val": len(self.val_sequences),
                "test": len(self.test_sequences),
            },
            "split_sample_counts": {
                "train": len(self.train_dataset),
                "val": len(self.val_dataset),
                "test": len(self.test_dataset),
            },
            "held_out_metrics": {
                "validation": {
                    "baseline": evaluate_ngram(
                        self.ngram_model,
                        self.val_sequences,
                        self.tokenizer.vocab_size,
                    ),
                    "lstm": evaluate_lstm(self.lstm_model, self.val_dataset),
                },
                "test": {
                    "baseline": evaluate_ngram(
                        self.ngram_model,
                        self.test_sequences,
                        self.tokenizer.vocab_size,
                    ),
                    "lstm": evaluate_lstm(self.lstm_model, self.test_dataset),
                },
            },
        }

        external_metrics = self._evaluate_on_new_data()
        if external_metrics is not None:
            report["external_new_data_metrics"] = external_metrics

        qualitative_samples = build_qualitative_samples(
            tokenizer=self.tokenizer,
            lstm_model=self.lstm_model,
            ngram_model=self.ngram_model,
            test_token_sequences=self.test_sequences,
            num_samples=max(config.qual_num_samples, 1),
            max_length=max(config.qual_max_length, 1),
        )
        report["qualitative_sample_count"] = len(qualitative_samples)

        eval_json_path = os.path.join(config.report_dir, "evaluation_summary.json")
        qual_txt_path = os.path.join(config.report_dir, "qualitative_samples.txt")
        save_json_report(report, eval_json_path)
        save_qualitative_text(qualitative_samples, qual_txt_path)

        print(f"Saved evaluation summary to {eval_json_path}")
        print(f"Saved qualitative samples to {qual_txt_path}")
        return report

    def run(self):
        """
        Run the full pipeline.
        """
        self.load_data()
        self.clean_data()
        self.split_data()
        self.tokenize()
        self.build_dataset()
        self.train_baseline()
        self.train_lstm()
