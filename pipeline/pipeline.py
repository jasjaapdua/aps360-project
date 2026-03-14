"""
pipeline.py

Main pipeline orchestrating the entire lyrics generation workflow.
"""

from config import config

from data.loader import LyricsDatasetLoader
from preprocessing.text_cleaner import clean_lyrics
from preprocessing.tokenizer import Tokenizer

from project_datasets.sequence_dataset import SequenceDataset

from models.ngram_model import NGramLanguageModel
from models.lstm_model import LSTMLanguageModel

from training.trainer import Trainer
from training.generate import generate_text


class LyricsGenerationPipeline:
    """
    End-to-end pipeline for lyric generation.
    """

    def __init__(self):
        self.loader = LyricsDatasetLoader()

        self.tokenizer = Tokenizer()

        self.ngram_model = None
        self.lstm_model = None

        self.dataset = None

        self.lyrics = None
        self.cleaned_lyrics = None
        self.token_sequences = None

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
        Build vocabulary and encode lyrics.
        """
        print("Building tokenizer vocabulary...")
        self.tokenizer.build_vocab(self.cleaned_lyrics)
        print("Encoding lyrics...")
        self.token_sequences = self.tokenizer.encode_batch(self.cleaned_lyrics)

    def build_dataset(self):
        """
        Build sequence dataset for LSTM training.
        """
        print("Creating sequence dataset...")
        self.dataset = SequenceDataset(self.token_sequences, config.seq_length)
        print(f"Sequence samples: {len(self.dataset)}")

    def train_baseline(self):
        """
        Train N-gram baseline model.
        """
        print("Training baseline n-gram model...")
        self.ngram_model = NGramLanguageModel(n=2)
        self.ngram_model.train(self.token_sequences)

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
        trainer = Trainer(self.lstm_model, self.dataset)
        trainer.train()

    def generate(self, seed_text):
        """
        Generate lyrics from trained model.
        """
        print("Generating lyrics...\n")
        result = generate_text(self.lstm_model, self.tokenizer, seed_text)
        print(result)
        return result

    def run(self):
        """
        Run the full pipeline.
        """
        self.load_data()
        self.clean_data()
        self.tokenize()
        self.build_dataset()
        self.train_baseline()
        self.train_lstm()
