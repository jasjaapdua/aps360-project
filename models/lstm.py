"""LSTM language model components for lyric generation.

This module provides:
- a next-token training dataset over flattened token streams,
- a compact LSTM language model,
- training and sampling utilities,
- checkpoint save/load helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from preprocessing.tokenizer import LyricsTokenizer


class SequenceDataset(Dataset):
    """Create fixed-length next-token training windows from a token sequence.

    Given a long token stream `[t0, t1, ...]` and `seq_len = N`,
    each sample is:
    - input:  `[ti, ..., ti+N-1]`
    - target: `[ti+1, ..., ti+N]`
    """

    def __init__(self, token_ids: list[int], seq_len: int):
        """Precompute sliding windows for autoregressive training.

        Args:
            token_ids: Full token-id stream.
            seq_len: Length of each input sequence.
        """
        self.seq_len = seq_len
        self.x: list[torch.Tensor] = []
        self.y: list[torch.Tensor] = []

        for i in range(0, len(token_ids) - seq_len):
            x = token_ids[i : i + seq_len]
            y = token_ids[i + 1 : i + seq_len + 1]
            self.x.append(torch.tensor(x, dtype=torch.long))
            self.y.append(torch.tensor(y, dtype=torch.long))

    def __len__(self) -> int:
        """Return the number of training windows."""
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one `(input_ids, target_ids)` training pair."""
        return self.x[idx], self.y[idx]


class LyricsLSTM(nn.Module):
    """Word-level LSTM language model with embedding and output projection."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """Initialize model layers.

        Args:
            vocab_size: Number of distinct tokens.
            embedding_dim: Token embedding width.
            hidden_dim: LSTM hidden size.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout used between LSTM layers when `num_layers > 1`.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run a forward pass.

        Args:
            input_ids: Tensor of shape `(batch, seq_len)`.
            hidden: Optional recurrent state `(h, c)` from a previous step.

        Returns:
            Tuple `(logits, hidden)` where:
            - `logits` has shape `(batch, seq_len, vocab_size)`,
            - `hidden` is the updated recurrent state.
        """
        emb = self.embedding(input_ids)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


@dataclass
class TrainConfig:
    """Hyperparameters for LSTM training."""

    seq_len: int = 20
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-3
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2


def flatten_corpus_ids(corpus: Iterable[list[str]], tokenizer: LyricsTokenizer) -> list[int]:
    """Encode a tokenized corpus into one continuous token-id stream.

    Args:
        corpus: List/iterable of tokenized lines.
        tokenizer: Tokenizer used to convert tokens to ids.

    Returns:
        Flattened token-id stream with `<bos>/<eos>` inserted per line.
    """
    all_ids: list[int] = []
    for line in corpus:
        all_ids.extend(tokenizer.encode_tokens(line, add_special_tokens=True))
    return all_ids


def train_lstm_model(
    train_corpus: list[list[str]],
    tokenizer: LyricsTokenizer,
    config: TrainConfig,
    val_corpus: list[list[str]] | None = None,
    device: str = "cpu",
) -> tuple[LyricsLSTM, list[float], list[float]]:
    """Train an LSTM language model on lyric tokens.

    Args:
        train_corpus: Tokenized lyric lines for training.
        tokenizer: Tokenizer aligned with the training vocabulary.
        config: Training hyperparameters.
        val_corpus: Optional tokenized lyric lines for validation.
        device: Torch device string (`\"cpu\"`, `\"cuda\"`, etc.).

    Returns:
        Tuple `(model, train_losses, val_losses)` where each loss list stores
        average loss per epoch.

    Raises:
        ValueError: If the corpus is too small for the configured sequence length.
    """
    token_ids = flatten_corpus_ids(train_corpus, tokenizer)
    if len(token_ids) <= config.seq_len + 1:
        raise ValueError("Corpus is too small for the configured sequence length.")

    dataset = SequenceDataset(token_ids, config.seq_len)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = LyricsLSTM(
        vocab_size=len(tokenizer.stoi),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    for _epoch in range(config.epochs):
        model.train()
        total = 0.0
        count = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += loss.item()
            count += 1

        train_losses.append(total / max(count, 1))

        if val_corpus:
            val_loss = evaluate_lstm_loss(
                model=model,
                corpus=val_corpus,
                tokenizer=tokenizer,
                seq_len=config.seq_len,
                batch_size=config.batch_size,
                device=device,
            )
            val_losses.append(val_loss)

    return model, train_losses, val_losses


@torch.no_grad()
def evaluate_lstm_loss(
    model: LyricsLSTM,
    corpus: list[list[str]],
    tokenizer: LyricsTokenizer,
    seq_len: int,
    batch_size: int = 32,
    device: str = "cpu",
) -> float:
    """Evaluate average cross-entropy loss on a tokenized corpus.

    Args:
        model: Trained LSTM model.
        corpus: Tokenized lines used for evaluation.
        tokenizer: Tokenizer matching the model vocabulary.
        seq_len: Sequence length used to build evaluation windows.
        batch_size: Evaluation batch size.
        device: Torch device string.

    Returns:
        Mean cross-entropy loss over all evaluation batches.
    """
    token_ids = flatten_corpus_ids(corpus, tokenizer)
    if len(token_ids) <= seq_len + 1:
        raise ValueError("Evaluation corpus is too small for the configured sequence length.")

    dataset = SequenceDataset(token_ids, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += loss.item()
        count += 1

    if was_training:
        model.train()

    return total / max(count, 1)


@torch.no_grad()
def generate_text(
    model: LyricsLSTM,
    tokenizer: LyricsTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 20,
    device: str = "cpu",
) -> str:
    """Generate lyric text by autoregressive token sampling.

    Args:
        model: Trained LSTM language model.
        tokenizer: Tokenizer used during training.
        prompt: Seed text to prime generation.
        max_new_tokens: Maximum number of sampled tokens to append.
        temperature: Logit temperature (>1.0 more random, <1.0 more conservative).
        top_k: If positive, sample only from the top-k candidates each step.
        device: Torch device string.

    Returns:
        Generated lyric text as a decoded string.
    """
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)

    hidden = None
    generated = ids[:-1]

    for _ in range(max_new_tokens):
        logits, hidden = model(input_ids, hidden)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k > 0 and top_k < next_logits.size(-1):
            values, indices = torch.topk(next_logits, k=top_k, dim=-1)
            probs = torch.softmax(values, dim=-1)
            sampled = indices.gather(-1, torch.multinomial(probs, num_samples=1))
            next_id = sampled.squeeze(0).item()
        else:
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

        if next_id == tokenizer.eos_id:
            break

        generated.append(next_id)
        input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return tokenizer.decode(generated)


def save_checkpoint(
    model: LyricsLSTM,
    path: str,
    config: TrainConfig,
) -> None:
    """Save model weights and minimal reconstruction metadata.

    Args:
        model: Trained model instance.
        path: Destination checkpoint file path.
        config: Training config required to reconstruct architecture.
    """
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "vocab_size": model.embedding.num_embeddings,
        },
        path,
    )


def load_checkpoint(path: str, device: str = "cpu") -> tuple[LyricsLSTM, TrainConfig]:
    """Load a model checkpoint produced by :func:`save_checkpoint`.

    Args:
        path: Checkpoint file path.
        device: Torch device mapping target.

    Returns:
        Tuple `(model, config)` with loaded weights and hyperparameters.
    """
    ckpt = torch.load(path, map_location=device)
    config = TrainConfig(**ckpt["config"])
    model = LyricsLSTM(
        vocab_size=ckpt["vocab_size"],
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    return model, config
