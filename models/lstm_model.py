import torch
from torch import nn


class LSTMLanguageModel(nn.Module):
    """
    Simple LSTM language model.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        logits = self.fc(last_output)

        return logits
