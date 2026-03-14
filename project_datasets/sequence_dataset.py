import torch
from torch.utils.data import Dataset
from config import config


class SequenceDataset(Dataset):
    """
    Dataset for next-word prediction.
    """

    def __init__(self, token_sequences, seq_length):
        self.seq_length = seq_length
        self.samples = []
        max_train_samples = config.max_train_samples if config.max_train_samples > 0 else None

        for seq in token_sequences:
            if len(seq) <= seq_length:
                continue

            for i in range(len(seq) - seq_length):
                x = seq[i:i + seq_length]
                y = seq[i + seq_length]

                self.samples.append((x, y))
                if max_train_samples and len(self.samples) >= max_train_samples:
                    break
            if max_train_samples and len(self.samples) >= max_train_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
