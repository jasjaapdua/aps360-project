import torch
from torch.utils.data import DataLoader
from torch import nn
from config import config


class Trainer:
    """
    Handles model training.
    """

    def __init__(self, model, dataset):
        self.model = model

        self.loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        self.device = torch.device(config.device)

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        self.criterion = nn.CrossEntropyLoss()

    def train(self):

        for epoch in range(config.epochs):

            total_loss = 0

            self.model.train()

            for x, y in self.loader:

                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(x)

                loss = self.criterion(logits, y)

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.loader)

            print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f}")
