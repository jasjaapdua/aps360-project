import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch import nn

from config import config
from visualisation.plotter import TrainingPlotter
from training.persistence import save_model


class Trainer:

    def __init__(self, model, dataset):

        self.model = model
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)

        self.loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=max(config.num_workers, 0),
            pin_memory=(config.device == "cuda")
        )

        self.device = torch.device(config.device)

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )

        self.criterion = nn.CrossEntropyLoss()

        self.loss_history = []

    def train(self):

        for epoch in range(config.epochs):

            total_loss = 0
            steps_run = 0

            self.model.train()

            for step, (x, y) in enumerate(self.loader, start=1):

                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(x)

                loss = self.criterion(logits, y)

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()
                steps_run = step
                if config.max_steps_per_epoch > 0 and step >= config.max_steps_per_epoch:
                    break

            avg_loss = total_loss / max(steps_run, 1)

            self.loss_history.append(avg_loss)

            print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f}")

        # save model
        save_model(self.model, "checkpoints/lstm_model.pt")

        # plot training curve
        TrainingPlotter.plot_loss(
            self.loss_history,
            save_path="plots/training_loss.png"
        )
