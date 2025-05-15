import torch
from torch import nn
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer,loss, train_loader, val_loader=None, device=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = loss  # MAE for regression

        self.model.to(self.device)

    def train(self, num_epochs=10):
        for epoch in range(1, num_epochs + 1):
            print(f"\n🧪 Epoch {epoch}/{num_epochs}")
            train_loss = self._train_one_epoch(epoch)
            print(f"📉 Train MAE: {train_loss:.4f}")

            if self.val_loader:
                val_loss = self._validate_one_epoch(epoch)
                print(f"🧾 Val MAE: {val_loss:.4f}")

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for inputs, targets in tqdm(self.train_loader, desc=f"Training {epoch}"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(inputs)
            loss = self.loss_fn(predictions, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        return total_loss / len(self.train_loader.dataset)

    def _validate_one_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc=f"Validating {epoch}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                predictions = self.model(inputs)
                loss = self.loss_fn(predictions, targets)
                total_loss += loss.item() * inputs.size(0)

        return total_loss / len(self.val_loader.dataset)