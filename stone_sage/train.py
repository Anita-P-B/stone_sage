import torch
from torch import nn
from tqdm import tqdm
from stone_sage.utils.utils import save_checkpoint, save_evaluation_summary
import matplotlib.pyplot as plt
import csv
import os

class Trainer:
    def __init__(self, model, optimizer,loss, train_loader, val_loader, run_dir,scheduler = None, device=None):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = loss  # MAE for regression
        self.scheduler = scheduler
        self.best_loss =  float("inf")
        # training metrics
        self.history = []

        # set model
        self.model.to(self.device)

    def save_history_and_plot(self):
        # get field names dynamically
        fieldnames = list(self.history[0].keys())

        # Save raw history as CSV
        csv_path = os.path.join(self.run_dir, "training_history.csv")

        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)
        print(f"📄 Training history saved to: {csv_path}")

        # Save loss curve plot
        epochs = [entry["epoch"] for entry in self.history]
        train_losses = [entry["train_loss"] for entry in self.history]
        val_losses = [entry["val_loss"] for entry in self.history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss", marker="o")
        plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.run_dir, "training_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"📈 history curve plot saved to: {plot_path}")


    def train(self, num_epochs=10):
        for epoch in range(1, num_epochs + 1):
            print(f"\n🧪 Epoch {epoch}/{num_epochs}")
            # train step
            train_loss = self._train_one_epoch(epoch)
            print(f"📉 Train MAE: {train_loss:.4f}")
            # validation step
            val_loss = self._validate_one_epoch(epoch)
            print(f"🧾 Val MAE: {val_loss:.4f}")

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:  {val_loss:.4f}")

            # Save best checkpoint
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                if not self.run_dir:
                    raise ValueError("⚠️ Cannot save checkpoint: `checkpoint_path` is not set or is empty.")

                save_checkpoint(
                    run_dir=self.run_dir,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss
                )
            # 🔥 Update training history
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
        self.save_history_and_plot()

        final_train_mae = self.history[-1]["train_loss"]
        final_val_mae = self.history[-1]["val_loss"]

        save_evaluation_summary(self.run_dir, {
            "train_mae": final_train_mae,
            "val_mae": final_val_mae,
        })


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