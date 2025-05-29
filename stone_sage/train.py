import torch
from torch import nn
from tqdm import tqdm
from stone_sage.utils.utils import save_checkpoint, save_evaluation_summary, relative_mae_percentage, \
    build_epoch_metrics, extract_val_loss, smape
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.ticker as mticker


class Trainer:
    def __init__(self, model, optimizer, loss, train_loader, val_loader, run_dir,
                 scheduler=None, device=None,
                 n_best_checkpoints=3, target_mean = None, target_std = None,
                 debug = False):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = loss  # MAE for regression
        self.metrics = {
            "rel_mae_percent": relative_mae_percentage,
            "smape": smape
            # add more here if needed
        }
        self.scheduler = scheduler
        self.best_loss = float("inf")
        # training metrics
        self.history = []

        # set model
        self.model.to(self.device)
        self.n_best_checkpoints = n_best_checkpoints

        self.target_mean = target_mean
        self.target_std = target_std

        self.debug = debug

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
        train_rel_mae = [entry["train_rel_mae"] for entry in self.history]
        val_rel_mae = [entry["val_rel_mae"] for entry in self.history]
        train_smape = [entry["train_smape"] for entry in self.history]
        val_smape = [entry["val_smape"] for entry in self.history]

        # plot
        fig, axs = plt.subplots(1, 3, figsize=(20, 8), sharex=True)
        fig.subplots_adjust(hspace=0.4)  # Space between plots

        # 🔹 Subplot 1: Loss
        axs[0].plot(epochs, train_losses, label="Train Loss", marker="o")
        axs[0].plot(epochs, val_losses, label="Validation Loss", marker="o")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Training & Validation Loss")
        axs[0].legend()
        axs[0].grid(True)

        # 🔸 Subplot 2: Relative MAE %
        axs[1].plot(epochs, train_rel_mae, label="Train Rel MAE (%)", marker="s")
        axs[1].plot(epochs, val_rel_mae, label="Validation Rel MAE (%)", marker="s")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Relative MAE (%)")
        axs[1].set_title("Relative MAE Percentage")
        axs[1].legend()
        axs[1].yaxis.set_major_locator(mticker.MultipleLocator(5))
        axs[1].grid(True, which="major", axis="y", linestyle="--", alpha=0.7)
        axs[1].axhline(5, color="red", linestyle="--", linewidth=1.5, label="Goal: 5%")
        axs[1].legend()

        # 🔸 Subplot 2: Relative MAE %
        axs[2].plot(epochs, train_smape, label="Train Rel MAE (%)", marker="s")
        axs[2].plot(epochs, val_smape, label="Validation Rel MAE (%)", marker="s")
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("smape (%)")
        axs[2].set_title("Symmetric mean absolute percentage")
        axs[2].legend()
        axs[2].yaxis.set_major_locator(mticker.MultipleLocator(5))
        axs[2].grid(True, which="major", axis="y", linestyle="--", alpha=0.7)
        axs[2].axhline(5, color="red", linestyle="--", linewidth=1.5, label="Goal: 5%")
        axs[2].legend()

        # Save the figure
        plot_path = os.path.join(self.run_dir, "training_curves.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 training curves saved to: {plot_path}")

    def train(self, num_epochs=10):
        for epoch in range(1, num_epochs + 1):
            print(f"\n🧪 Epoch {epoch}/{num_epochs}")
            # train step
            train_loss, train_preds, train_targets = self._train_one_epoch(epoch, num_epochs)
            # Calculate metrics
            train_mae = mean_absolute_error(train_targets, train_preds)
            train_rel_mae = self.metrics["rel_mae_percent"](train_targets, train_preds,
                                                            self.target_mean, self.target_std)
            train_smape = self.metrics["smape"](train_targets, train_preds,
                                                            self.target_mean, self.target_std)
            if self.debug:
                rel_mae_fn = self.metrics["rel_mae_percent"]
                train_rel_mae = rel_mae_fn(train_targets, train_preds, self.target_mean, self.target_std)
            print(
                f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | "
                f"Train MAE: {train_mae:.4f} | Train smape: {train_smape:.2f}%"
            )
            # validation step
            val_loss, val_preds, val_targets = self._validate_one_epoch(epoch)

            # calculate validation metrics
            val_mae = mean_absolute_error(val_targets, val_preds)
            val_rel_mae = self.metrics["rel_mae_percent"](val_targets, val_preds,
                                                          self.target_mean, self.target_std)
            val_smape = self.metrics["smape"](val_targets, val_preds,
                                                          self.target_mean, self.target_std)

            print(
                f"Epoch {epoch + 1} | "
                f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val RelMAE: {val_smape:.2f}%"
            )

            # train loos metrics
            losses = {"train_loss": train_loss, "val_loss": val_loss}
            metrics = {
                "train_mae": train_mae,
                "val_mae": val_mae,
                "train_rel_mae": train_rel_mae,
                "val_rel_mae": val_rel_mae,
                "train_smape": train_smape,
                "val_smape": val_smape
            }

            epoch_metrics = build_epoch_metrics(epoch, losses, metrics)
            # 🔥 Update training history
            self.history.append(epoch_metrics)

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
                    epoch_metrics=epoch_metrics
                )
                # keep only the best n checkpoints
                checkpoint_files = [f for f in os.listdir(self.run_dir) if f.endswith(".pt")]
                sorted_checkpoints = sorted(checkpoint_files, key=extract_val_loss)
                if len(checkpoint_files) > self.n_best_checkpoints:
                    os.remove(os.path.join(self.run_dir,  sorted_checkpoints[-1]))

        self.save_history_and_plot()

        best_idx = np.argmin([entry["val_mae"] for entry in self.history])
        final_metrics = self.history[best_idx]
        save_evaluation_summary(self.run_dir, final_metrics)

    def _train_one_epoch(self, epoch, total_epochs):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for inputs, targets in tqdm(self.train_loader, desc=f"Training {epoch}"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            predictions =  self.model(inputs)

            loss = self.loss_fn(predictions, targets)
            loss.backward()
            self.optimizer.step()

            if self.debug:
                if epoch % 50 == 0 or epoch ==total_epochs:
                    self.model.eval()
                    with torch.no_grad():
                        pred = self.model(inputs)
                        print(f"[Epoch {epoch}]")
                        print("Pred:", (pred * self.target_std + self.target_mean).cpu().numpy().flatten())
                        print("True:", (targets * self.target_std + self.target_mean).cpu().numpy().flatten())
                        self.model.train()

            total_loss += loss.item() * inputs.size(0)
            all_preds.append(predictions.detach().cpu().numpy())  # converts to numpy for calculations
            all_targets.append(targets.detach().cpu().numpy())

        epoch_loss = total_loss / len(self.train_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        return epoch_loss, all_preds, all_targets

    def _validate_one_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc=f"Validating {epoch}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                predictions =  self.model(inputs)
                loss = self.loss_fn(predictions, targets)

                total_loss += loss.item() * inputs.size(0)
                all_preds.append(predictions.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())

        epoch_loss = total_loss / len(self.val_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        return epoch_loss, all_preds, all_targets
