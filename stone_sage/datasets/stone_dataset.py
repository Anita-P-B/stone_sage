import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class StoneDataset(Dataset):
    def __init__(self, dataframe, target_column, debug = False):
        self.features = dataframe.drop(columns=[target_column, 'partition']).values.astype('float64')
        self.targets = dataframe[target_column].values.astype('float64').reshape(-1, 1)

        self.debug = debug
        self.log_features, self.mean_features, self.std_features = self.normalize(self.features)
        self.log_targets, self.mean_targets, self.std_targets = self.normalize(self.targets)

    def normalize(self, y):
        mean = np.mean(y, axis=0)
        std = np.std(y, axis=0)

        norm_y = (y - mean) / std

        if self.debug:
            # Reconstruct for sanity check
            y_reconstructed = norm_y * std + mean
            abs_diff = np.abs(y - y_reconstructed)
            rel_diff = abs_diff / (y + 1e-6) * 100  # Avoid division by zero

            print(f"Max absolute error: {abs_diff.max():.6f}")
            print(f"Max relative error: {rel_diff.max():.6f}%")
            print(f"Mean relative error: {rel_diff.mean():.6f}%")

            # Plot reconstruction check

            plt.figure(figsize=(6, 6))
            plt.scatter(y, y_reconstructed, alpha=0.5)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal')
            plt.xlabel("Original Target")
            plt.ylabel("Reconstructed Target")
            plt.title("Linear Normalization Sanity Check")
            plt.legend()
            plt.grid(True)
            plt.show()

        return norm_y.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)

    def log_normalization(self, y):
        epsilon = 1e-6
        log_y = np.log(y + epsilon)
        mean_log = np.mean(log_y, axis = 0)
        std_log = np.std(log_y, axis =0)
        norm_y = (log_y - mean_log) / std_log
        if self.debug :
            y_log_reconstructed = norm_y * std_log + mean_log
            y_reconstructed = np.exp(y_log_reconstructed) - epsilon
            # Step 6: Compare original and reconstructed
            abs_diff = np.abs(y - y_reconstructed)
            rel_diff = abs_diff / (y + epsilon) * 100  # % relative error

            # Display summary
            print(f"Max absolute error: {abs_diff.max():.6f}")
            print(f"Max relative error: {rel_diff.max():.6f}%")
            print(f"Mean relative error: {rel_diff.mean():.6f}%")

            # Optional: plot comparison


            plt.figure(figsize=(6, 6))
            plt.scatter(y, y_reconstructed, alpha=0.5)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal')
            plt.xlabel("Original Target")
            plt.ylabel("Reconstructed Target")
            plt.title("Log-Normalization Sanity Check")
            plt.legend()
            plt.grid(True)
            plt.show()
        return norm_y.astype(np.float32), mean_log.astype(np.float32), std_log.astype(np.float32)
    def __len__(self):
        return len(self.log_features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.log_features[idx], dtype=torch.float32),
            torch.tensor(self.log_targets[idx], dtype=torch.float32)
        )

    @property
    def input_dim(self):
        return self.log_features.shape[1]
