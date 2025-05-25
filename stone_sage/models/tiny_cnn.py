# model/tiny_cnn.py

import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (1, input_dim)),
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)
