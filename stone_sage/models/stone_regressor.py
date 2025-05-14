import torch
import torch.nn as nn

class StoneRegressor(nn.Module):
    def __init__(self, input_dim=8, hidden_dims=[32, 16], dropout=0.1):
        super(StoneRegressor, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))  # Output layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)