import torch
import torch.nn as nn
from torch.nn.functional import dropout
from torchinfo import summary
import torch.nn.functional as F



class ShallowDeep(nn.Module):
    def __init__(self,input_dim, dropout = 0):
        super().__init__()
        self.shallow = nn.Linear(input_dim, 8)
        self.deep = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,8),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(16, 1)
    def forward(self, x):
        x_shallow = self.shallow(x)
        x_deep = self.deep(x)
        x_combined = torch.cat([x_shallow,x_deep], dim = 1)
        x_combined = self.output_layer(x_combined)
        return x_combined

