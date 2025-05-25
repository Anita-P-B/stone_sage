import torch
import torch.nn as nn
from stone_sage.models.tiny_cnn import TinyCNN

class StoneRegressor(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(StoneRegressor, self).__init__()
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)

    @staticmethod
    def build(configs, input_dim):
        model_type = getattr(configs, "MODEL").lower()

        model_dispatch = {
            "mlp": StoneRegressor._build_mlp,
            "tiny_deep": StoneRegressor._build_tiny_deep,
            "wide_shallow": StoneRegressor._build_wide_shallow,
            "tiny_cnn": StoneRegressor._build_tiny_cnn
        }

        if model_type not in model_dispatch:
            raise ValueError(f"Unknown model type: {model_type}")

        return model_dispatch[model_type](configs, input_dim)

    @staticmethod
    def _build_mlp(configs, input_dim):
        layers = []
        prev_dim = input_dim
        for h in configs.HIDDEN_DIMS:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(configs.DROPOUT) if configs.DROPOUT > 0 else nn.Identity())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        return StoneRegressor(layers)

    @staticmethod
    def _build_tiny_deep(configs, input_dim):
        # More layers, smaller dims
        dims = [16, 16, 8]
        return StoneRegressor._from_dims(input_dim, dims, configs.DROPOUT)

    @staticmethod
    def _build_wide_shallow(configs, input_dim):
        # Fewer layers, wider dims
        dims = [128]
        return StoneRegressor._from_dims(input_dim, dims, configs.DROPOUT)

    @staticmethod
    def _from_dims(input_dim, hidden_dims, dropout):
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        return StoneRegressor(layers)

    @staticmethod
    def _build_tiny_cnn(configs, input_dim):
        return TinyCNN(input_dim, configs.DROPOUT)