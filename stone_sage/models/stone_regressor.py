import torch
import torch.nn as nn

class StoneRegressor(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(StoneRegressor, self).__init__()
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)

    @staticmethod
    def build(configs, input_dim):
        model_type = getattr(configs, "MODEL").lower()

        if model_type == "mlp":
            return StoneRegressor._build_mlp(configs, input_dim)
        elif model_type == "tiny_deep":
            return StoneRegressor._build_tiny_deep(configs, input_dim)
        elif model_type == "wide_shallow":
            return StoneRegressor._build_wide_shallow(configs, input_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _build_mlp(configs, input_dim):
        layers = []
        prev_dim = input_dim
        for h in configs.HIDDEN_DIMS:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(configs.DROPOUT))
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
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        return StoneRegressor(layers)