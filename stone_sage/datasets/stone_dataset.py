import torch
from torch.utils.data import Dataset

class StoneDataset(Dataset):
    def __init__(self, dataframe, target_column, mean=None, std=None, target_mean= None, target_std= None):
        self.features = dataframe.drop(columns=[target_column, 'partition']).values.astype('float32')
        self.targets = dataframe[target_column].values.astype('float32').reshape(-1, 1)

        # Normalize if mean and std are provided
        if mean is not None and std is not None:
            self.features = (self.features - mean) / std
        if target_mean is not None and target_std is not None:
            self.targets = (self.targets - target_mean) / target_std

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.targets[idx])

    @property
    def input_dim(self):
        return self.features.shape[1]
