import torch
from torch.utils.data import Dataset

class StoneDataset(Dataset):
    def __init__(self, dataframe, target_column):
        self.features = dataframe.drop(columns=[target_column]).values.astype('float32')
        self.targets = dataframe[target_column].values.astype('float32').reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.targets[idx])
