import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class HousePriceModel(nn.Module):
    def __init__(self, in_features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(in_features, 96),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.ReLU())
    def forward(self, x):
        return self.model(x)
