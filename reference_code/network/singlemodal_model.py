import torch
import torch.nn as nn


class SingleModalModel(nn.Module):
    def __init__(self, encoder: nn.Module, regressor: nn.Module):
        super().__init__()
        self._encoder = encoder
        self._regressor = regressor

    def forward(self, x: torch.Tensor):
        feature = self._encoder(x)
        out = self._regressor(feature)
        return feature, out
