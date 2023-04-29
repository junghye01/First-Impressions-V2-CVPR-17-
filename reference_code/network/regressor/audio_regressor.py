import torch
import torch.nn as nn


class AudioRegressor(nn.Module):
    def __init__(
        self,
        audio_hidden_size: int,
        num_class: int,
    ):
        super().__init__()
        self._layer = self._make_layer(audio_hidden_size, num_class)

    def _make_layer(self, feature_size: int, num_class: int):
        return nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.Linear(feature_size, num_class),
        )

    def forward(self, multimodal_feature):
        out = self._layer(multimodal_feature)
        out = torch.sigmoid(out)
        return out
