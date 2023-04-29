import torch
import torch.nn as nn


class VidoeRegressor(nn.Module):
    def __init__(
        self,
        video_hidden_size: int,
        fusion_hidden_size: int,
        num_class: int,
    ):
        super().__init__()
        self._layer = self._make_layer(video_hidden_size, fusion_hidden_size, num_class)

    def _make_layer(self, feature_size: int, hidden_size: int, num_class: int):
        return nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_class),
        )

    def forward(self, multimodal_feature):
        out = self._layer(multimodal_feature)
        out = torch.sigmoid(out)
        return out
