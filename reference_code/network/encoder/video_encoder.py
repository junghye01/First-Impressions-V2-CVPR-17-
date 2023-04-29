import torch
import torch.nn as nn

from network.layers import ConvBlock2D


class VideoEncoder(nn.Module):
    def __init__(
        self,
        video_channels: list,
        video_input_size: int,
        video_hidden_size: int,
        video_num_layers: int,
    ):
        super().__init__()
        self.video_conv = self._make_conv_layers(video_channels)
        self.video_lstm = nn.LSTM(
            video_input_size, video_hidden_size, video_num_layers, batch_first=True
        )

    def _make_conv_layers(self, channels: list):
        layers = []
        for channel in channels:
            layers.append(ConvBlock2D(channel[0], channel[1]))
        return nn.Sequential(*layers)

    def _get_video_feature(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.size()
        video = video.reshape(B * T, C, H, W)
        video_feature = self.video_conv(video)
        video_feature = video_feature.reshape(B, T, -1)
        video_out, _ = self.video_lstm(video_feature)
        return video_out[:, -1]

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        video_out = self._get_video_feature(video)
        return video_out.flatten(start_dim=1)
