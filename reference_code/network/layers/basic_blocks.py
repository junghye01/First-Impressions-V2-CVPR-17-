import torch
import torch.nn as nn


class ConvBlock2D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1
    ) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
