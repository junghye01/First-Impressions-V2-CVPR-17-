import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(
        self,
        audio_input_size: int,
        audio_hidden_size: int,
        audio_num_layers: int,
    ):
        super().__init__()

        self.audio_fc = nn.Linear(audio_input_size, audio_hidden_size)
        self.audio_LSTM = nn.LSTM(
            audio_hidden_size, audio_hidden_size, audio_num_layers, batch_first=True
        )

    def _get_audio_feature(self, audio: torch.Tensor) -> torch.Tensor:
        audio_feature = self.audio_fc(audio)
        audio_out, _ = self.audio_LSTM(audio_feature)
        return audio_out[:, -1]

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        audio_out = self._get_audio_feature(audio)
        return audio_out.flatten(start_dim=1)
