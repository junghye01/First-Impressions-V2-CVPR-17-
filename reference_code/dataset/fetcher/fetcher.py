import os
import random
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image


def video_fetcher(video_file_path: str, frame_count: int = 30, seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
    frame_list = sorted(glob(os.path.join(video_file_path, "*.jpg")))
    frames_split_with_interval = np.array_split(frame_list, frame_count)
    sampled_frames = [random.choice(f) for f in frames_split_with_interval]

    image_tensor_list = [read_image(png_file) for png_file in sampled_frames]
    time_image_tensor = torch.stack(image_tensor_list, dim=0)
    return time_image_tensor / 255.0


def audio_fetcher(audio_file_path: str, frame_count: int = 30):
    audio_feat = pd.read_csv(audio_file_path, header=None).to_numpy()
    # padding
    if audio_feat.shape[0] < frame_count:
        padding = audio_feat[np.newaxis, -1, :].repeat(frame_count - audio_feat.shape[0], axis=0)
        audio_feat = np.concatenate((audio_feat, padding))
    # slice
    elif audio_feat.shape[0] > frame_count:
        audio_feat = audio_feat[:frame_count,]

    audio_tensor = torch.from_numpy(audio_feat)
    return audio_tensor


def bert_english_text_fetcher(
    script, tokenizer, max_len=512
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokenized_script = ["[CLS]"] + tokenizer.tokenize(script) + ["[SEP]"]
    tokenized_script = tokenizer.convert_tokens_to_ids(tokenized_script)
    tokenized_script = torch.tensor(tokenized_script, dtype=torch.long)

    segment_id = torch.ones_like(tokenized_script)
    mask = torch.ones_like(tokenized_script)

    def _pad(tensor: torch.Tensor, max_len) -> torch.Tensor:
        pad = torch.zeros(max_len - len(tensor), dtype=torch.long)
        return torch.cat([tensor, pad])

    tokenized_script = _pad(tokenized_script, max_len)
    segment_id = _pad(segment_id, max_len)
    mask = _pad(mask, max_len)

    return tokenized_script, segment_id, mask
