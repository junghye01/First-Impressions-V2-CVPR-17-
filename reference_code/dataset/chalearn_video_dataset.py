import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset.fetcher import video_fetcher
from dataset.transform import Chalearn_VideoTransform


class Chalearn_VideoDataset(Dataset):
    def __init__(
        self,
        video_dir: str,
        label_dir: str,
        target_list: list,
        frame_count: int = 30,
        seed: int = 0,
        video_transform=None,
    ) -> None:
        super().__init__()

        self.video_dir = video_dir
        self.frame_count = frame_count
        self.target_list = target_list

        self.seed = seed
        self.annotation_df = self._set_annotation_df(label_dir)
        if video_transform == None:
            self.video_transform = Chalearn_VideoTransform()
        else:
            self.video_transform = video_transform

    def _set_annotation_df(self, label_dir: str) -> pd.DataFrame:
        return pd.read_csv(label_dir)

    def __len__(self) -> int:
        return len(self.annotation_df)

    def _get_target_and_meta_data(self, index: int) -> dict:
        target_data = []
        for target in self.target_list:
            target_data.append(self.annotation_df.iloc[index][target])
        target_data = torch.tensor(target_data).float()

        data_item = self.annotation_df.iloc[index]
        return {
            "video_name": data_item["video_name"],
            "youtube_id": data_item["youtube_id"],
            "ethnicity": data_item["ethnicity"],  # (Asian, Caucasian, African-American)
            "ethnicity_label": data_item["ethnicity_label"],  # (0, 1, 2)
            "gender": data_item["gender"],  # (M, F)
            "gender_label": data_item["gender_label"],  # (0, 1)
            "target_data": target_data,
        }

    def _get_video(self, index: int) -> torch.Tensor:
        file_name = self.annotation_df.iloc[index]["video_name"]
        video_path = os.path.join(self.video_dir, file_name)
        video_tensor = video_fetcher(video_path, self.frame_count, self.seed)
        return video_tensor

    def __getitem__(self, index):
        target_and_meta_data = self._get_target_and_meta_data(index)

        video_tensor = self._get_video(index)
        if self.video_transform is not None:
            video_tensor = self.video_transform(video_tensor)

        input_data = {
            "video": video_tensor.float(),
        }
        return input_data, target_and_meta_data
