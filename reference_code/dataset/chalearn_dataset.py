import os
import torch
import pandas as pd
from .fetcher import video_fetcher, audio_fetcher, bert_english_text_fetcher


class Chalearn_MultiModalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        video_dir: str,
        audio_dir: str,
        text_dir: str,
        annotation_dir: str,
        tokenizer,
        target_list: list = None,
        frame_count: int = 30,
        text_max_len: int = 512,
        seed: int = 0,
        video_transform=None,
        target_transform=None,
    ) -> None:
        super().__init__()

        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.frame_count = frame_count
        self.text_max_len = text_max_len

        self.target_list = target_list
        self.tokenizer = tokenizer
        self.text_df = self._set_text_df(text_dir)

        self.seed = seed
        self.annotation_df = self._set_annotation_df(annotation_dir)

        self.video_transform = video_transform
        self.target_transform = target_transform

    def _set_annotation_df(self, annotation_dir: str) -> pd.DataFrame:
        return pd.read_csv(annotation_dir)

    def _set_text_df(self, text_dir: str) -> pd.DataFrame:
        return pd.read_csv(text_dir + "transcription.csv")

    def __len__(self) -> int:
        return (self.annotation_df.shape)[0]

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

    def _get_audio(self, index: int) -> torch.Tensor:
        file_name = self.annotation_df.iloc[index]["video_name"]
        audio_path = os.path.join(self.audio_dir, file_name + ".wav_st.csv")
        audio_tensor = audio_fetcher(audio_path, self.frame_count)
        return audio_tensor

    def _get_text(self, index: int) -> torch.Tensor:
        video_name = self.annotation_df.iloc[index]["video_name"]
        scripts = self.text_df.loc[self.text_df["video_name"] == video_name]
        scripts = scripts["transcription"].iloc[0]
        text_tensor = bert_english_text_fetcher(scripts, self.tokenizer, self.text_max_len)
        return text_tensor

    def __getitem__(self, index):
        target_and_meta_data = self._get_target_and_meta_data(index)

        video_tensor = self._get_video(index)
        if self.video_transform is not None:
            video_tensor = self.video_transform(video_tensor)
        audio_tensor = self._get_audio(index)
        text_tensor = self._get_text(index)

        input_data = {
            "video": video_tensor.float(),
            "audio": audio_tensor.float(),
            "text": text_tensor,
        }
        return input_data, target_and_meta_data
