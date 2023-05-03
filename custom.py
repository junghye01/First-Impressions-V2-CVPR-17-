import pandas as pd
import torch
import os
import random as rd
from fetch import video_fetcher

## 여기에 target data반환도 추가해야함 
class MultiModaldataset(torch.utils.data.Dataset):
    def __init__(self,video_dir:str,annotation_path:str,frame_count:int=30,seed:int=0,video_transform=None):
        super().__init__

        self.video_dir=video_dir # 전처리된 비디오 경로
        self.annotation_df=pd.read_csv(annotation_path)
        self.target_list=['extraversion','neuroticism','agreeableness','conscientiousness','interview','openness']
        self.seed=seed
        self.frame_count=frame_count
        self.video_transform=video_transform
    


    def __len__(self) -> int:
        return(self.annotation_df.shape[0])
    

    def _get_target_and_meta_data(self,idx:int) -> dict:
        target_data=[]

        for target in self.target_list: # idx번째 비디오의 target값 저장
            target_data.append(self.annotation_df.iloc[idx][target])

        target_data=torch.tensor(target_data).float() # target값 텐서화 -> float 변환

        data_item=self.annotation_df.iloc[idx]

        return {
            "video_name":data_item['video_name'],
            "youtube_id":data_item['YouTubeID'],
            "ethnicity":data_item['Ethnicity'],
            "gender":data_item['Gender'],
            'target_data':target_data,
        }

        
    def _get_video(self,idx:int):
        video_name=self.annotation_df.iloc[idx]['video_name'][:-4]
        video_path=os.path.join(self.video_dir,video_name)
        video_tensor=video_fetcher(video_path,self.frame_count,self.seed)
        return video_tensor
        
        # 샘플링 된 프레임을 텐서 형태로 저장

        
    

    def __getitem__(self,idx): # idx 번째 행의 비디오 가져오기 
        target_and_meta_data=self._get_target_and_meta_data(idx)

        video_tensor=self._get_video(idx)
        if self.video_transform is not None:
            video_tensor=self.video_transform(video_tensor)

        input_data={
            'video':video_tensor.float() # float형태로 변환
        }
        return input_data,target_and_meta_data