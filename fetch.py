import os
import random
from glob import glob
import torch
import numpy as np
import pandas as pd
from torchvision.io import read_image


def video_fetcher(video_path:str,frame_count:int=30,seed=None):
    if seed is not None:
        random.seed(seed)
    #frame_list = sorted(glob(os.path.join(video_path,'*.jpg')))
    frame_list=glob(os.path.join(video_path,'*jpg')) # sort는 전처리 단계에서
    # frame을 frame_count씩 나눔
    frames_split_with_interval = np.array_split(frame_list, frame_count)

    #각 덩어리에서 한 개씩 선택
    sampled_frames = [random.choice(f) for f in frames_split_with_interval]
    
    # 샘플링된 프레임들을 read_image함수를 이용해 이미지 텐서로 읽고 image_tensor_list에 저장
    image_tensor_list = [read_image(png_file) for png_file in sampled_frames]

    # 시간 차원을 따라 스택
    time_image_tensor = torch.stack(image_tensor_list, dim=0)
    #0에서 1까지 숫자로 정규화 
    return time_image_tensor / 255.0


n=10
video_dir='/home/irteam/junghye-dcloud-dir/chalearn_data/Completed'
annotation_df=pd.read_csv('/home/irteam/junghye-dcloud-dir/chalearn_data/Labels/annotation_training.csv')
for index in range(n):
    video_name='--Ymqszjv54.004'
    
    video_path=os.path.join(video_dir,video_name)
    frames=os.listdir(video_path)
    if frames is None:
        print(video_path,'no frames here')
    else:
        video_tensor=video_fetcher(video_path,frame_count=30)

print(video_tensor.float())
    