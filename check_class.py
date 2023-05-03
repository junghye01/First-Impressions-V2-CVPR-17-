
import torch
import torchvision
from torch.utils.data import DataLoader
from train import MultiModaldataset

n=10
video_dir='/home/irteam/junghye-dcloud-dir/chalearn_data/Completed'
annotation_dir='/home/irteam/junghye-dcloud-dir/chalearn_data/Labels/annotation_training_new.csv'
frame_count=30 # 한 묶음당 프레임 수 30개
seed=10
train_dataset=MultiModaldataset(
    video_dir=video_dir,
    annotation_path=annotation_dir,
    frame_count=frame_count,
    video_transform=None
)
print('train_dataset')

for index in range(n):
    x,y=train_dataset.__getitem__(index)
    print(x['video'].shape,y['target_data'].shape)