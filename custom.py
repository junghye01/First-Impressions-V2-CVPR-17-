import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self,annotations_file,video_dir,transform=None,target_transform=None):
        self.video_labels=pd.read_csv(annotations_file)
        self.video_dir=video_dir
        self.transform=transform
        self.target_transform=target_transform

 

