import pandas as pd
import os
import random as rd
from load import *

class MultiModaldataset:
    def __init__(self):
        self.fps=10
        self.labels=['interview']
        

        training=False
        testing=False
        validation=False

    def __len__(self):
        len(self.annotation)

    def __getitem__(self): # idx 번째 행의 비디오 가져오기 
        
        
        video_path=os.path.join(self.video_dir,)
        idx=rd.randint(0,len(self.annotation))

        video_name=self.annotation.iloc[idx]['video_name']
        video_data=os.path.join(self.video_dir,video_name)

        #frame sampling by random