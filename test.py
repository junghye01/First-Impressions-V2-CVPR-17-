import glob
import cv2 as cv
import os
import pandas as pd

all_dir=glob.glob('/home/irteam/junghye-dcloud-dir/chalearn/data/*.mp4',recursive=True)
all_dir2=glob.glob('/home/irteam/junghye-dcloud-dir/chalearn/Completed/*',recursive=True)

print(len(all_dir))
sr='/home/irteam/junghye-dcloud-dir/chalearn/data/4F-WBGPXiqQ.001.mp4'

#print(sr[-19:-4])

#print(cv.__version__)

#test = cv.VideoCapture('/home/irteam/junghye-dcloud-dir/chalearn/data/_mtiuHyOFXg.001.mp4')
#print(test.isOpened())
#print(cv.getBuildInformation())

#update_mp4_list=os.listdir('/home/irteam/junghye-dcloud-dir/chalearn/Updated_mp4')
#print(len(update_mp4_list))

data=pd.read_csv('/home/irteam/junghye-dcloud-dir/chalearn/data/labels.csv',index_col=0)
print(data.columns)
print(data.index)
