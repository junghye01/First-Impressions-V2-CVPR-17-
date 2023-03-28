import glob
import cv2 as cv
import os
import pandas as pd
import pickle
from collections import Counter

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

#data=pd.read_csv('/home/irteam/junghye-dcloud-dir/chalearn/data/labels.csv',index_col=0)
#print(data.columns)
#print(data.index)

#test=['a','b']+['c']
#print(test)

#print(len(all_dir2))

eth_gender_file=pd.read_csv('/home/irteam/junghye-dcloud-dir/chalearn_data/data/eth_gender_annotations_dev.csv')

completed_list=os.listdir('/home/irteam/junghye-dcloud-dir/chalearn_data/Completed')
for i in range(len(eth_gender_file)):
    lst=[]
    lst=list(eth_gender_file.iloc[i][0].split(';'))
    if lst[0][:-4] in completed_list:
        print(lst[0][:-4],'exists')
    print(lst)
    if i==3: break

#print(len(eth_gender_file))

ground_truth_path='/home/irteam/junghye-dcloud-dir/chalearn_data/data/annotation_training.pkl'
ground_truth_path2='/home/irteam/junghye-dcloud-dir/chalearn_data/data/annotation_validation.pkl'

ground_truth_f=pd.read_pickle(ground_truth_path)

print(ground_truth_f.keys())
print(type(ground_truth_f))

ground_truth_path='/home/irteam/junghye-dcloud-dir/chalearn_data/data/annotation_*.pkl'
ground_truth_path=ground_truth_path.replace('*','opip')
print(ground_truth_path)