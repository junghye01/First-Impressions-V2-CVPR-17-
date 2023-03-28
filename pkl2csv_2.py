import pickle
import pandas as pd
from pathlib import Path
import os
import csv

# 
index=['training','test','validation']

# true-value pkl 
ground_truth_path='/home/irteam/junghye-dcloud-dir/chalearn/data/annotation_*.pkl'

# eth_gender file path
eth_gender_path='/home/irteam/junghye-dcloud-dir/chalearn/data/eth_gender_annotations_dev.csv'

#result csv path
csv_save_path='/home/irteam/junghye-dcloud-dir/chalearn/data/annotation_*.csv'




for idx in index:
    # fix path
    csv_save_path=csv_save_path.replace('*',idx)
    if os.path.isfile(csv_save_path):
        Path(csv_save_path).touch()

    with open(ground_truth_path.replace('*',idx),'rb')as f:
        u=pickle._Unpickler(f)
        u.encoding='latin1'
        ground_truth_f=u.load()
    
    
    if idx=='test':
        eth_gender_file=pd.read_csv(eth_gender_path.replace('dev','test'))
    eth_gender_file=pd.read_csv(eth_gender_path)
    header=['video_name','YouTubeID','Ethnicity','Gender']+list(ground_truth_f.keys())
    
    with open(csv_save_path,'w',encoding='utf8') as csvfile:
        csvfile.truncate()

        writer=csv.writer(csvfile,delimiter=',',quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        lst=[]

        # eth_gender
        for idx in range(len(eth_gender_file)):
            lst=[]
            if idx==0:
                continue
            lst=list(eth_gender_file.iloc[idx][0].split(';'))
            video_name=lst[0]
            if video_name[0]=='-':
                video_name=video_name[1:]
            for label in list(ground_truth_f.keys()):
                lst.append(ground_truth_f[label][video_name])

            writer.writerow(lst)
    
    
    csvfile.close()

    



        


    

    




