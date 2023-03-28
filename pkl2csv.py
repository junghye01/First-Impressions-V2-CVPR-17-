import pickle
import pandas as pd
from pathlib import Path
import os
import csv

# 
index=['training','validation','test']

# true-value pkl 
ground_truth_path='/home/irteam/junghye-dcloud-dir/chalearn_data/data/annotation_*.pkl'


# eth_gender file path
eth_gender_1=pd.read_csv('/home/irteam/junghye-dcloud-dir/chalearn_data/data/eth_gender_annotations_dev.csv',sep=';')
eth_gender_2=pd.read_csv('/home/irteam/junghye-dcloud-dir/chalearn_data/data/eth_gender_annotations_test.csv',sep=';')
eth_gender_file=pd.concat([eth_gender_1,eth_gender_2]).reset_index(drop=True)


#result csv path
csv_save_path='/home/irteam/junghye-dcloud-dir/chalearn_data/data/annotation_*.csv'




for idx in index:
   
    ground_truth_path='/home/irteam/junghye-dcloud-dir/chalearn_data/data/annotation_*.pkl'
    csv_save_path='/home/irteam/junghye-dcloud-dir/chalearn_data/data/annotation_*.csv'
    # path to save
    csv_save_path=csv_save_path.replace('*',idx)
    print(csv_save_path)
    if os.path.isfile(csv_save_path)==False:
        Path(csv_save_path).touch()

    ground_truth_path=ground_truth_path.replace('*',idx)
    # read annotation pkl
    ground_truth_f=pd.read_pickle(ground_truth_path)
    

    #csv header
    header=['video_name','YouTubeID','Ethnicity','Gender']+list(ground_truth_f.keys())
    # df
    result_df=pd.DataFrame(columns=header)
    
    video_name_lst=ground_truth_f['interview'].keys()

    for v in video_name_lst:
        # eth_gender_file에서 찾기
        tmp=eth_gender_file[eth_gender_file['VideoName']==v].reset_index(drop=True)
        video_name=tmp.loc[0]['VideoName']
        youtube_id=tmp.loc[0]['YouTubeID']
        ethnicity=tmp.loc[0]['Ethnicity']
        gender=tmp.loc[0]['Gender']
        #personality
        extraversion=ground_truth_f['extraversion'][v]
        neuroticism=ground_truth_f['neuroticism'][v]
        
        agreeableness=ground_truth_f['agreeableness'][v]
        
        conscientiousness=ground_truth_f['conscientiousness'][v]
        
        interview=ground_truth_f['interview'][v]
        openness=ground_truth_f['openness'][v]
        item=dict()
        item = {
            "video_name": video_name,
            "YouTubeID": youtube_id,
            "Ethnicity": ethnicity,
            
            "Gender": gender,
         
            "extraversion": extraversion,
            "neuroticism": neuroticism,
            "agreeableness": agreeableness,
            "conscientiousness": conscientiousness,
            "interview": interview,
            "openness": openness,
        }

        result_df=result_df.append(item,ignore_index=True)
    result_df.to_csv(csv_save_path)
    print(csv_save_path,'done')

        

        
   

    



        


    

    




