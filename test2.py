import pickle
import pandas as pd
import os

# read pickle file
with open('/home/irteam/junghye-dcloud-dir/chalearn/data/annotation_training.pkl','rb')as f:
    u=pickle._Unpickler(f)
    u.encoding='latin1'
    p=u.load()


# row
idx=p.keys()

# video name
video_name=p['interview'].keys()

data=dict()
for v in video_name:
    lst=[]
    lst.append(p['extraversion'][v])
    lst.append(p['neuroticism'][v])
    lst.append(p['agreeableness'][v])
    lst.append(p['conscientiousness'][v])
    lst.append(p['interview'][v])
    lst.append(p['openness'][v])
    data[v]=lst

df=pd.DataFrame(data,columns=list(video_name),index=list(idx))

df.to_csv('/home/irteam/junghye-dcloud-dir/chalearn/data'+'/labels.csv')





    
