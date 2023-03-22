import glob
import os
from pathlib import Path



# File list in 'data' and 'Completed'
all_dir=glob.glob('/home/irteam/junghye-dcloud-dir/chalearn/data/*.mp4',recursive=True)
all_dir2=glob.glob('/home/irteam/junghye-dcloud-dir/chalearn/Completed/*',recursive=True)

# Write mp4_list.txt
cnt=0
num=0
f=open('/home/irteam/junghye-dcloud-dir/chalearn/mp4_list_0.txt','w')
f.truncate()
for files in all_dir:
    if files in all_dir and files[:-4] not in all_dir:
        cnt+=1
        
        # 100개 단위로 리스트 생성
        if cnt%100==0:
            num=+1
            if os.path.isfile('/home/irteam/junghye-dcloud-dir/chalearn'+'/mp4_list_'+str(num)+'.txt'):
                Path('/home/irteam/junghye-dcloud-dir/chalearn'+'/mp4_list_'+str(num)+'.txt').touch()
            f=open('/home/irteam/junghye-dcloud-dir/chalearn'+'/mp4_list_'+str(num)+'.txt','w')
            f.truncate()
        f.write(files+'\n')
f.close()
