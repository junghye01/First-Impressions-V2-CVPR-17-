# 업데이트 된 것만 읽어서 frame으로 변환하는 코드
#https://silly-tartan-5b8.notion.site/0321-chalearn-a543e3c0e93a4c65b1eb05736fdd3daf
import cv2
import os
import shutil

out_path='/home/irteam/junghye-dcloud-dir/chalearn/Updated_mp4'
mp4_list='/home/irteam/junghye-dcloud-dir/chalearn/resample_list.txt'


with open(mp4_list,'r',encoding='utf8') as updated_list:
    for file in updated_list:
        file=file.replace('\n','')


        # 폴더 삭제
        #if os.path.exists(file):
         #   shutil.rmtree(file)
        cap=cv2.VideoCapture(file)
        path_to_save=os.path.join(out_path,file[-19:-4])
        print(path_to_save)
        # mkdir
        if not(os.path.isdir(path_to_save)):
            os.mkdir(path_to_save)
        
        current_frame=1

        if(cap.isOpened()==False):
            print('Cap is not open')

        #cap open successfully
        while(cap.isOpened()):

            #capture each frame
            ret,frame=cap.read()

            if(ret==True):
                # total_frames
                current_frame+=1

                # sampling
               
                name='frame'+str(current_frame)+'.jpg'
                print(f'Creating:{name}')
                cv2.imwrite(os.path.join(path_to_save,name),frame)

            else:
                break

        cap.release()
        print('done')




