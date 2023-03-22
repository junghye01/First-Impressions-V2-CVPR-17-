# 업데이트 된 것만 읽어서 frame으로 변환하는 코드
import cv2
import os

out_path='/home/irteam/junghye-dcloud-dir/chalearn/Updated_mp4'
mp4_list='/home/irteam/junghye-dcloud-dir/chalearn/mp4_list_1.txt'


with open(mp4_list,'r',encoding='utf8') as updated_list:
    for file in updated_list:
        file=file.replace('\n','')

        cap=cv2.VideoCapture(file)
        path_to_save=os.path.join(out_path,file[-19:-4])
        
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
                if(current_frame %10 ==0):
                    name='frame'+str(current_frame)+'.jpg'
                    print(f'Creating:{name}')
                    cv2.imwrite(os.path.join(path_to_save,name),frame)

            else:
                break

        cap.release()
        print('done')




