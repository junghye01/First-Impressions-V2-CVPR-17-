import os
import sys
import cv2
from tqdm import tqdm

video_path = "E:\\first_impressions_v2_2017\\chalearn_mp4\\"
frame_path = "E:\\first_impressions_v2_2017\\tmp_frame\\"

for mp4_filename in os.listdir(video_path):
    input_filename = video_path + mp4_filename
    print(input_filename)
    output_folder_name = frame_path + mp4_filename
    os.makedirs(output_folder_name, exist_ok=True)
    print(output_folder_name)
    vidcap = cv2.VideoCapture(input_filename)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("frame_count =", frame_count)
    for frame_id in tqdm(range(0, frame_count)):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, image = vidcap.read()
        cv2.imwrite(output_folder_name + "\\%06d.png" % frame_id, image)
        cv2.waitKey(0)
    sys.exit()
