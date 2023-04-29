import os
from torchvision.io import read_image
from torchvision.transforms import Resize
import numpy as np
import sys
from tqdm import tqdm

r = Resize((300, 300))

frames_filepath = "F:/chalearn_first_impression_dataset/frames/"
npy_filepath = "F:/chalearn_first_impression_dataset/frames_npy_300/"
f = os.listdir(frames_filepath)
f.sort()
for mp4_name in f:
    v1 = frames_filepath + mp4_name + "/"
    v2 = npy_filepath + mp4_name + "/"
    os.makedirs(v2, exist_ok=True)
    print(v1)
    for jpg in tqdm(os.listdir(v1)):
        t_np = r(read_image(v1 + jpg)).numpy()
        np.save(v2 + jpg + ".npy", t_np)
