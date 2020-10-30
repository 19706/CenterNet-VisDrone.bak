import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm

def main():
    dirs = r'../../data/VisDrone2020-DET/VisDrone2019-DET-train/images'  # 修改你自己的图片路径
    img_file_names = os.listdir(dirs)
    m_list, s_list = [], []
    for img_filename in tqdm(img_file_names):
        img = cv2.imread(dirs + '/' + img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print("mean = ", m[0][::-1])
    print("std = ", s[0][::-1])


if __name__ == '__main__':
    main()

# the mean and std of VisDrone2019 is
# mean =  [0.37294899 0.37837514 0.36463863]
# std =  [0.19171683 0.18299586 0.19437608]
