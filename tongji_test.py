import os
import numpy as np
from PIL import Image
import cv2
import os, sys
import matplotlib.pyplot as plt
import os
def load_annotations(ann_file):
    data_infos = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        for i,j in samples:
            data_infos[i] = np.array((j), dtype=np.int64)
    return data_infos
def unique_values(source_img):
    concat_unique = np.empty(1)
    img = np.asarray(Image.open(source_img))
    unique = np.unique(img)
    concat_unique = np.concatenate([concat_unique, unique])
    return list(sorted(np.unique(concat_unique)))


if __name__ == '__main__':
    co=0
    source_path = os.path.abspath('./result2')
    for f in os.listdir(source_path):
        source_dir=os.path.join(source_path, f)
        for f2 in os.listdir(source_dir):
            co=co+1
            source_img = os.path.join(source_dir, f2)
            source = source_img.replace('./result2/','')
            # print(source_img)
            unique = unique_values(source_img)
            max=-1
            maxi=-1
            for i in range (len(unique)):
                if unique[i]>=1:
                    img = np.asarray(Image.open(source_img))
                    r = img[:,:,0] == unique[i]
                    if np.sum(r)>max:
                        max=np.sum(r)
                        maxi=unique[i]
            with open(os.path.join('./weights/pass50/pixel_finetuning_384', "test.txt"), "a") as f:
                f.write(source + ' ' + str(int(maxi) - 1))
                f.write("\n")
    # print('co:',co)



