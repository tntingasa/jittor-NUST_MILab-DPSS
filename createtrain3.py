import numpy as np
import cv2
import os, sys
import matplotlib.pyplot as plt

def load_annotations_1(ann_file):
    data_infos_d1 = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        for i in range (len(samples)):
            j1=samples[i][0]
            data_infos_d1[i] = j1
    return data_infos_d1
#第三步 建train
source_path = os.path.abspath('./sam/test')
target_path = os.path.abspath('./weights/pass50/pixel_classification/ccy_newclu_qzclu_43636_0.8867/train')
for f in os.listdir(source_path):
    target_dir = os.path.join(target_path, f)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
img_label1= load_annotations_1('./weights/pass50/pixel_classification/ccy_newclu_qzclu_43636_0.8867/train.txt')
D_1 = list(img_label1.values())

root='/home/litingting/jittor/ImageNetS50/train'
root2='./weights/pass50/pixel_classification/ccy_newclu_qzclu_43636_0.8867/train'
count=0
for i in range (len(img_label1)):
    path_0=D_1[i]

    path_sou=path_0
    path_0 = path_0.replace('_1.JPEG', '.JPEG')
    image_path_0 = os.path.join(root, path_0)


    image_path_sou = os.path.join(root2, path_sou)
    if os.path.exists(image_path_0):
        picture = cv2.imread(image_path_0)  # 读取数据
    else:
        image_path_0= image_path_0.replace('.JPEG', '_1.JPEG')
        picture = cv2.imread(image_path_0)  # 读取数据
    print(image_path_0)
    cv2.imwrite(image_path_sou, picture)
    count=count+1
print(count)
