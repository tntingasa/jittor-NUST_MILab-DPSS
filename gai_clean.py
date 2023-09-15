import numpy as np
import cv2
import os, sys
import matplotlib.pyplot as plt
def load_annotations_3(ann_file):
    data_infos_d1 = {}
    data_infos_d2 = {}
    data_infos_d3 = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        for i in range (len(samples)):
            j1=samples[i][0]
            j2 = samples[i][1]
            j3 = samples[i][2]
            data_infos_d1[i] = j1
            data_infos_d2[i] = np.array((j2), dtype=np.int8)
            data_infos_d3[i] = np.array((j3), dtype=np.float16)
    return data_infos_d1,data_infos_d2,data_infos_d3
def load_annotations_S(ann_file):
    data_infos_d1 = {}
    data_infos_d2 = {}
    data_infos_d3 = {}
    data_infos_d4 = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]

        for i in range (len(samples)):
            j1=samples[i][0]
            j2 = samples[i][1]
            j3 = samples[i][2]
            j4 = samples[i][3]

            data_infos_d1[i] = j1
            data_infos_d2[i] = np.array((j2), dtype=np.int8)
            data_infos_d3[i] = np.array((j3), dtype=np.float16)
            data_infos_d4[i] = np.array((j4), dtype=np.float16)


    return data_infos_d1,data_infos_d2,data_infos_d3,data_infos_d4
def load_annotations(ann_file):
    data_infos = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        for i,j in samples:
            data_infos[i] = np.array((j), dtype=np.int64)
    return data_infos
img_label1,img_label2,img_label3 = load_annotations_3('./label_fen.txt')
D_1 = list(img_label1.values())
D_2 = list(img_label2.values())
D_3 = list(img_label3.values())
img_label12,img_label22,img_label32 = load_annotations_3('./label2_fen.txt')
D_12 = list(img_label12.values())
D_22 = list(img_label22.values())
D_32 = list(img_label32.values())

root_t = './result'
root_2='./result2'
for i in range (len(img_label1)):
    im1 = D_1[i]
    la1=D_2[i]
    s1=D_3[i]
    im1 = im1.replace("JPEG", "png")
    target_img1 = os.path.join(root_t, im1)
    image1 = cv2.imread(target_img1)

    im2 = D_12[i]
    la2 = D_22[i]
    s2 = D_32[i]
    target_img2 = os.path.join(root_2, im1)
    image2 = cv2.imread(target_img2)

    if la1 in [10,19,49,32,34] and s1<12 :
        if la1==la2 :
            # print('-2:',im1,la1)
            image1[:, :] = np.where(image1[:, :] != 0, 0, image1[:, :])
            cv2.imwrite(target_img1, image1)
        if la1!=la2 and s2-s1>1.2:
            # print('huan:',im1,la1)
            cv2.imwrite(target_img1, image2)
    elif la1==-2 and la2!=-2 and s2>13 :
        # print('-2huan:',im1)
        cv2.imwrite(target_img1, image2)


