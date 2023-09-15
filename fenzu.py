#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 将一个文件夹下图片按比例分在两个文件夹下，比例改0.7这个值即可
import os
import random
import shutil
from shutil import copy2

input = './weights/pass50/pixel_finetuning_1/sam-train'
lj1 = './weights/pass50/pixel_finetuning_1/sam-train-3/train_1'
lj2 = './weights/pass50/pixel_finetuning_1/sam-train-3/train_2'
lj3 = './weights/pass50/pixel_finetuning_1/sam-train-3/train_3'

list1 = os.listdir(input)
list1.sort(key=lambda x: int(x[1:]))
for image_folder in list1:
    list2 = os.listdir(os.path.join(input, image_folder))
    list2.sort(key=lambda x: int(x[10:]))
    for filename in list2:
        list3 = os.listdir(os.path.join(input, image_folder,filename))
        list3.sort(key=lambda x: int(x[:-4]))
        for i in list3:
            img = os.path.join(input, image_folder,filename,i)
            lj_1 = os.path.join(lj1,image_folder)
            lj_2 = os.path.join(lj2, image_folder)
            lj_3 = os.path.join(lj3, image_folder)
            if not os.path.exists(lj_1):
                os.makedirs(lj_1)
            if not os.path.exists(lj_2):
                os.makedirs(lj_2)
            if not os.path.exists(lj_3):
                os.makedirs(lj_3)
            base = os.path.splitext(i)[0]
            if int(base) == 0:
                copy2(img, lj_1)
                re_name_1 = os.path.join(lj_1,i)
                new_name_1 = os.path.join(lj_1,filename+'.png')
                os.rename(re_name_1, new_name_1)
            if int(base) == 1:
                copy2(img, lj_2)
                re_name_2 = os.path.join(lj_2, i)
                new_name_2 = os.path.join(lj_2, filename+'.png')
                os.rename(re_name_2, new_name_2)
            if int(base) == 2:
                copy2(img, lj_3)
                re_name_3 = os.path.join(lj_3, i)
                new_name_3 = os.path.join(lj_3, filename+'.png')
                os.rename(re_name_3, new_name_3)


