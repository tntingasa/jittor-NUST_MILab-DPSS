import numpy as np
import cv2
import os, sys
import matplotlib.pyplot as plt
def load_annotations(ann_file):
    data_infos = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        for i,j in samples:
            data_infos[i] = np.array((j), dtype=np.int64)
    return data_infos


# #第3步   按分类去噪后的标签更改像素值t
img_label = load_annotations('./weights/pass50/pixel_finetuning_1/train_label_e46.txt')
root_t = './weights/pass50/pixel_attention_1/sam-train'
if not os.path.exists(root_t):
    os.makedirs(root_t)
image_name = list(img_label.keys())
label = list(img_label.values())
for i in range (len(image_name)):
        im=image_name[i]
        im=im.replace("JPEG", "png")
        target_img = os.path.join(root_t, im)
        if not os.path.exists(target_img) or label[i]==-2:
            mask_path = './weights/pass50/pixel_attention_1/train'
            mask_path2=os.path.join(mask_path, im)
            image_mask = cv2.imread(mask_path2)
            cv2.imwrite(target_img, image_mask)
        else:
            la=label[i]
            image = cv2.imread(target_img)
            # print(image.shape)
            image[:, :] = np.where(image[:, :]!=0, la+1, image[:, :])
            # 红
            b, g, r = cv2.split(image)
            # print(r.shape)
            zeros = np.zeros(image.shape[:2], dtype="uint8")  # 512x512
            image = cv2.merge([zeros, zeros, r])
            # print(image[250, 200][2])
            # print(image.shape)
            cv2.imwrite(target_img, image)

