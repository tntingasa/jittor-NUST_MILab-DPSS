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

#根据inference的标签更改标签
#./label.txt
img_label = load_annotations('./weights/pass50/pixel_finetuning_384/test.txt')
root_t = './result2'
image_name = list(img_label.keys())
label = list(img_label.values())
for i in range (len(image_name)):
        im=image_name[i]
        im=im.replace("JPEG", "png")
        target_img = os.path.join(root_t, im)
        la=label[i]
        image = cv2.imread(target_img)
        # print(image.shape)
        if la>-1:
            image[:, :] = np.where(image[:, :]!=0, la + 1, image[:, :])
        cv2.imwrite(target_img, image)

