import cv2
import numpy as np
import os


def get_path(images_path, masks_path, visualized_path):
    for filename in os.listdir(images_path):
        img_path = os.path.join(images_path, filename)
        print(img_path)
        mask_path = os.path.join(masks_path, filename[:-4] + 'png')
        print(mask_path)
        img = cv2.imread(img_path, 1)
        mask = cv2.imread(mask_path, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
        img = img[:, :, ::-1]
        img[..., 2] = np.where(mask == 1, 255, img[..., 2])
        cv2.imwrite(os.path.join(visualized_path, filename[:-4] + 'png'), img)
        print("{} saved".format(filename))
    print("finish")


images_path = '/farm/litingting/SAM_WSSS/data/train/n01443537/'
# masks_path = '/farm/litingting/SAM_WSSS/pseudo_labels/n01443537'
masks_path = '/farm/litingting/SAM_WSSS/processed_masks/n01443537/n01443537_n01443537_max_iou_imp2'
visualized_path = '/farm/litingting/SAM_WSSS/data/improve_mask_2/'
get_path(images_path, masks_path, visualized_path)

