from merge.merge_base import Merger
import os
from PIL import Image
import numpy as np
from pathlib import Path
import cv2 as cv


class MaxIoU_IMP2(Merger):
    def __init__(self, params, num_cls, threshold=0.2):
        super(MaxIoU_IMP2, self).__init__(params, num_cls)
        self.threshold = threshold

    def merge(self, predict, name, sam_folder, save_path):
        seen = []
        processed_mask = np.zeros_like(predict)

        for i in range(1, self.num_cls):
            pre_cls = predict == i
            if np.sum(pre_cls) == 0:
                continue
            iou = 0
            candidates = []
            sam_mask = np.zeros_like(pre_cls)
            # print(sam_folder.path)
            pre_bg = predict == 0
            for filename in os.scandir(sam_folder):
                # print('sam_filename:',filename.path)
                if filename.is_file() and filename.path.endswith('png') and filename.path not in seen:
                    cur_sam = np.array(Image.open(filename.path)) == 255
                    
                    if cur_sam.shape != sam_mask.shape:
#                           print("Inconsistent image and label sizes!")
#                           print(cur_sam.shape)
#                           print(sam_mask.shape)
                           cur_sam = cur_sam.transpose((1, 0))
#                           sam_mask.resize(cur_sam.shape)

#                     x1 = np.size(cur_sam, 0)
#                     y1 = np.size(cur_sam, 1)
#                     x2 = np.size(sam_mask, 0)
#                     y2 = np.size(sam_mask, 1)
#                     if x1 != x2 | y1 != y2:
#                         print("Inconsistent image and label sizes!")
#                         cur_sam = cur_sam.transpose((1, 0))
                    #print(filename)
                    sam_mask = np.logical_or(sam_mask, cur_sam)
                    improve_background_thresh = np.sum((pre_bg == cur_sam) * pre_bg) / np.sum(pre_bg)
                    improve_background_thresh_sam = np.sum((pre_bg == cur_sam) * pre_bg) / np.sum(cur_sam)
                    # improve_thresh = 2 * np.sum((pre_cls == cur_sam) * pre_cls) - np.sum(cur_sam)
                    improve_thresh = np.sum((pre_cls == cur_sam) * pre_cls) / np.sum(cur_sam)
                    # Note that the two way calculating (improve) are equivalent
                    improve_pred_thresh = np.sum((pre_cls == cur_sam) * pre_cls) / np.sum(pre_cls)

                    #if improve_background_thresh < 0.5 and improve_background_thresh_sam < 0.5:
                    if improve_thresh > 0.5 or improve_pred_thresh >= 0.85:
                                candidates.append(cur_sam)
                                seen.append(filename.path)
                                iou += np.sum(pre_cls == cur_sam)

            cam_mask = np.logical_and(sam_mask == 0, pre_cls == 1)
            # Trust CAM if SAM has no prediction on that pixel
            candidates.append(cam_mask)
            processed_mask[np.sum(candidates, axis=0) > 0] = i
        # print('#################################################')
        # print(name)
        r = processed_mask
        g = np.zeros(processed_mask.shape[:2], dtype=processed_mask.dtype)
        b = np.zeros(processed_mask.shape[:2], dtype=processed_mask.dtype)
        processed_mask = cv.merge([r, g, b])
        im = Image.fromarray(processed_mask)
        im.save(f'{save_path}/{name}.png')