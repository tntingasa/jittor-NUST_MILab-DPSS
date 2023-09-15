import os
import jittor as jt
import jittor.nn as nn
import numpy as np
from PIL import Image
import jittor.dataset as datasets


def load_annotations(ann_file):
    data_infos = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        for i,j in samples:
            data_infos[i] = np.array((j), dtype=np.int64)
    return data_infos
class claDataset(datasets.ImageFolder):
    """Dataset for the finetuing stage."""
    def __init__(
        self,
        root,
        transform,
        pseudo_path
    ):
        super(claDataset, self).__init__(root, transform)
        self.pseudo_path = pseudo_path
        self.img_label = load_annotations('./weights/pass50/pixel_classification/train_clean_ccy_e36.txt')
        self.image_name = list(self.img_label.keys())
        self.label = list(self.img_label.values())
        self.data_dir = './weights/pass50/pixel_classification/train_clean_ccy_e36'
        self.image_path = [os.path.join(self.data_dir, img) for img in self.image_name]
        # print('123')

    def __getitem__(self, index):
        """
        Returns:
        img (Tensor): The loaded image. (3 x H x W)
        pseudo (str): The generated pseudo label. (H x W)
        """
        # path, _ = self.imgs[index]
        # print(index)
        # print('self.image_path:',len(self.image_path))
        path=self.image_path[index]
        img = Image.open(path).convert('RGB')
        pseudo_label=self.label[index]
        # pseudo = self.load_semantic(path)
        # pseudo = jt.array(np.array(pseudo)).permute(2, 0, 1).unsqueeze(0)
        # pseudo = nn.interpolate(pseudo.float(), (img.size[1], img.size[0]), mode="nearest").squeeze(0)
        # pseudo = Image.fromarray(pseudo.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        # img, pseudo = self.transform(img, pseudo_label)
        # print('path',path)
        # print('pseudo_label',pseudo_label)
        img = self.transform(img)
        # print('shape',img.shape)

        return img, pseudo_label


