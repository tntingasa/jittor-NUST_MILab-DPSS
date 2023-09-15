from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms
import cv2 as cv


# img = Image.open(r"/farm/litingting/SAM_WSSS/pseudo_labels/labels/n01443537_2.png")
# image = np.array(img)
# input_transform = transforms.Compose([
#  transforms.ToTensor(),
# ])
# image = input_transform(image).unsqueeze(0)
# # 输出图片转为tensor后的格式
# print("image_tensor: ", image.shape)
# print(image.shape)
# img_1 = img.convert("L")
# img_1.save(os.path.join('/farm/litingting/SAM_WSSS/pseudo_labels/labels/','new.png'))
# img = cv.imread('/farm/litingting/SAM_WSSS/pseudo_labels/labels/n01443537_2.png')
# print('单通道merge：',img)

# img = cv.imread('/farm/litingting/SAM_WSSS/sam/n01443537/n01443537_2/1.png')
# img = Image.open('/farm/litingting/SAM_WSSS/sam/n01443537/n01443537_2/1.png')
# img = np.array(img)
# print('sam:',img)
# print('sam.shape:',img.shape)

img_1 = Image.open('/farm/litingting/SAM_WSSS/processed_masks/labels/labels_n01443537_max_iou_imp2/n01443537_2.png')
img_1 = np.array(img_1)
print('三通道Merge:',img_1)
print('三通道Merge。shape:',img_1.shape)
# print(img_1.type)

# img = Image.open('/farm/litingting/SAM_WSSS/processed_masks/labels/labels_n01443537_max_iou_imp2/单通道Merge.png')
# img = np.array(img)
# print('三通道merge：',img)

zeros = np.zeros((324,600,3),dtype='uint')
print('zeros.shape:',zeros.shape)

mse = np.mean( (img_1 - zeros) ** 2 )
print(mse)








