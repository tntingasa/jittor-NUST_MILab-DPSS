import numpy as np
import os
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
try:
    from cv2 import imread, imwrite
except ImportError:
    # 如果没有安装OpenCV，就是用skimage
    from skimage.io import imread, imsave
    imwrite = imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from multiprocessing import Pool
import argparse

"""
original_image_path  原始图像路径
predicted_image_path  之前用自己的模型预测的图像路径
CRF_image_path  即将进行CRF后处理得到的结果图像保存路径
"""
def CRFs(original_image_path,predicted_image_path,CRF_image_path):
    

    img = imread(original_image_path)
    mask=imread(predicted_image_path)
    anno_rgb = imread(predicted_image_path).astype(np.uint32)
    
    if img.shape != anno_rgb.shape:
        print("Inconsistent image and label sizes!")
        print(predicted_image_path)
        anno_rgb = anno_rgb.transpose((1, 0, 2))
    #print(mask.shape)
    # w, h = img.shape[0:2]
    # w1,h1=mask.shape[0:2]
    # if w>900 and h>900 and w1==512 and h1==512:
    #     img = cv2.resize(img, (512, 512))
    # 将predicted_image的RGB颜色转换为uint32颜色 0xbbggrr
    #anno_rgb = imread(predicted_image_path).astype(np.uint32)
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)

    # 将uint32颜色转换为1,2,...
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # 如果你的predicted_image里的黑色（0值）不是待分类类别，表示不确定区域，即将分为其他类别
    # 那么就取消注释以下代码
    # HAS_UNK = 0 in colors
    # if HAS_UNK:
    #     colors = colors[1:]

    # 创建从predicted_image到32位整数颜色的映射。
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    # 计算predicted_image中的类数。
    n_labels = len(set(labels.flat))
    # n_labels = len(set(labels.flat)) - int(HAS_UNK) ##如果有不确定区域，用这一行代码替换上一行

    ###########################
    ###     设置CRF模型     ###
    ###########################
    use_2d = False               
    #use_2d = True   
    ###########################################################   
    ##不是很清楚什么情况用2D        
    ##作者说“对于图像，使用此库的最简单方法是使用DenseCRF2D类”
    ##作者还说“DenseCRF类可用于通用（非二维）密集CRF”
    ##但是根据我的测试结果一般情况用DenseCRF比较对
    #########################################################33
    if use_2d:                   
        # 使用densecrf2d类
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 增加了与颜色无关的术语，功能只是位置而已
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 增加了颜色相关术语，即特征是(x,y,r,g,b)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        if n_labels-1==0:
            imwrite(CRF_image_path, mask)
            return 
        # 使用densecrf类
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 这将创建与颜色无关的功能，然后将它们添加到CRF中
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 这将创建与颜色相关的功能，然后将它们添加到CRF中
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ###         做推理和计算         ###
    ####################################

    # 进行5次推理
    Q = d.inference(5)

    # 找出每个像素最可能的类
    MAP = np.argmax(Q, axis=0)

    # 将predicted_image转换回相应的颜色并保存图像
    MAP = colorize[MAP,:]
    imwrite(CRF_image_path, MAP.reshape(img.shape))
    #print("CRF图像保存在",CRF_image_path,"!")

    # CRFs("original.png","predict.png","predict_CRFs.png")
def main(args):
    # args = parser.parse_args()
    ##数据集地址##
    original_image_dir = os.path.join(args.data_path, args.mode)
    ##伪标签地址##
    predicted_image_dir = os.path.join(args.predict_path, args.mode)
    ##输出地址##
    CRF_image_dir = os.path.join(args.dump_path, args.mode+'_crf')
    count = 0
    list1 = os.listdir(original_image_dir)
    list1.sort(key=lambda x: int(x[1:]))
    for f in list1:
        print(f)
        count += 1
        original_image_dir2 = os.path.join(original_image_dir, f)
        predicted_image_dir2 = os.path.join(predicted_image_dir, f)
        CRF_image_dir2 = os.path.join(CRF_image_dir, f)
        if not os.path.exists(CRF_image_dir2):
            os.makedirs(CRF_image_dir2)
        list2 = os.listdir(original_image_dir2)
        list2.sort(key=lambda x: int(x[15:-5]))
        with Pool(processes=8) as pool:
            for f2 in list2:
                f3 = f2.replace('JPEG', 'png')
                original_image = os.path.join(original_image_dir2, f2)
                predicted_image = os.path.join(predicted_image_dir2, f3)
                CRF_image = os.path.join(CRF_image_dir2, f3)
                # if os.path.isfile(CRF_image):
                #     continue
                pool.apply_async(CRFs, (original_image, predicted_image, CRF_image))
                # CRFs(original_image, predicted_image, CRF_image)
            # print(original_image)
            # print(predicted_image)
            # print(CRF_image)
            pool.close()
            pool.join()

    print(count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path',
                        default=None,
                        type=str,
                        help='The path to the predictions.')
    parser.add_argument('--data_path',
                        default=None,
                        type=str,
                        help='The path to ImagenetS dataset')
    parser.add_argument('--dump_path',
                        default=None,
                        type=str,
                        help='The path to ImagenetS dataset')
    parser.add_argument('--mode',
                        type=str,
                        default='validation',
                        # choices=['validation', 'test'],
                        help='Evaluating on the validation or test set.')
    args = parser.parse_args()
    main(args)

