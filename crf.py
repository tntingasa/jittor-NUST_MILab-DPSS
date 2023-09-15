import numpy as np
import os
import pydensecrf.densecrf as dcrf
try:
    from cv2 import imread, imwrite
except ImportError:
    from skimage.io import imread, imsave
    imwrite = imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
# from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from multiprocessing import Pool

def CRFs(original_image_path,predicted_image_path,CRF_image_path):

    img = imread(original_image_path)
    mask=imread(predicted_image_path)

    if len(img.shape)==2:
        img = np.stack((img,) * 3, axis=-1)

    anno_rgb = imread(predicted_image_path).astype(np.uint32)
    if img.shape != anno_rgb.shape:
        print("Inconsistent image and label sizes!")
        print(predicted_image_path)
        anno_rgb = anno_rgb.transpose((1, 0, 2))
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)

    colors, labels = np.unique(anno_lbl, return_inverse=True)
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    n_labels = len(set(labels.flat))

    use_2d = False
    if use_2d:
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=None)
        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)


        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        if n_labels-1==0:
            imwrite(CRF_image_path, mask)
            return
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)
        U = unary_from_labels(labels, n_labels, gt_prob=0.9, zero_unsure=None)
        d.setUnaryEnergy(U)

        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=2,kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)


        feats = create_pairwise_bilateral(sdims=(25, 25), schan=(7, 7, 7),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(15)

    MAP = np.argmax(Q, axis=0)

    MAP = colorize[MAP,:]
    imwrite(CRF_image_path, MAP.reshape(img.shape))

if __name__ == "__main__":

    original_image_dir = '/farm/litingting/jittor/ImageNetS50/test'
    predicted_image_dir = './result'
    CRF_image_dir = './result_crf'
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
                pool.apply_async(CRFs, (original_image, predicted_image, CRF_image))
                # CRFs(original_image, predicted_image, CRF_image)
            # print(original_image)
            # print(predicted_image)
            # print(CRF_image)
            pool.close()
            pool.join()

    print(count)


    # CRFs(original_image_path,predicted_image_path,CRF_image_path)
