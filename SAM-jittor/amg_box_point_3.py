# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore
import numpy as np
import torch
import jittor as jt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from skimage import measure
import random
import argparse
import json
import os
from typing import Any, Dict, List

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--mask",
    type=str,
    help="Path to either a single mask image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)
parser.add_argument(
    "--txt1_path",
    type=str,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        # print(mask_data[bool("segmentation")])
        mask = mask_data
        filename = f"{i}.png"
        # base = os.path.basename(path)
        # filename = f"{base}.png"
        cv2.imwrite(os.path.join(path,filename), mask * 255)
        # cv2.imwrite(os.path.join(path +'.png'), mask * 255)
    #     mask_metadata = [
    #         str(i),
    #         str(mask_data["area"]),
    #         *[str(x) for x in mask_data["bbox"]],
    #         *[str(x) for x in mask_data["point_coords"][0]],
    #         str(mask_data["predicted_iou"]),
    #         str(mask_data["stability_score"]),
    #         *[str(x) for x in mask_data["crop_box"]],
    #     ]
    #     row = ",".join(mask_metadata)
    #     metadata.append(row)
    # metadata_path = os.path.join(path, "metadata.csv")
    # with open(metadata_path, "w") as f:
    #     f.write("\n".join(metadata))
    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

def largestConnectComponent(bw_img):
    labeled_img, num = measure.label(bw_img,connectivity=None, background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(0, num):
        if np.sum(labeled_img == 1) > max_num:
            max_num = np.sum(labeled_img == 1)
            max_label = i
    mcr = (labeled_img == max_label)
    return labeled_img,mcr

def load_annotations(ann_file):
    data_infos = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        for i,j in samples:
            data_infos[i] = np.array(j)
    return data_infos

def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model](checkpoint=args.checkpoint)
    # _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    predictor = SamPredictor(sam)

    ##获取含有path和点坐标的字典
    # txt1_path = '/farm/litingting/jittor/resnet_aspp_fcn/scripts/weights_epoch1000/pass50/pixel_attention_new2/valPoint.txt'
    img_point_1 = load_annotations(args.txt1_path)

    list1 = os.listdir(args.input)
    list1.sort(key=lambda x: int(x[1:]))
    for image_folder in list1:
        list2 = os.listdir(os.path.join(args.input, image_folder))
        list2.sort(key=lambda x: int(x[10:-5]))
        if not os.path.exists(os.path.join(args.output, image_folder)):
            os.makedirs(os.path.join(args.output, image_folder))

        for filename in list2:
            base = os.path.basename(filename)
            base = os.path.splitext(base)[0]
            save_base = os.path.join(args.output, image_folder, base)
            if os.path.exists(save_base):
                continue

            print(f"Processing '{filename}'...")
            img_path = os.path.join(args.input, image_folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not load '{filename}' as an image, skipping...")
                continue
            mask_path = os.path.join(args.mask, image_folder, filename[:-5] + '.png')
            mask = cv2.imread(mask_path)

            if image.shape != mask.shape:
                mask = mask.transpose((1, 0, 2))
            mask = mask[:, :, 2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if cv2.countNonZero(mask) == 0:
                continue
            # 输入prompts  box
            predictor.set_image(image)

            ###找框###
            ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
            ##寻找最大连通区域并框出###
            labeled_img, mcr = largestConnectComponent(mask)
            props = measure.regionprops(labeled_img)
            numPix = []
            for ia in range(len(props)):
                numPix += [props[ia].area]
            # 像素最多的连通区域及其指引
            maxnum = max(numPix)
            index = numPix.index(maxnum)
            minr, minc, maxr, maxc = props[index].bbox
            input_box = np.array([minc, minr, maxc, maxr])
            # cv2.rectangle(image, (minc, minr), (maxc, maxr), (0, 255, 255), 2)
            # cv2.imwrite(save_base2, image)
            path_1 = os.path.join(image_folder,filename)
            ###找点###
            zuobiao_1 = str(img_point_1[path_1])
            # print(zuobiao_1,zuobiao_2)
            zuobiao_1 = zuobiao_1.split(',')
            l_1 = int(zuobiao_1[0])
            h_1 = int(zuobiao_1[1])

            # print(l_1,l_2)

            if l_1 >= minc & l_1 <= maxc & h_1 >= minr & h_1 <= maxr :
                input_point = np.array([[l_1, h_1]])
                # input_point = np.expand_dims(input_point, axis=0)
                input_label = np.array([1])
                # masks, scores, logits = predictor.predict(
                #     point_coords=input_point,
                #     point_labels=input_label,
                #     box=input_box,
                #     multimask_output=True,
                # )
                # mask_input = logits[np.argmax(scores), :, :]
                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box,
                    # mask_input=mask_input[None, :, :],
                    # multimask_output=False,
                )
            else:
                # input_point = np.array([[l_1, h_1]])
                # input_point = np.expand_dims(input_point, axis=0)
                input_label = np.array([1])
                # masks, scores, logits = predictor.predict(
                #     point_coords=None,
                #     point_labels=input_label,
                #     box=input_box,
                #     multimask_output=True,
                # )
                # mask_input = logits[np.argmax(scores), :, :]
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=input_label,
                    box=input_box,
                    # mask_input=mask_input[None, :, :],
                    # multimask_output=False,
                )

            if output_mode == "binary_mask":
                if os.path.exists(save_base):
                    continue
                else:
                    os.makedirs(save_base, exist_ok=False)
                write_masks_to_folder(masks, save_base)
                print(save_base, ' finish!')
            else:
                save_file = save_base + ".json"
                with open(save_file, "w") as f:
                    json.dump(masks, f)
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    jt.flags.use_cuda = 1
    main(args)
