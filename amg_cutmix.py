# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import json
import os
from typing import Any, Dict, List
import numpy as np
import random

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
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
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
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

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


def seg_cutmix(src_img, mask, dest_path, dset_cutmix_path):
    dest_image = cv2.imread(dest_path)
    dest_h = dest_image.shape[0]
    dest_w = dest_image.shape[1]
    
    mask = np.array(mask*255).astype('uint8')
    grayImage = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    grayImage = cv2.resize(grayImage, (dest_w,dest_h))
    
    resizeed_srcimg = cv2.resize(src_img, (dest_w, dest_h))
    
    dest_image[grayImage > 0.5] = resizeed_srcimg[grayImage > 0.5]
    cv2.imwrite(dset_cutmix_path, dest_image)

def xytranslate(image,x_translation, y_translation):
    # x_translation = 50  # Move 50 pixels to the right (X-axis)
    # y_translation = 30  # Move 30 pixels down (Y-axis)

    # Define the transformation matrix for translation
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])

    # Use cv2.warpAffine() to apply the translation
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    return translated_image

def roateimg(image, angle_degrees, scale):
    # Define the rotation angle (in degrees) and the center of rotation
    #angle_degrees = 45  # Rotate by 45 degrees
    center = (image.shape[1] // 2, image.shape[0] // 2)  # Center of rotation

    # Create a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale)

    # Use cv2.warpAffine() to apply the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image


def zoomout(image, zoom_factor):
    # print (zoom_factor)
    # Define the scaling factor for zooming out (e.g., 0.5 for 50% zoom-out)
    # zoom_factor = 0.9  # 50% zoom-out

    # Get the dimensions of the original image
    original_height, original_width = image.shape[0], image.shape[1]

    # Calculate the new dimensions after zooming out
    new_width = int(original_width * zoom_factor)
    new_height = int(original_height * zoom_factor)

    # Calculate the coordinates for cropping a portion from the center of the image
    left = (original_width - new_width) // 2
    top = (original_height - new_height) // 2
    right = (original_width + new_width) // 2
    bottom = (original_height + new_height) // 2

    # Crop the centered portion of the image
    cropped_image = image[top:bottom, left:right]

    # Use cv2.resize() to resize the cropped portion while keeping the original dimensions
    zoomed_out_image = cv2.resize(cropped_image, (original_width, original_height))
    return zoomed_out_image

def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)
    
    #for idx, dest_path in enumerate(targets[0:20]):
    for idx, dest_path in enumerate(targets):
        print(f"Processing '{dest_path}'...")
        dest_image = cv2.imread(dest_path)
        dest_h = dest_image.shape[0]
        dest_w = dest_image.shape[1]
        rlist = random.sample(range(len(targets)), random.randint(2,3))
        print (rlist)
        bnamewoext = os.path.splitext(os.path.basename(dest_path))[0]
        extonly = os.path.splitext(os.path.basename(dest_path))[1] 

        seg_poll = map(lambda i: targets[i], rlist)
        for seg_path in seg_poll:
            seg_image = cv2.imread(seg_path)
            seg_image = cv2.resize(seg_image, (dest_w, dest_h))
            seg_imagewrgb = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
            masks = generator.generate(seg_imagewrgb)
            
            segmask = masks[0]["segmentation"]
            # numpy to cv
            segmask = np.array(segmask*255).astype('uint8')
            segmask_img = cv2.cvtColor(segmask, cv2.COLOR_GRAY2BGR)
            
            angle_degrees = random.randint(0, 180)
            if random.randint(0,10) >5:
                angle_degrees *= -1
            scale = random.randint(30, 75)/100
            seg_image = roateimg(seg_image, angle_degrees, scale)
            segmask_img = roateimg(segmask_img, angle_degrees, scale)
            
            zh = random.randint(dest_h//16, dest_h//8)
            zw = random.randint(dest_w//16, dest_w//8)
            if random.randint(0,10) > 5:
              zh *=-1  
              zw *=-1  
            seg_image = xytranslate(seg_image, zw, zh)
            segmask = xytranslate(segmask_img, zw, zh)
            
            dest_image[segmask > 0.5] = seg_image[segmask > 0.5]

        dest_cutmix_path = f"/home/user/work_2023/segment-anything/yun/train/Images/{bnamewoext}_cutmix{extonly}"
        cv2.imwrite(dest_cutmix_path, dest_image)
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
