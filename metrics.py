#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from PIL import Image, ImageDraw
def saveImageSpecial(image_to_save, origin_image_path, str_addition="redsquare"):
    directory, filename = os.path.split(origin_image_path)
    filename_no_ext, ext = os.path.splitext(filename)
    # Output path for the modified image
    output_path = os.path.join(directory, filename_no_ext + "_" + str_addition + ext)
    # Save the modified image
    image_to_save.save(output_path)
    
    
def readImages(renders_dir, gt_dir, red_flag=False, top_left_x=850, top_left_y=15, width=300, height=150):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        if "redsquare" in fname: continue
        render_path = renders_dir / fname
        render = Image.open(render_path)
        gt_path = gt_dir / fname
        gt = Image.open(gt_path)
        if red_flag:
            render = render.crop((top_left_x, top_left_y, top_left_x + width, top_left_y + height))
            saveImageSpecial(image_to_save=render, origin_image_path=render_path, str_addition="redsquarecrop")
            gt = gt.crop((top_left_x, top_left_y, top_left_x + width, top_left_y + height))
            saveImageSpecial(image_to_save=gt, origin_image_path=gt_path, str_addition="redsquarecrop")
            
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        render.close()
        gt.close()
        image_names.append(fname)
    return renders, gts, image_names

def draw_box(images_dir, top_left_x, top_left_y, width, height):
    for fname in os.listdir(images_dir):
        if "redsquare" in fname: continue
        image_path = images_dir / fname
        image = Image.open(image_path)
        # Create Draw object
        draw = ImageDraw.Draw(image)
        # Define the coordinates of the box
        top_left = (top_left_x, top_left_y)
        bottom_right = (top_left_x + width, top_left_y + height)
        # Draw the box on the image
        draw.rectangle([top_left, bottom_right], outline="red", width=2)
        saveImageSpecial(image_to_save=image, origin_image_path=image_path, str_addition="redsquare")
        image.close()

def evaluate(scene_dir, red_flag=True, top_left_x=850, top_left_y=15, width=300, height=150):
    scene_dir = scene_dir[0]
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    try:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        
        #for set_name in ["train"]:
        for set_name in ["train", "eval"]:
            print("Set:", set_name)

            full_dict[scene_dir][set_name] = {}
            per_view_dict[scene_dir][set_name] = {}
            full_dict_polytopeonly[scene_dir][set_name] = {}
            per_view_dict_polytopeonly[scene_dir][set_name] = {}

            set_dir = Path(scene_dir) / set_name
            gt_dir = set_dir/ "gt"
            renders_dir = set_dir / "renders"
            red_flag_str = "_!redsquare!" if red_flag else ""
            print("  Red flag:", red_flag)
            if red_flag:
                draw_box(gt_dir, top_left_x, top_left_y, width, height)
                draw_box(renders_dir, top_left_x, top_left_y, width, height)
            renders, gts, image_names = readImages(renders_dir, gt_dir, red_flag, top_left_x, top_left_y, width, height)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc=f"Metric evaluation progress {red_flag_str}"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][set_name].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item()})
            per_view_dict[scene_dir][set_name].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

        with open(scene_dir + f"/results{red_flag_str}.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + f"/per_view{red_flag_str}.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)
    except:
        print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--scene_dir', '-s', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.scene_dir)
