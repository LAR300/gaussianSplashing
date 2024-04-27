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

import torch
import sys
from datetime import datetime
import numpy as np
import random
import cv2
from PIL import Image


def load_image(filename, code=cv2.IMREAD_COLOR, depth=False):
    if depth:
        return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(filename, code)
        if code == cv2.IMREAD_COLOR or code==cv2.IMREAD_UNCHANGED:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR->RGB
        if np.max(image) > 1:
            image = image / 255.0
        return image
    
       
def save_image(filename, image_orig, depth=False):
    if type(image_orig) == torch.Tensor:
        image = image_orig.detach().cpu().numpy()
        try:
            image = image.squeeze(0)
        except:
            pass
    if depth:
        cv2.imwrite(filename, image)
        # image = (image - image.min())/(image.max() - image.min())
        # image = (image * 255).astype(np.uint8) 
        # cv2.imwrite(filename, image)
        return
    
    if np.max(image) <= 1:
        image = (image * 255).astype(np.uint8) 
    if image.shape[0] in [1,2,3] and image.ndim >2:
        image = np.transpose(image, (1, 2, 0)) # transpose CHW -> HWC
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGB->BGR
    cv2.imwrite(filename, image)

def save_imagePIL(img_path, image_orig):
    if type(image_orig) == torch.Tensor:
        image = image_orig.detach().cpu().numpy()
        try:
            image = image.squeeze(0)
        except:
            pass
    if image.shape[0] in [1,2,3] and image.ndim >2:
        image = np.transpose(image, (1, 2, 0)) # transpose CHW -> HWC
    im = Image.fromarray(image)
    im.save(img_path)

def inverse_sigmoid(x):
    return torch.log(x/(1-x)) # sigmoid(x) = 1/(1+exp(-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    
def CVtoTorch(cv_image, resolution):
    # images suppose to be in RGB format, as in the case of PIL images (load_image function takes care for cv2 bgr format)
    # pil_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    resized_image_cv2 = cv2.resize(cv_image, resolution)
    if np.max(resized_image_cv2) > 1:
        resized_image = torch.from_numpy(resized_image_cv2.astype(np.float32) / 255.0)
    else:
        resized_image = torch.from_numpy(resized_image_cv2.astype(np.float32))
    if len(resized_image.shape) == 2: # grayscale meaning 1 channel
        resized_image = resized_image.unsqueeze(2) # add channel dimension
    if resized_image.shape[2] in [1,2,3]:
        # Permute dimensions to match the format expected by PyTorch (C, H, W)
        return resized_image.permute(2, 0, 1) # HWC -> CHW
    return resized_image
    
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
def get_minimum_axis(scales, rotations):
    sorted_idx = torch.argsort(scales, descending=False, dim=-1)
    R = build_rotation(rotations)
    R_sorted = torch.gather(R, dim=2, index=sorted_idx[:,None,:].repeat(1, 3, 1)).squeeze()
    x_axis = R_sorted[:,0,:] # normalized by defaut

    return x_axis

def flip_align_view(normal, viewdir):
    # normal: (N, 3), viewdir: (N, 3)
    dotprod = torch.sum(normal * -viewdir, dim=-1, keepdims=True) # (N, 1)
    non_flip = dotprod>=0 # (N, 1)
    normal_flipped = normal*torch.where(non_flip, 1, -1) # (N, 3)
    return normal_flipped, non_flip