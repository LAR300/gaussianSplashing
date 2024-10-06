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

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from utils.image_utils import erode, psnr
from uw_formation.backscatter import preform_full_bs_removal

def loss_func(iteration, gt_image, opt, gaussians, uw_flag=None, render_pkg=None, bs_dict=None, bs_image=None, uw_additions=None, uw=None):
    # Loss
    Ll1 = l1_loss( render_pkg["render"], gt_image)
    Ll2 = l2_loss( render_pkg["render"], gt_image).detach()
    Psnr = psnr( render_pkg["render"], gt_image).mean().double().detach()
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim( render_pkg["render"], gt_image))
    #loss = 0.6 * Ll1 + 0.4 * (1.0 - ssim(image, gt_image))
    if uw_flag.startswith("HYB"): 
        depth_map = render_pkg["rendered_depth"]
        if opt.lambda_bsest > 0:
            # binf = render_pkg["binf"]
            # bs = render_pkg["bs"]
            loss += opt.lambda_bsest * bs_loss(iteration=iteration, gaussians=gaussians, gt=gt_image, image_depth=depth_map, binf=uw_additions["binf"], bs=uw_additions["bs"], bs_dict=bs_dict, bs_image=bs_image, uw=uw, render_pkg=render_pkg)
            loss = loss + opt.opacity_reg * torch.abs(gaussians.get_opacity).mean()
            loss = loss + opt.scale_reg * torch.abs(gaussians.get_scaling).mean()
            #loss = loss + opt.lambda_bsest *  total_variation_loss(image=render_pkg["render"])
            # direct color loss
            loss = loss + opt.lambda_bsest *  (F.relu(-1*(uw_additions["binf"][0][1]-uw_additions["binf"][0][0])) + F.relu(-1*(uw_additions["binf"][0][2]-uw_additions["binf"][0][0])))
            loss = loss + opt.lambda_bsest *  (F.relu(+1*(uw_additions["direct"][0][1]-uw_additions["direct"][0][0])) + F.relu(+1*(uw_additions["direct"][0][2]-uw_additions["direct"][0][0])))
        
    return loss, Ll1, Ll2, Psnr

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def total_variation_loss(image):
    """
    Computes the Total Variation Loss for an input image.
    
    Args:
    - image (torch.Tensor): The input image tensor of shape (C, H, W),
                            where B is batch size, C is number of channels,
                            H is height, and W is width.
                            
    Returns:
    - torch.Tensor: The total variation loss for the image.
    """
    
    # Calculate the difference between neighboring pixels (horizontal and vertical)
    # Shifted images to compute differences between adjacent pixels
    diff_h = torch.abs(image[:, 1:, :] - image[:, :-1, :])  # Vertical differences
    diff_w = torch.abs(image[:, :, 1:] - image[:, :, :-1])  # Horizontal differences
    
    # Sum the differences
    loss = torch.sum(diff_h) + torch.sum(diff_w)
    
    return loss

def bs_loss(iteration, gaussians, gt, image_depth, binf, bs, bs_dict=None, bs_image=None, uw=None, render_pkg=None):
    curr_bs_dict = bs_dict
    if iteration > uw.remove_bs_every*2  and iteration % uw.remove_bs_every == 0:
        try:
            arr = image_depth.detach().cpu().numpy().squeeze(axis=0)
            arr[arr == 1] = 1.3
            clean_from_bs, bs_image, curr_bs_dict = preform_full_bs_removal(gt.cpu().numpy(), bs_dict=None, image_depth=arr, return_depth=False)
            
        except:
            print("Error in BS estimation")
    if curr_bs_dict is None or not bool(curr_bs_dict)\
    or np.isnan(curr_bs_dict['Bc_inf']).any() or  np.isnan(curr_bs_dict['betac_b']).any(): 
        return 0.0
    else:
        bs_dict = curr_bs_dict
    b_inf_gt = torch.Tensor(bs_dict['Bc_inf']).to(binf.device)
    bs_gt = torch.Tensor(bs_dict['betac_b']).to(bs.device)  
    # gaussians._binf = b_inf_gt
    # gaussians._bs = bs_gt
    # return 0.0
    return l2_loss(render_pkg["rendered_backscattering"], torch.Tensor(bs_image).to(render_pkg["rendered_backscattering"].device))
    return 0.5 * torch.mean((gaussians._bs - bs_gt) ** 2) + 0.5 * torch.mean((gaussians._binf - b_inf_gt) ** 2) 


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def zero_one_loss(img):
    zero_epsilon = 1e-3
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss

def predicted_normal_loss(normal, normal_ref, alpha=None):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (3, H, W), normal_ref: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device) 
    else:
        weight = torch.ones_like(normal_ref)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    n = normal_ref.permute(1,2,0).reshape(-1,3).detach()
    n_pred = normal.permute(1,2,0).reshape(-1,3)
    loss = (w * (1.0 - torch.sum(n * n_pred, axis=-1))).mean()

    return loss

def delta_normal_loss(delta_normal_norm, alpha=None):
    # delta_normal_norm: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device) 
    else:
        weight = torch.ones_like(delta_normal_norm)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    l = delta_normal_norm.permute(1,2,0).reshape(-1,3)[...,0]
    loss = (w * l).mean()

    return loss

def cam_depth2world_point(cam_z, pixel_idx, intrinsic, extrinsic):
    '''
    cam_z: (1, N)
    pixel_idx: (1, N, 2)
    intrinsic: (3, 3)
    extrinsic: (4, 4)
    world_xyz: (1, N, 3)
    '''
    valid_x = (pixel_idx[..., 0] + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
    valid_y = (pixel_idx[..., 1] + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
    ndc_xy = torch.stack([valid_x, valid_y], dim=-1)
    # inv_scale = torch.tensor([[W - 1, H - 1]], device=cam_z.device)
    # cam_xy = ndc_xy * inv_scale * cam_z[...,None]
    cam_xy = ndc_xy * cam_z[...,None]
    cam_xyz = torch.cat([cam_xy, cam_z[...,None]], dim=-1)
    world_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,0:1])], axis=-1) @ torch.inverse(extrinsic).transpose(0,1)
    world_xyz = world_xyz[...,:3]
    return world_xyz, cam_xyz
