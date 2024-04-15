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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,  uw_flag="OFF", scaling_modifier = 1.0, override_color = None, override_color_cmed = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity # initial opacity is (inverse_sigmoid) 0.1 (for example, for the first iteration of the optimization process)
    

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python: #  usually False
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling # initialization is: the average distance of the points from its 3 nearest neighbors (ln(sqrt(dist))) 
        rotations = pc.get_rotation # initialization is: no rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if uw_flag=="ON":
        colors_precomp_cmed = None # addition uw
    if uw_flag=="HYB":
        direct = pc.get_direct
        binf = pc.get_binf
        bs = pc.get_bs
    if override_color is None: # usually None
        if pipe.convert_SHs_python: # usually False
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            if uw_flag=="ON":
                # start addition uw
                shs_view_cmed = pc.get_features_cmed.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp_cmed = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features_cmed.shape[0], 1))
                dir_pp_cmed_normalized = dir_pp_cmed/dir_pp_cmed.norm(dim=1, keepdim=True)
                sh2rgb_cmed = eval_sh(pc.active_sh_degree, shs_view_cmed, dir_pp_cmed_normalized)
                colors_precomp_cmed = torch.clamp_min(sh2rgb_cmed + 0.5, 0.0)
                # end addition uw
        else:
            shs = pc.get_features # colors are represented by SHs
            if uw_flag=="ON":
                # start addition uw
                shs_cmed = pc.get_features_cmed
    else:
        colors_precomp = override_color
        colors_precomp_cmed = override_color_cmed # addition uw

    if uw_flag=="ON":
        # start addition uw
        opm_shs_view = pc.get_opacity_med.transpose(1, 2).view(-1, 1, (pc.max_sh_degree+1)**2) # (P, 1, 16)
        opm_dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity_med.shape[0], 1))
        opm_dir_pp_normalized = opm_dir_pp/opm_dir_pp.norm(dim=1, keepdim=True)
        opm_sh2rgb = eval_sh(pc.active_sh_degree, opm_shs_view, opm_dir_pp_normalized)
        opacitymedium_precomp = torch.clamp_min(opm_sh2rgb + 0.5, 0.0) # (P, 1, 1)
        # end addition uw
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # fowrad() returns a dictionary with the rendered image, depth and alpha,
    if uw_flag=="ON":
        # start addition uw
        rendered_image, radii, rendered_depth, rendered_alpha  = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            # start addition uw
            shs_cmed = shs_cmed,
            colors_precomp_cmed = colors_precomp_cmed, 
            opacities_med = opacitymedium_precomp,
            # end addition uw
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        # end addition uw
    elif uw_flag=="HYB":
        # start addition uw
        rendered_image, radii, rendered_depth, rendered_J, rendered_backscattering = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            # start addition uw
            direct = direct,
            binf = binf, 
            bs = bs,
            # end addition uw
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        # end addition uw
    else:
        if pipe.use_depth_rasterizer:
            rendered_image, radii, rendered_depth, rendered_alpha  = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
        else:
            rendered_image, radii= rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if uw_flag=="ON" and pipe.use_depth_rasterizer:
        return {"render": rendered_image,
                "rendered_depth": rendered_depth,
                "rendered_alpha": rendered_alpha,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}
    elif uw_flag=="HYB":
        return {"render": rendered_image,
                "rendered_depth": rendered_depth,
                "rendered_J" : rendered_J,
                "rendered_backscattering" : rendered_backscattering,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "direct": direct,
                "binf": binf,
                "bs": bs}
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}