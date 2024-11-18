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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import copy
from scipy.spatial.transform import Rotation as R
import numpy as np

from utils.graphics_utils import getWorld2View2
from utils.image_utils import apply_depth_colormap
from uw_formation.video_maker import create_video_from_images

def slerp(q0, q1, t):
    """Spherical linear interpolation."""
    dot = torch.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q0 + t * (q1 - q0)
        return result / torch.norm(result)
    theta_0 = torch.acos(dot)
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / torch.norm(q2)
    return q0 * torch.cos(theta) + q2 * torch.sin(theta)

def matrix_to_quaternion(matrix):
    """Convert a rotation matrix to a quaternion."""
    # r = R.from_matrix(matrix.cpu().numpy())
    # quaternion = r.as_quat() #  x, y, z, w
    # return torch.tensor([quaternion[3], quaternion[0], quaternion[1], quaternion[2]], device='cuda:0') # w, x, y, z
    ##############
    m = matrix
    qw = torch.sqrt(1 + m[0, 0] + m[1, 1] + m[2, 2]) / 2
    qx = (m[2, 1] - m[1, 2]) / (4 * qw)
    qy = (m[0, 2] - m[2, 0]) / (4 * qw)
    qz = (m[1, 0] - m[0, 1]) / (4 * qw)
    return torch.tensor([qw, qx, qy, qz], device='cuda:0')

def quaternion_to_matrix(quaternion):
    """Convert a quaternion to a rotation matrix."""
    # # w, x, y, z
    # quaternion = torch.tensor([quaternion[1], quaternion[2], quaternion[3], quaternion[0]], device='cuda:0')
    # r = R.from_quat(quaternion.cpu().numpy()) # x, y, z, w
    # rotation_matrix = r.as_matrix()
    # return torch.tensor(rotation_matrix, device='cuda:0')
    ##############
    qw, qx, qy, qz = quaternion
    qx2, qy2, qz2 = qx * qx, qy * qy, qz * qz
    qw2 = qw * qw
    m = torch.tensor([
        [1 - 2 * (qy2 + qz2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx2 + qz2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx2 + qy2)]
    ], device='cuda:0')
    return m

def interpolate_transform(initial_matrix, final_matrix, t):
    initial_translation, initial_rotation = extract_transform(initial_matrix)
    final_translation, final_rotation = extract_transform(final_matrix)
    # initial_rotation = initial_matrix[:3, :3]
    # final_rotation = final_matrix[:3, :3]
    # initial_translation = initial_matrix[:3, 3]
    # final_translation = final_matrix[:3, 3]
    

    # Convert rotation matrices to quaternions
    initial_quaternion = matrix_to_quaternion(initial_rotation)
    final_quaternion = matrix_to_quaternion(final_rotation)

    # Interpolate quaternions
    interpolated_quaternion = slerp(initial_quaternion, final_quaternion, t)

    # Convert interpolated quaternion back to rotation matrix
    interpolated_rotation = quaternion_to_matrix(interpolated_quaternion)

    # Interpolate translations
    interpolated_translation = initial_translation * (1 - t) + final_translation * t

    # Combine into a single transformation matrix
    
    interpolated_matrix = getWorld2View2(interpolated_rotation.cpu().numpy(), interpolated_translation.cpu().numpy())
    interpolated_matrix = interpolated_matrix.transpose()
    interpolated_matrix = torch.tensor(interpolated_matrix, device='cuda:0')
    return interpolated_matrix



def extract_transform(Rt):
    Rt = Rt.cpu().numpy()
    Rt = Rt.transpose(1, 0)
    extracted_t = Rt[:3, 3]
    extracted_R = Rt[:3, :3].transpose()
    return torch.tensor(extracted_t, device='cuda:0'), torch.tensor(extracted_R, device='cuda:0')
def interpolate_cameras(n_frames, initial_viewpoint_orig, final_viewpoint):
    
    interpolated_viewpoints = []
    initial_world_view_transform = initial_viewpoint_orig.world_view_transform
    final_world_view_transform = final_viewpoint.world_view_transform
    for i in range(n_frames):
        t = i / (n_frames - 1)
        if t == 1:
            break # Skip the last frame, which is the same as the final viewpoint
        intepolated_viewpoint = copy.deepcopy(initial_viewpoint_orig)
        interpolated_matrix = interpolate_transform(initial_world_view_transform, final_world_view_transform, t)
        intepolated_viewpoint.world_view_transform = interpolated_matrix
        intepolated_viewpoint.full_proj_transform = (intepolated_viewpoint.world_view_transform.unsqueeze(0).bmm(intepolated_viewpoint.projection_matrix.unsqueeze(0))).squeeze(0)
        intepolated_viewpoint.image_name = f"{intepolated_viewpoint.image_name}_" +  str(i).zfill(3)
        interpolated_viewpoints.append(intepolated_viewpoint)
    return interpolated_viewpoints

def create_video_set(scene, gaussians, pipe, bg, uw_flag, n_frames=10, save_str="all", flag_str="OFF"):
    if save_str:
        if save_str == "all":
            video_lst1 = ["video_rendered", "video_depth"]
            video_lst2 = ["video_rendered_J", "video_rendered_J_normalize","video_rendered_backscattering_normalize"]
            video_lst = video_lst1 + video_lst2
        else: 
            video_lst = [f"video_{save_str}"]
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_stack = sorted(viewpoint_stack, key=lambda x: x.image_name)
    print("video list: ", video_lst)
    print(f"Saving {viewpoint_stack[0].image_name} - {viewpoint_stack[-1].image_name}, frames between consecutive = {n_frames}...")
    for i in range(0, len(viewpoint_stack)-1):
        viewpoint_cam = viewpoint_stack[i]
        interpolated_viewpoints = interpolate_cameras(n_frames, viewpoint_stack[i], viewpoint_stack[i+1])
        if i == len(viewpoint_stack)-2:
            viewpoint_stack[i+1].image_name = f"{viewpoint_stack[i+1].image_name}_" +  str(0).zfill(3)
            interpolated_viewpoints.append(viewpoint_stack[i+1]) # Add the last viewpoint
        for viewpoint_cam in interpolated_viewpoints:
            rendering_pkg = render(viewpoint_cam, gaussians, pipe, bg, uw_flag=uw_flag)
            rendering = rendering_pkg["render"]
            
            rendering_depth_orig = rendering_pkg['rendered_depth'].detach().cpu().permute(1, 2, 0).numpy()
            rendering_depth = apply_depth_colormap(rendering_pkg['rendered_depth'].detach().cpu().permute(1, 2, 0)).numpy()
            
            scene.scene_save_image(rendering_depth, viewpoint_cam.image_name, os.path.join("train","video_depth"), depth=True)
            if viewpoint_cam.image_name.split("_")[-1] == "000":
                scene.scene_save_image(rendering_depth_orig, viewpoint_cam.image_name[:-4], os.path.join("train","video_depth_orig"), depth=True, bit16=True)
            scene.scene_save_image(rendering, viewpoint_cam.image_name, os.path.join("train","video_rendered"))
            if flag_str.startswith("HYB"):
                rendering_J = rendering_pkg["rendered_J"]
                scene.scene_save_image(rendering_J, viewpoint_cam.image_name, os.path.join("train","video_rendered_J"))
                rendering_J = rendering_pkg["rendered_J"]
                rendering_Bs = rendering_pkg["rendered_backscattering"]
                rendering_J_norm = scene.normalize_image(rendering_J)
                rendering_Bs_norm = scene.normalize_image(rendering_Bs)
                scene.scene_save_image(rendering_J_norm, viewpoint_cam.image_name, os.path.join("train","video_rendered_J_normalize"))
                scene.scene_save_image(rendering_Bs_norm, viewpoint_cam.image_name, os.path.join("train","video_rendered_backscattering_normalize"))
    if not flag_str.startswith("HYB"):
        for video_name in video_lst2:
            video_lst.remove(video_name) if video_name in video_lst else None
    for video_name in video_lst:
        image_directory = os.path.join(scene.model_path, "train", video_name)
        output_video_file =  os.path.join(scene.model_path, "train", video_name, f"{video_name}.mp4")
        create_video_from_images(image_directory, output_video_file, fps=10)
        

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)