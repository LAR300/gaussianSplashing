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
import os
import cv2
import torch
import torchvision
import numpy as np
from random import randint
from metrics import evaluate
from render import create_video_set
from utils.loss_utils import l1_loss, l2_loss, loss_func
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, save_image
import uuid
import time
from tqdm import tqdm
from utils.image_utils import psnr, apply_depth_colormap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, WandbParams, UnderwaterParams, NestedDict, add_general_params, write_dictionary_to_file
import wandb
from utils.sh_utils import eval_sh
from uw_formation.depthmap import depth_estimation
from uw_formation.plotBs import save_backscatter_plot
from scene.gaussian_model import build_scaling_rotation




def train_sweep(namespace_dict=None, sweep_train_flag=True):
    if sweep_train_flag:
        run_sweep = wandb.init(
                project="gaussian_splattingUW",
                group="GroupSweepDebugUW",
                config=namespace_dict,
            )
        namespace_dict = run_sweep.config
        
    cfg_dot_dic = NestedDict(namespace_dict)
    if not sweep_train_flag:
        print("Optimizing " + cfg_dot_dic.LoadingParameters.model_path)
    else:
        print("Sweeping... ")

    # Initialize system state (RNG)
    safe_state(cfg_dot_dic.GeneralParameters.quiet)

    # Start GUI server, configure and run training
    if cfg_dot_dic.GeneralParameters.ip != "0" and cfg_dot_dic.GeneralParameters.port != -1:
        network_gui.init(cfg_dot_dic.GeneralParameters.ip, cfg_dot_dic.GeneralParameters.port)
    torch.autograd.set_detect_anomaly(cfg_dot_dic.GeneralParameters.detect_anomaly)
    checkpoint_path = None
    if type(cfg_dot_dic.GeneralParameters.start_checkpoint)==str and os.path.isfile(cfg_dot_dic.GeneralParameters.start_checkpoint):
        checkpoint_path = cfg_dot_dic.GeneralParameters.start_checkpoint
    training_miutes, training_seconds = training(cfg_dot_dic ,cfg_dot_dic.LoadingParameters, cfg_dot_dic.OptimizationParameters, cfg_dot_dic.PipelineParameters,  cfg_dot_dic.UnderwaterParameters,
                                                 checkpoint=checkpoint_path, sweep_train_flag=sweep_train_flag)
    # All done
    print("Training complete.")
    print(f"Training execution time: {int(training_miutes)} minutes {training_seconds:.2f} seconds")
    



def training(cfg_dot_dic, dataset, opt, pipe, uw, checkpoint=None, sweep_train_flag=False):
    
    start_training = time.time()
    first_iter = 0
    prepare_output(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.underwater_processing)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # start UW processing
    curr_bs_dict = {}
    curr_bs_image = None
    uw_additions = {}
    clone_split_range_iter = 0
    for iteration in range(first_iter, opt.iterations + 1):        
        if cfg_dot_dic.GeneralParameters.ip != "0" and cfg_dot_dic.GeneralParameters.port != -1:
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background,  uw_flag=dataset.underwater_processing, scaling_modifier=scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None
        # end UW processing
        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == cfg_dot_dic.GeneralParameters.debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background # [0,0,0] most of the times
        if dataset.underwater_processing.startswith("HYB"):
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, uw_flag=dataset.underwater_processing)
            image, depth_map, rendered_J, rendered_backscattering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["rendered_depth"], render_pkg["rendered_J"], render_pkg["rendered_backscattering"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, uw_flag=dataset.underwater_processing)
            image, rendered_depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["rendered_depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # if iteration in cfg_dot_dic.GeneralParameters.test_iterations and dataset.underwater_processing.startswith("HYB"):
        #     save_image(os.path.join("uw_formation", "depth_gs" ,f"{viewpoint_cam.image_name}_gs{iteration}_depth.png"), depth_map, depth=True)
        pipe.debug = False
        # Underwater backscatter removal
        gt_image = viewpoint_cam.original_image.cuda()
        # Loss
        if dataset.underwater_processing.startswith("HYB"):
            uw_additions["binf"] =  gaussians._binf #eval_sh(gaussians.active_sh_degree, gaussians.get_featuresbinf.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2), viewpoint_cam.camera_front_dir_in_world)
            uw_additions["bs"] = gaussians._bs  #eval_sh(gaussians.active_sh_degree, gaussians.get_featuresbs.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2), viewpoint_cam.camera_front_dir_in_world)
            uw_additions["direct"] =  gaussians._direct #eval_sh(gaussians.active_sh_degree, gaussians.get_featuresdirect.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2), viewpoint_cam.camera_front_dir_in_world)
        loss, Ll1, Ll2, Psnr = loss_func(iteration, gt_image, opt, gaussians, uw_flag=dataset.underwater_processing, render_pkg = render_pkg, bs_dict=curr_bs_dict, bs_image=curr_bs_image, uw_additions=uw_additions, uw=uw)
        
        # NaN detection and handling
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARNING] NaN/Inf loss detected at iteration {iteration}: {loss.item()}")
            if dataset.underwater_processing.startswith("HYB"):
                print(f"  - binf: {gaussians._binf}")
                print(f"  - bs: {gaussians._bs}")  
                print(f"  - direct: {gaussians._direct}")
                print(f"  - Rendered image stats: min={render_pkg['render'].min():.6f}, max={render_pkg['render'].max():.6f}, mean={render_pkg['render'].mean():.6f}")
                if 'rendered_J' in render_pkg:
                    print(f"  - Rendered J stats: min={render_pkg['rendered_J'].min():.6f}, max={render_pkg['rendered_J'].max():.6f}, mean={render_pkg['rendered_J'].mean():.6f}")
                if 'rendered_backscattering' in render_pkg:
                    print(f"  - Rendered backscattering stats: min={render_pkg['rendered_backscattering'].min():.6f}, max={render_pkg['rendered_backscattering'].max():.6f}, mean={render_pkg['rendered_backscattering'].mean():.6f}")
            # Skip this iteration if loss is NaN/Inf
            continue
            
        if dataset.underwater_processing.startswith("HYB"):
            gaussians.enforce_constraints_hyb(direct_range=(uw.exp_range_dl, uw.exp_range_dh), binf_range=(uw.binf_range_l, uw.binf_range_h), bs_range=(uw.exp_range_bl, uw.exp_range_bh),
                                                uw_additions=uw_additions)
            #uw_additions["direct"], uw_additions["binf"], uw_additions["bs"] = render_pkg["direct"], render_pkg["binf"], render_pkg["bs"]
            uw_additions["binf"] =  gaussians._binf #eval_sh(gaussians.active_sh_degree, gaussians.get_featuresbinf.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2), viewpoint_cam.camera_front_dir_in_world)
            uw_additions["bs"] = gaussians._bs  #eval_sh(gaussians.active_sh_degree, gaussians.get_featuresbs.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2), viewpoint_cam.camera_front_dir_in_world)
            uw_additions["direct"] =  gaussians._direct #eval_sh(gaussians.active_sh_degree, gaussians.get_featuresdirect.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2), viewpoint_cam.camera_front_dir_in_world)
        loss.backward()
        
        # Gradient clipping to prevent gradient explosion
        if dataset.underwater_processing.startswith("HYB"):
            # Clip gradients for underwater parameters to prevent numerical instability
            max_grad_norm = 5.0
            if gaussians._binf.grad is not None:
                torch.nn.utils.clip_grad_norm_([gaussians._binf], max_grad_norm)
            if gaussians._bs.grad is not None:
                torch.nn.utils.clip_grad_norm_([gaussians._bs], max_grad_norm)
            if gaussians._direct.grad is not None:
                torch.nn.utils.clip_grad_norm_([gaussians._direct], max_grad_norm)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_rep_uw =  dataset.underwater_processing
            def save_for_metrics(set_name='train', flag_str="OFF"):
                execution_time_lst = []
                viewpoint_stack = scene.getTrainCameras().copy() if set_name=="train" else scene.getTestCameras().copy()
                print(f"Saving train for metrics - {set_name} , {os.path.join(scene.model_path, set_name)} ...")
                for viewpoint_cam in viewpoint_stack:
                    print(f"processing image - {viewpoint_cam.image_name}")
                    start_time = time.time()
                    rendering_pkg = render(viewpoint_cam, gaussians, pipe, bg, uw_flag=dataset.underwater_processing)
                    end_time = time.time()
                    execution_time_lst.append(end_time - start_time)
                    rendering = rendering_pkg["render"]
                    rendering_depth = apply_depth_colormap(rendering_pkg['rendered_depth'].detach().cpu().permute(1, 2, 0)).numpy()
                    rendering_depth_orig = rendering_pkg['rendered_depth'].detach().cpu().numpy().squeeze(axis=0)
                    gt = viewpoint_cam.original_image[0:3, :, :]
                    scene.scene_save_image(rendering_depth_orig, viewpoint_cam.image_name, os.path.join(set_name,"rendered_depth_orig"), depth=True, bit16=True)
                    scene.scene_save_image(rendering_depth_orig, viewpoint_cam.image_name, os.path.join(set_name,"rendered_depth_orig255"), depth=True, bit16=False)
                    scene.scene_save_image(rendering_depth, viewpoint_cam.image_name, os.path.join(set_name,"rendered_depth"), depth=True)
                    scene.scene_save_image(gt, viewpoint_cam.image_name, os.path.join(set_name,"gt"))
                    scene.scene_save_image(rendering, viewpoint_cam.image_name, os.path.join(set_name,"renders"))
                    if flag_str.startswith("HYB"):
                        rendering_J = rendering_pkg["rendered_J"]
                        rendering_Bs = rendering_pkg["rendered_backscattering"]
                        rendering_J = scene.normalize_image(rendering_J)
                        rendering_Bs = scene.normalize_image(rendering_Bs)
                        scene.scene_save_image(rendering_J, viewpoint_cam.image_name, os.path.join(set_name,"rendered_J"))
                        scene.scene_save_image(rendering_Bs, viewpoint_cam.image_name, os.path.join(set_name,"rendered_backscattering"))
                    
                fps = sum(execution_time_lst) / len(execution_time_lst) 
                print(f"Execution (render {set_name}) time:", fps, "seconds")
                scene.scene_save_fps(f"Frame Per Second (render {set_name}): " + str(int(1/fps)) + " or " + str(fps) + " seconds per frame")
            if cfg_dot_dic.GeneralParameters.save_for_metrics and iteration == opt.iterations:
                #gaussians.prune_MCMC(percentile=67)
                save_for_metrics(set_name='train', flag_str=dataset.underwater_processing)
                save_for_metrics(set_name='eval', flag_str=dataset.underwater_processing)
                evaluate(cfg_dot_dic.LoadingParameters.model_path[2:])
                    
                


            
            training_report(iteration, Ll1, loss, l1_loss,
                            Ll2, l2_loss, Psnr,
                            iter_start.elapsed_time(iter_end), cfg_dot_dic.GeneralParameters.test_iterations, scene, render, [pipe, bg, training_rep_uw],
                            (dataset.underwater_processing, uw, uw_additions, dataset.removal_using_monocular_depth), curr_bs_dict)
            if (iteration in cfg_dot_dic.GeneralParameters.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                print("\n[ITER {}] Saving was denied by the user".format(iteration))
                #scene.save(iteration)
        
            # Densification
            if iteration % opt.opacity_reset_interval == 0:# or (dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()
            if dataset.densification == "OFF":
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    
                    # Prune near-camera large Gaussians BEFORE densification (after max_radii2D update)
                    if (iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and 
                        opt.prune_near_large_gaussians and iteration > opt.densify_from_iter + opt.densification_interval):
                        gaussians.prune_near_large_gaussians(
                            camera_center=viewpoint_cam.camera_center,
                            distance_threshold=opt.near_camera_distance_threshold,
                            screen_size_threshold=opt.large_gaussian_screen_size_threshold,
                            opacity_threshold=opt.near_large_gaussian_opacity_threshold,
                            verbose=(iteration % (5 * opt.densification_interval) == 0)  # Log every 5th densification
                        )
                    
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = opt.large_gaussian_densification_screen_size_threshold if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune_orig(opt.densify_grad_threshold, opt.min_opacity_prune, scene.cameras_extent, size_threshold, radii)
                        gaussians.prune_nan()
                        
            elif dataset.densification == "MCMC":
                if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # Prune near-camera large Gaussians BEFORE MCMC operations (after max_radii2D update)
                    if (opt.prune_near_large_gaussians and iteration > opt.densify_from_iter + opt.densification_interval):
                        gaussians.prune_near_large_gaussians(
                            camera_center=viewpoint_cam.camera_center,
                            distance_threshold=opt.near_camera_distance_threshold,
                            screen_size_threshold=opt.large_gaussian_densification_screen_size_threshold,
                            opacity_threshold=opt.near_large_gaussian_opacity_threshold,
                            verbose=(iteration % (5 * opt.densification_interval) == 0)  # Log every 5th densification
                        )
                    
                    # if iteration % (5*opt.densification_interval) == 0:
                    #     gaussians.prune_MCMC()
                    dead_mask = (gaussians.get_opacity <= opt.remove_min_opacity_threshold).squeeze(-1)
                    gaussians.relocate_gs(dead_mask=dead_mask)
                    gaussians.add_new_gs(cap_max=args.cap_max)
        
                        
                
            if opt.masking_interval != -1 and (iteration % opt.masking_interval) == 0:
                gaussians.prune_by_maskingBG(scene.getTrainCameras())
                
            
                
            # Optimizer step
            if iteration <= opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if dataset.densification == "MCMC":
                    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)

                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))
                    
                    noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)
                    
                

            if (iteration in cfg_dot_dic.GeneralParameters.checkpoint_iterations and not sweep_train_flag):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            
    
    end_training = time.time()
    elapsed_training_time = end_training - start_training
    minutes, seconds = divmod(elapsed_training_time, 60)
    
    prepare_output_end(cfg_dot_dic, minutes, seconds, sweep_train_flag)
    if wandb.run is not None:
        wandb.finish(exit_code=0)
    if cfg_dot_dic.GeneralParameters.save_video:
        print("Creating video set...")
        with torch.no_grad():
            create_video_set(scene, gaussians, pipe, background, dataset.underwater_processing, save_str="all", flag_str=dataset.underwater_processing)
    
    
    return minutes, seconds

def prepare_output(dataset):           
    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    
def prepare_output_end(cfg_dot_dic, minutes, seconds, sweep_train_flag=False):
    cfg_dot_dic['TrainingTime'] = {"minutes": minutes, "seconds": seconds}
    if not sweep_train_flag:
        write_dictionary_to_file(cfg_dot_dic, os.path.join(cfg_dot_dic.LoadingParameters.model_path,"cfg_args.json"))
    if wandb.run is not None:
        wandb.config.update({"TrainingTime": {"minutes": minutes, "seconds": seconds}})
    # evaluate(cfg_dot_dic.LoadingParameters.model_path[2:])

def training_report(iteration, Ll1, loss, l1_loss,
                    Ll2, l2_loss, Psnr,
                    elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,
                    uw_tuple=("OFF",None, None, False), curr_bs_dict=None):

    if wandb.run is not None:
        wandb.run.log(data={'train_loss_patches/l1_loss': Ll1.item(), 'train_loss_patches/total_loss': loss.item(),
                            'train_loss_patches/l1_loss': Ll2.item(), 'train_loss_patches/l1_loss': Psnr.item(),
                            'time/iter_time': elapsed}, step=iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(0, 30, 4)]})

        for config in validation_configs:
            
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                l2_test = 0.0
                psnr_test = 0.0
                
                im_test_flag = True if config['name'] == "test" else False # im_test_flag = True if config['name'] == "test" else False
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, im_test_flag=im_test_flag)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clip(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    depth_map = torch.clip((render_pkg["rendered_depth"] - render_pkg["rendered_depth"].min()) / (render_pkg["rendered_depth"].max() - render_pkg["rendered_depth"].min()), 0.0, 1.0)
                    try:
                        depth_map_unnormalize = render_pkg["rendered_depth"].detach().cpu().numpy().squeeze(axis=0)
                    except:
                        depth_map_unnormalize = render_pkg["rendered_depth"].detach().cpu().numpy()
                    
                    if uw_tuple[-1]: #removal_using_monocular_depth
                            if viewpoint.image_depth_monocular is None:
                                depth_map_unnormalize = depth_estimation(viewpoint.original_image.cpu().numpy())
                                viewpoint.image_depth_monocular = depth_map_unnormalize
                            else:
                                depth_map_unnormalize = viewpoint.image_depth_monocular
                            depth_map_normalize = torch.clip((torch.Tensor(depth_map_unnormalize) - torch.Tensor(depth_map_unnormalize).min()) / (torch.Tensor(depth_map_unnormalize).max() - torch.Tensor(depth_map_unnormalize).min()), 0.0, 1.0)
                    if uw_tuple[0].startswith("HYB"):
                        rendered_backscattering = scene.normalize_image(render_pkg["rendered_backscattering"])
                        rendered_J = scene.normalize_image(render_pkg["rendered_J"])
                        # rendered_backscattering = torch.clamp(render_pkg["rendered_backscattering"], 0.0, 1.0)
                        # rendered_J = torch.clamp(render_pkg["rendered_J"], 0.0, 1.0)     
                    l1_curr = l1_loss(image, gt_image).mean().double()
                    l1_test += l1_curr
                    l2_curr = l2_loss(image, gt_image).mean().double()
                    l2_test += l2_curr
                    psnr_curr = psnr(image, gt_image).mean().double()
                    psnr_test += psnr_curr
    
                    if wandb.run is not None and (idx < 8):
                        def torch2pil(torch_image):
                            tr = torchvision.transforms.ToPILImage()
                            return tr(torch_image.squeeze(0).detach().cpu())
                        def wb_torch2np(torch_image):
                            rgb_image = torch_image.squeeze(0).detach().cpu().numpy()
                            if rgb_image.shape[0] in [1,2,3]:
                                rgb_image = np.transpose(rgb_image, (1, 2, 0)) # C*H*W -> H*W*C 
                            return rgb_image#bgr_image[..., ::-1] # Convert BGR (CV format) to RGB (WB format)
                        round_num = 5
                        wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/rendered": wandb.Image(wb_torch2np(image[None]), caption=f'l1 = {round(float(l1_curr),round_num)},  psnr = {round(float(psnr_curr),round_num)}')}, step=iteration)
                        try:
                                wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/rendered_depth": wandb.Image(apply_depth_colormap(render_pkg['rendered_depth'].detach().cpu().permute(1, 2, 0)).numpy(), caption=f'depth_map')}, step=iteration)
                                if uw_tuple[-1]: #removal_using_monocular_depth
                                    wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/rendered_depth_monocular": wandb.Image(apply_depth_colormap(depth_map_normalize.detach().cpu().unsqueeze(-1)).numpy(), caption=f'depth_map_monocular')}, step=iteration)
                        except:
                                pass
                        if uw_tuple[0].startswith("HYB"):
                            if iteration==testing_iterations[0] and viewpoint.mask is not None :
                                wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/mask": wandb.Image(wb_torch2np(viewpoint.mask), caption=f'mask')}, step=iteration)
                            # Convert tensors to NumPy arrays and flatten them
                            binf_values = uw_tuple[2]["binf"].detach().cpu().numpy().flatten()
                            bs_values = uw_tuple[2]["bs"].detach().cpu().numpy().flatten()
                            bd_values = uw_tuple[2]["direct"].detach().cpu().numpy().flatten()
                            rendered_J_caption = f'binf0: {round(float(binf_values[0]),round_num)}\n binf1: {round(float(binf_values[1]),round_num)}\n binf2: {round(float(binf_values[2]),round_num)}\n \
                            bs0: {round(float(bs_values[0]),round_num)}\n bs1: {round(float(bs_values[1]),round_num)}\n bs2: {round(float(bs_values[2]),round_num)}\n\
                            direct0: {round(float(bd_values[0]),round_num)}\n direct1: {round(float(bd_values[1]),round_num)}\n direct2: {round(float(bd_values[2]),round_num)}'
                            wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/rendered_J": wandb.Image(wb_torch2np(rendered_J[None]), caption=rendered_J_caption)}, step=iteration)
                            wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/rendered_backscattering": wandb.Image(wb_torch2np(rendered_backscattering[None]), caption=f'rendered_backscattering')}, step=iteration)
                            
                            
                            
                            # wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/Bs_histogram": wandb.Histogram(uw_tuple[2]["bs"].detach().cpu()),
                            #                     config['name'] + f"_view/image_{viewpoint.image_name}/Binf_histogram": wandb.Histogram(uw_tuple[2]["binf"].detach().cpu()),
                            #                     config['name'] + f"_view/image_{viewpoint.image_name}/Bd_histogram": wandb.Histogram(uw_tuple[2]["direct"].detach().cpu())
                            #                     }
                            #               ,step= iteration)
                            # if idx == 0 and config['name'] == "train" :
                            #     save_backscatter_plot(render_pkg["rendered_depth"], render_pkg["rendered_backscattering"], str_name=config['name'] + f"_view/image_{viewpoint.image_name}/bc_plot",
                            #                         wandb=wandb, iteration=iteration)
        

                        if iteration == testing_iterations[0]:
                            wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/ground-truth": wandb.Image(wb_torch2np(gt_image[None]), caption=f'gt')}, step=iteration)
                            

                    
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} L2 {} PSNR {}".format(iteration, config['name'], l1_test, l2_test, psnr_test))
                if wandb.run is not None:
                    wandb.run.log(data={config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                                        config['name'] + '/loss_viewpoint - l2_loss': l2_test,
                                        config['name'] + '/loss_viewpoint - psnr': psnr_test},step=iteration)
                    

            if wandb.run is not None:
                wandb.run.log(data={"scene/opacity_histogram": wandb.Histogram(scene.gaussians.get_opacity.detach().cpu()), 'total_points': scene.gaussians.get_xyz.shape[0]},step= iteration)
            
            if curr_bs_dict is not None and bool(curr_bs_dict):
                wandb.run.log(data={"underwater/backscattering_data/Bc_inf": curr_bs_dict['Bc_inf'].astype("float32"),
                                   "underwater/backscattering_data/beta_b": curr_bs_dict['betac_b'].astype("float32"),
                                    "underwater/backscattering_data/pdark": curr_bs_dict['pdark']},step= iteration)
        torch.cuda.empty_cache()
        
def organize_logger(cfg_args, action_groups):
    namespace_dict = {}
    for group in action_groups:
        group_dict = {action.dest: getattr(cfg_args, action.dest, None) for action in group._group_actions}
        namespace_dict[group.title.replace(' ', '')] = group_dict
    
    #namespace_dict["TrainingTime"] = {"minutes": 0, "seconds": 0} # updated when training is finished
    
    #cfg_dot_dic = NestedDict(namespace_dict)

    wb_resume, wb_use_logger, wb_project, wb_group, wb_dir, wb_id, next_filename, new_run = logger_aux(namespace_dict)
    if wb_use_logger:
        wandb.login()
        if new_run:
             # new run
            run = wandb.init(
                project=wb_project,
                group=wb_group,
                name=next_filename,
                dir=wb_dir,
                resume=wb_resume, 
                id =wb_id,  
                config=namespace_dict,
            )
        else:
            run = wandb.init(
                project=wb_project,
                group=wb_group,
                name=namespace_dict["WandbParameters"]["wb_name"],
                id= wb_id,
                resume=wb_resume,
                dir=wb_dir, 
                config=namespace_dict,
            )
        
    return wandb.config if wb_use_logger else namespace_dict

def logger_aux(namespace_dict):
    wb_resume = namespace_dict["WandbParameters"]["wb_resume"]
    wb_use_logger = namespace_dict["WandbParameters"]["wb_use_logger"]
    wb_project = namespace_dict["WandbParameters"]["wb_project"]
    wb_group = namespace_dict["WandbParameters"]["wb_group"]
    wb_dir = namespace_dict["WandbParameters"]["wb_dir"]
    wb_id = namespace_dict["WandbParameters"]["wb_id"]
    underwater_processing = namespace_dict["LoadingParameters"]["underwater_processing"]
    mcmc_mode = namespace_dict["LoadingParameters"]["densification"]
    source_path = namespace_dict["LoadingParameters"]["source_path"]
    iteration = namespace_dict["OptimizationParameters"]["iterations"]
    
    namespace_dict["WandbParameters"]["wb_name"] = namespace_dict["WandbParameters"]["wb_name"] + "_" + os.path.splitext(os.path.normpath(source_path).split(os.path.sep)[-1])[0] # base_name_without_extension
    wb_name = namespace_dict["WandbParameters"]["wb_name"]
    wb_group = wb_group + "_" + f"MCMode_{mcmc_mode}" + "_" + f"uw_{underwater_processing}"
    folder_path = os.path.join("./output/",  wb_group)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    else:
        print(f"Directory '{folder_path}' already exists.")
    files = os.listdir(folder_path)
    matching_files = [file for file in files if file.startswith(f"{wb_group}_{wb_name}_iter{iteration}")]
    numbers = [int(file.split("_")[-1]) for file in matching_files]
    if numbers:
        next_number = max(numbers) + 1
    else:
        next_number = 0  # If no files found, start with 0
    next_number_padded = str(next_number).zfill(5)
    next_filename = f"{wb_group}_{wb_name}_iter{iteration}_{next_number_padded}"
    new_run=False
    if not wb_resume in ["allow", "must", "auto"]:
        # new run
        namespace_dict["LoadingParameters"]["model_path"] = os.path.join(folder_path ,next_filename)
        new_run=True
    return wb_resume,wb_use_logger,wb_project,wb_group,wb_dir,wb_id,next_filename,new_run
        
        
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    wbp = WandbParams(parser)
    uwp = UnderwaterParams(parser)
    parser = add_general_params(parser)
    args = parser.parse_args(sys.argv[1:])
    if args.iterations not in args.test_iterations:
        args.test_iterations.append(args.iterations)
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    namespace_dict = organize_logger(cfg_args=args, action_groups=parser._action_groups)

    
    train_sweep(namespace_dict, sweep_train_flag=False)
    
    
    exit(code=0)