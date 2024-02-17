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
from utils.loss_utils import l1_loss, l2_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import time
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, WandbParams, UnderwaterParams, NestedDict, add_general_params, write_dictionary_to_file
import wandb
from uw_formation.depthmap import depth_estimation
from uw_formation.backscatter import create_backscatter_image, preform_full_bs_removal, preform_full_bs_addition


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
    if cfg_dot_dic.GeneralParameters.ip != "" and cfg_dot_dic.GeneralParameters.port != "":
        network_gui.init(cfg_dot_dic.GeneralParameters.ip, cfg_dot_dic.GeneralParameters.port)
    torch.autograd.set_detect_anomaly(cfg_dot_dic.GeneralParameters.detect_anomaly)
    
    training_miutes, training_seconds = training(cfg_dot_dic ,cfg_dot_dic.LoadingParameters, cfg_dot_dic.OptimizationParameters, cfg_dot_dic.PipelineParameters,  cfg_dot_dic.UnderwaterParameters,
                                                 sweep_train_flag=sweep_train_flag)
    # All done
    print("Training complete.")
    print(f"Training execution time: {int(training_miutes)} minutes {training_seconds:.2f} seconds")
    
def training(cfg_dot_dic, dataset, opt, pipe, uw, checkpoint=None, sweep_train_flag=False):
    
    start_training = time.time()
    first_iter = 0
    prepare_output(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
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
    # for UW processing
    curr_bs_dict = None
    for iteration in range(first_iter, opt.iterations + 1):        
        if cfg_dot_dic.GeneralParameters.ip != "" and cfg_dot_dic.GeneralParameters.port != "":
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

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

        bg = torch.rand((3), device="cuda") if opt.random_background else background # [0,0,0] msot of the times

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, depth_map, alpha_map, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["rendered_depth"], render_pkg["rendered_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Underwater backscatter removal
        if dataset.underwater_processing=="ON":
            
            if uw.remove_bs_from_images:
                if uw.removal_using_monocular_depth:
                    # gt image without bs, using monocular depth
                    if viewpoint_cam.original_image_without_bs is None:
                        
                        if curr_bs_dict is None:
                            gt_image_without_bs, bs_image, bs_dict, image_depth = preform_full_bs_removal(image=viewpoint_cam.original_image.cpu().numpy(), return_depth=True)
                            gt_image_without_bs = torch.from_numpy(gt_image_without_bs).float().cuda()
                            
                            curr_bs_dict = bs_dict
                            viewpoint_cam.original_image_without_bs = gt_image_without_bs
                            
                            viewpoint_cam.image_depth_monocular = image_depth
                        else:
                            gt_image_without_bs, bs_image, bs_dict = preform_full_bs_removal(image=viewpoint_cam.original_image.cpu().numpy(), bs_dict=curr_bs_dict, image_depth=viewpoint_cam.image_depth_monocular)
                            gt_image_without_bs = torch.from_numpy(gt_image_without_bs).float().cuda()
                            # curr_bs_dict = bs_dict # no need, it is the same as before
                            viewpoint_cam.original_image_without_bs = gt_image_without_bs
                    gt_image = viewpoint_cam.original_image_without_bs.cuda()
                else:
                    if iteration % uw.bs_removal_every == 0:   
                        # gt image without bs, using gs depth iteratively
                        gt_image_without_bs, bs_image, bs_dict = preform_full_bs_removal(image=viewpoint_cam.original_image.cpu().numpy() ,image_depth=render_pkg["rendered_depth"].detach().cpu().numpy().squeeze(axis=0))
                        gt_image_without_bs = torch.from_numpy(gt_image_without_bs).float().cuda()
                        
                        curr_bs_dict= bs_dict
                        viewpoint_cam.original_image_without_bs = gt_image_without_bs
                        
                        gt_image = viewpoint_cam.original_image_without_bs.cuda()
                    else:
                        if curr_bs_dict is None:
                            gt_image = viewpoint_cam.original_image.cuda()
                        else:
                            gt_image_without_bs, bs_image, bs_dict = preform_full_bs_removal(image=viewpoint_cam.original_image.cpu().numpy(), bs_dict=curr_bs_dict ,image_depth=render_pkg["rendered_depth"].detach().cpu().numpy().squeeze(axis=0))
                            gt_image_without_bs = torch.from_numpy(gt_image_without_bs).float().cuda()
                            
                            # curr_bs_dict = bs_dict # no need, it is the same as before
                            viewpoint_cam.original_image_without_bs = gt_image_without_bs
                            
                            gt_image = viewpoint_cam.original_image_without_bs.cuda()
                        
            else:
                gt_image = viewpoint_cam.original_image.cuda()
        else:
            gt_image = viewpoint_cam.original_image.cuda()
        # Loss
        Ll1 = l1_loss(image, gt_image)
        Ll2 = l2_loss(image, gt_image).detach()
        Psnr = psnr(image, gt_image).mean().double().detach()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

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
            training_report(iteration, Ll1, loss, l1_loss,
                            Ll2, l2_loss, Psnr,
                            iter_start.elapsed_time(iter_end), cfg_dot_dic.GeneralParameters.test_iterations, scene, render, (pipe, background),
                            (dataset.underwater_processing=="ON",uw), curr_bs_dict)
            if (iteration in cfg_dot_dic.GeneralParameters.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in cfg_dot_dic.GeneralParameters.checkpoint_iterations and not sweep_train_flag):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    end_training = time.time()
    elapsed_training_time = end_training - start_training
    minutes, seconds = divmod(elapsed_training_time, 60)
    
    prepare_output_end(cfg_dot_dic, minutes, seconds, sweep_train_flag)
    if wandb.run is not None:
        wandb.finish(exit_code=0)
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

def training_report(iteration, Ll1, loss, l1_loss,
                    Ll2, l2_loss, Psnr,
                    elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, uw_tuple=(False,None), curr_bs_dict=None):

    if wandb.run is not None:
        wandb.run.log(data={'train_loss_patches/l1_loss': Ll1.item(), 'train_loss_patches/total_loss': loss.item(),
                            'train_loss_patches/l1_loss': Ll2.item(), 'train_loss_patches/l1_loss': Psnr.item(),
                            'time/iter_time': elapsed}, step=iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                l2_test = 0.0
                psnr_test = 0.0
                
                uw_l1_test = 0.0
                uw_l2_test = 0.0
                uw_psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    depth_map = torch.clip((render_pkg["rendered_depth"] - render_pkg["rendered_depth"].min()) / (render_pkg["rendered_depth"].max() - render_pkg["rendered_depth"].min()), 0.0, 1.0)
                    alpha_map = torch.clamp((render_pkg["rendered_alpha"] - render_pkg["rendered_alpha"].min()) / (render_pkg["rendered_alpha"].max() - render_pkg["rendered_alpha"].min()), 0.0, 1.0)
                    gt_image = torch.clip(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    depth_map_unnormalize = render_pkg["rendered_depth"].detach().cpu().numpy().squeeze(axis=0)
                    
                    if uw_tuple[0] and uw_tuple[1].removal_using_monocular_depth:
                        if viewpoint.image_depth_monocular is None:
                            depth_map_unnormalize = depth_estimation(viewpoint.original_image.cpu().numpy())
                            viewpoint.image_depth_monocular = depth_map_unnormalize
                        else:
                            depth_map_unnormalize = viewpoint.image_depth_monocular
                        
                    if uw_tuple[0] and uw_tuple[1].remove_bs_from_images and curr_bs_dict is not None:
                        if viewpoint.original_image_without_bs is not None:
                            gt_Ic = preform_full_bs_addition(Dc=viewpoint.original_image_without_bs.cpu().numpy(), bs_dict=curr_bs_dict, image_depth=depth_map_unnormalize)
                            gt_Ic = torch.from_numpy(gt_Ic).float().cuda()
                            gt_Ic = torch.clamp(gt_Ic, 0.0, 1.0)
                        else:
                            gt_Ic = torch.clip(viewpoint.original_image.to("cuda"), 0.0, 1.0) # training didn't change the image, using the original signal from the dataset
                        
                        Ic_rendered = preform_full_bs_addition(Dc=image.cpu().numpy(), bs_dict=curr_bs_dict, image_depth=depth_map_unnormalize)
                        Ic_rendered = torch.from_numpy(Ic_rendered).float().cuda()
                        Ic_rendered = torch.clamp(Ic_rendered, 0.0, 1.0)
                        
                        rendered_bs_image = create_backscatter_image(z=depth_map_unnormalize, bs_dict=curr_bs_dict)
                        rendered_bs_image = torch.from_numpy(rendered_bs_image).float()
                        rendered_bs_image = torch.clamp(rendered_bs_image, 0.0, 1.0)
                        
                        uw_l1_curr = l1_loss(Ic_rendered, gt_image).mean().double()
                        uw_l1_test += uw_l1_curr
                        uw_l2_curr = l2_loss(Ic_rendered, gt_image).mean().double()
                        uw_l2_test += uw_l2_curr
                        uw_psnr_curr = psnr(Ic_rendered, gt_image).mean().double()
                        uw_psnr_test += uw_psnr_curr
                     
                     
                            
                    l1_curr = l1_loss(image, gt_image).mean().double()
                    l1_test += l1_curr
                    l2_curr = l2_loss(image, gt_image).mean().double()
                    l2_test += l2_curr
                    psnr_curr = psnr(image, gt_image).mean().double()
                    psnr_test += psnr_curr
                    
                    
                    
                    
                    
                    if wandb.run is not None and (idx < 5):
                        def torch2pil(torch_image):
                            tr = torchvision.transforms.ToPILImage()
                            return tr(torch_image.squeeze(0).detach().cpu())
                        def wb_torch2np(torch_image):
                            rgb_image = torch_image.squeeze(0).detach().cpu().numpy()
                            if rgb_image.shape[0] in [1,2,3]:
                                rgb_image = np.transpose(rgb_image, (1, 2, 0)) # C*H*W -> H*W*C 
                            return rgb_image#bgr_image[..., ::-1] # Convert BGR (CV format) to RGB (WB format)
                        round_num = 5
                        dc_str = '  ||  !only Dc!' if curr_bs_dict is not None else ''
                        wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/rendered": wandb.Image(wb_torch2np(image[None]), caption=f'l1 = {round(float(l1_curr),round_num)},  psnr = {round(float(psnr_curr),round_num)}{dc_str}')}, step=iteration)
                        
                        wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/rendered_depth": wandb.Image(wb_torch2np(depth_map[None]), caption=f'depth_map')}, step=iteration)
                        wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/rendered_alpha": wandb.Image(wb_torch2np(alpha_map[None]), caption=f'alpha_map')}, step=iteration)
                        
                        
                        if uw_tuple[0] and uw_tuple[1].remove_bs_from_images and curr_bs_dict is not None: 
                            wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/ground-truth_I_c": wandb.Image(wb_torch2np(gt_Ic[None]), caption=f'gt_Ic')}, step=iteration)
                            wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/rendered_Bs": wandb.Image(wb_torch2np(rendered_bs_image[None]), caption=f'backscattering_image')}, step=iteration)
                            
                            wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/rendered_I_c": wandb.Image(wb_torch2np(Ic_rendered[None]), caption=f'l1 = {round(float(uw_l1_curr),round_num)},  psnr = {round(float(uw_psnr_curr),round_num)}')},step=iteration)
                            
                            if viewpoint.original_image_without_bs is not None:
                                gt_image_Dc = torch.clip(viewpoint.original_image_without_bs.to("cuda"), 0.0, 1.0)
                                wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/ground-truth_Dc": wandb.Image(wb_torch2np(gt_image_Dc[None]), caption=f'gt_Dc')}, step=iteration)
                            else:
                                gt_image_Dc, _, _ = preform_full_bs_removal(image=viewpoint.original_image.cpu().numpy(), bs_dict=curr_bs_dict, image_depth=render_pkg["rendered_depth"].detach().cpu().numpy().squeeze(axis=0))
                                gt_image_Dc = torch.from_numpy(gt_image_Dc).float()
                                gt_image_Dc = torch.clamp(gt_image_Dc, 0.0, 1.0)
                                wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/ground-truth_Dc": wandb.Image(wb_torch2np(gt_image_Dc[None]), caption=f'gt_Dc')}, step=iteration)
                            
                            
                            
                            
                            
                            
                            
                            
                            
                        if iteration == testing_iterations[0]:
                            wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/ground-truth": wandb.Image(wb_torch2np(gt_image[None]), caption=f'gt')}, step=iteration)
                            
                            if uw_tuple[0] and uw_tuple[1].removal_using_monocular_depth:
                                depth_map_unnormalize = np.expand_dims(depth_map_unnormalize, axis=2)
                                wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/monocular-depth": wandb.Image(
                                      np.clip((depth_map_unnormalize - depth_map_unnormalize.min()) / (depth_map_unnormalize.max() - depth_map_unnormalize.min()), 0.0, 1.0)
                                      , caption=f'monocular_depth')}, step=iteration)
                            
                            
                    
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} L2 {} PSNR {}".format(iteration, config['name'], l1_test, l2_test, psnr_test))
                if wandb.run is not None:
                    wandb.run.log(data={config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                                        config['name'] + '/loss_viewpoint - l2_loss': l2_test,
                                        config['name'] + '/loss_viewpoint - psnr': psnr_test},step=iteration)
                    
                    if curr_bs_dict is not None:
                        wandb.run.log(data={config['name'] + '/uw_loss_viewpoint - l1_loss': uw_l1_test,
                                        config['name'] + '/uw_loss_viewpoint - l2_loss': uw_l2_test,
                                        config['name'] + '/uw_loss_viewpoint - psnr': uw_psnr_test},step=iteration)

        if wandb.run is not None:
            wandb.run.log(data={"scene/opacity_histogram": wandb.Histogram(scene.gaussians.get_opacity.detach().cpu()), 'total_points': scene.gaussians.get_xyz.shape[0]},step= iteration)
            
            if curr_bs_dict is not None:
                wandb.run.log(data={"undewater/backscattering/Bc_inf": curr_bs_dict['Bc_inf'].astype("float32"),
                                   "undewater/backscattering/beta_b": curr_bs_dict['betac_b'].astype("float32"),
                                    "undewater/backscattering/pdark": curr_bs_dict['pdark']},step= iteration)
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
    source_path = namespace_dict["LoadingParameters"]["source_path"]
    iteration = namespace_dict["OptimizationParameters"]["iterations"]
    folder_path = os.path.join("./output/",  namespace_dict["WandbParameters"]["wb_group"])
    namespace_dict["WandbParameters"]["wb_name"] = os.path.splitext(os.path.normpath(source_path).split(os.path.sep)[-1])[0] # base_name_without_extension
    wb_name = namespace_dict["WandbParameters"]["wb_name"]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    else:
        print(f"Directory '{folder_path}' already exists.")
    files = os.listdir(folder_path)
    matching_files = [file for file in files if file.startswith(f"{wb_name}_")]
    numbers = [int(file.split("_")[-1]) for file in matching_files]
    if numbers:
        next_number = max(numbers) + 1
    else:
        next_number = 0  # If no files found, start with 0
    next_number_padded = str(next_number).zfill(5)
    next_filename = f"{wb_name}_iter{iteration}_{next_number_padded}"
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