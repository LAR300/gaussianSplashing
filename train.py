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
from arguments import ModelParams, PipelineParams, OptimizationParams, WandbParams, DotDict, add_general_params, write_namespace_to_file, read_namespace_from_file
import wandb

def training(args, dataset, opt, pipe, testing_iterations, saving_model_iterations, checkpoint_iterations, checkpoint, debug_from, action_groups):
    start_training = time.time()
    first_iter = 0
    prepare_output(args)
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
    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

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
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
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
            with torch.no_grad():
                training_report(iteration, Ll1, loss, l1_loss,
                                Ll2, l2_loss, Psnr,
                                iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_model_iterations):
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

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    end_training = time.time()
    elapsed_training_time = end_training - start_training
    minutes, seconds = divmod(elapsed_training_time, 60)
    
    prepare_output_end(args, action_groups, minutes, seconds)
    if wandb.run is not None:
        wandb.finish(exit_code=0)
    return minutes, seconds

def prepare_output(cfg_args):           
    # Set up output folder
    print("Output folder: {}".format(cfg_args.model_path))
    os.makedirs(cfg_args.model_path, exist_ok = True)
    
def prepare_output_end(cfg_args, action_groups, minutes, seconds):
    # Organize arguments into a dictionary by groups
    namespace_dict = {}
    for group in action_groups:
        group_dict = {action.dest: getattr(cfg_args, action.dest, None) for action in group._group_actions}
        namespace_dict[group.title] = group_dict
    namespace_dict["elapsed_training_time_MS"] = {"minutes": minutes, "seconds": seconds}
    write_namespace_to_file(namespace_dict, os.path.join(cfg_args.model_path,"cfg_args.json"))
    if wandb.run is not None:
        wandb.config.update(namespace_dict)

def training_report(iteration, Ll1, loss, l1_loss,
                    Ll2, l2_loss, Psnr,
                    elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):

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
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
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
                        def torch2np(torch_image):
                            nm = torch_image.squeeze(0).detach().cpu().numpy()
                            return np.transpose(nm, (1, 2, 0)) # C*H*W -> H*W*C 
                        round_num = 5
                        wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/rendered": wandb.Image(torch2np(image[None]), caption=f'l1 = {round(float(l1_curr),round_num)},  psnr = {round(float(psnr_curr),round_num)}')}, step=iteration)
                        if iteration == testing_iterations[0]:
                            wandb.run.log(data={config['name'] + f"_view/image_{viewpoint.image_name}/ground-truth": wandb.Image(torch2np(gt_image[None]), caption=f'gt')}, step=iteration)
                    
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} L2 {} PSNR {}".format(iteration, config['name'], l1_test, l2_test, psnr_test))
                if wandb.run is not None:
                    wandb.run.log(data={config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                                        config['name'] + '/loss_viewpoint - l2_loss': l2_test,
                                        config['name'] + '/loss_viewpoint - psnr': psnr_test},step=iteration)

        if wandb.run is not None:
            wandb.run.log(data={"scene/opacity_histogram": wandb.Histogram(scene.gaussians.get_opacity.detach().cpu()), 'total_points': scene.gaussians.get_xyz.shape[0]},step= iteration)
        torch.cuda.empty_cache()
        
def create_wb_logger(cfg):
    dot_dic=DotDict(vars(cfg))
    wandb.login()
    
    if not cfg.wb_resume:
        folder_path = os.path.join("./output/",  cfg.wb_group)
        files = os.listdir(folder_path)
        cfg.wb_name = os.path.splitext(os.path.normpath(cfg.source_path).split(os.path.sep)[-1])[0] # base_name_without_extension
        matching_files = [file for file in files if file.startswith(f"{cfg.wb_name}_")]
        numbers = [int(file.split("_")[1]) for file in matching_files]
        if numbers:
            next_number = max(numbers) + 1
        else:
            next_number = 0  # If no files found, start with 0
        next_number_padded = str(next_number).zfill(5)
        next_filename = f"{cfg.wb_name}_{next_number_padded}"
        cfg.model_path = os.path.join(folder_path ,next_filename)
        
    run = wandb.init(
        project=dot_dic.wb_project,
        group=dot_dic.wb_group,
        name=next_filename,
        dir=dot_dic.wb_dir,
        resume=dot_dic.wb_resume, 
        id = dot_dic.wb_id,  
        config=dot_dic,
)
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    wbp = WandbParams(parser)
    parser = add_general_params(parser)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.wb_use_logger:
        create_wb_logger(cfg=args)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training_miutes, training_seconds = training(args, lp.extract(args), op.extract(args), pp.extract(args), 
                                                 args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
                                                 args.start_checkpoint, args.debug_from,
                                                 parser._action_groups)
    
    
    # All done
    print("Training complete.")
    print(f"Training execution time: {int(training_miutes)} minutes {training_seconds:.2f} seconds")
    exit(code=0)
