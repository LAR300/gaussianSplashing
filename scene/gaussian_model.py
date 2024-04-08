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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, uw_flag : str = "OFF"):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        # start addition uw
        self.uw_flag = uw_flag
        if uw_flag == "ON":
            self._featurescmed_dc = torch.empty(0)
            self._featurescmed_rest = torch.empty(0)
            self._opacitymed_dc = torch.empty(0)
            self._opacitymed_rest = torch.empty(0)
        elif uw_flag == "HYB":
            self._direct = torch.empty(0)
            self._binf = torch.empty(0)
            self._bs = torch.empty(0)
        # end addition uw
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        if self.uw_flag == "ON":
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
             # start addition uw
            self._featurescmed_dc,
            self._featurescmed_rest,
            self._opacitymed_dc,
            self._opacitymed_rest,
            # end addition uw
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
        if self.uw_flag == "HYB":
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
             # start addition uw
            self._direct,
            self._binf,
            self._bs,
            # end addition uw
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
        else:
           return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        ) 
        
    
    def restore(self, model_args, training_args):
        if self.uw_flag == "ON":
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            # start addition uw
            self._featurescmed_dc,
            self._featurescmed_rest,
            self._opacitymed_dc,
            self._opacitymed_rest,
            # end addition uw
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        if self.uw_flag == "HYB":
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            # start addition uw
            self._direct,
            self._binf,
            self._bs,
            # end addition uw
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        else:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
     # start addition uw
    @property
    def get_features_cmed(self):
        featurescmed_dc = self._featurescmed_dc
        featurescmed_rest = self._featurescmed_rest
        return torch.cat((featurescmed_dc, featurescmed_rest), dim=1)
    
    @property
    def get_opacity_med(self):
        opacitymed_dc = self._opacitymed_dc
        opacitymed_rest = self._opacitymed_rest   
        return torch.cat((opacitymed_dc, opacitymed_rest), dim=1) # (P,16,1)
    @property
    def get_direct(self):
        direct = self._direct
        return direct 
    @property
    def get_binf(self):
        binf = self._binf
        return binf 
    @property
    def get_bs(self):
        bs = self._bs
        return bs 
    # end addition uw
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            
    def cluster_aux(self, orig_colors):
        # cluster the points and colors
        pcds_dir = os.path.join("clustering","pcds")
        A = orig_colors
        print(A.shape)
        B = np.reshape(np.load(os.path.join(pcds_dir,"dpmeansMiniBatch", "train_preds_40247.npy")), (-1, 1))   
        print(B.shape)
        clusters = np.load(os.path.join(pcds_dir,"dpmeansMiniBatch", "train_clusters_40247.npy")) 
        print(clusters.shape)
        C = np.concatenate((A, B), axis=1)
        unique_values = np.unique(C[:, -1])

        new_colors = np.zeros((len(clusters), A.shape[1]))

        for i, value in enumerate(unique_values):
            # Select rows where the last column is equal to the current unique value
            group = A[B.flatten() == value]
            average_point = np.mean(group, axis=0)
            new_colors[i, :] = average_point
        return clusters, new_colors
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, initialization_points_option: str = "pointcloud"):
        #  Create the model from a point cloud
        # pcd : BasicPointCloud
        # pcd.points : numpy array of shape (N, 3)
        # pcd.colors : numpy array of shape (N, 3)
        # pcd.normals : numpy array of shape (N, 3), optional
        
        if initialization_points_option == "random":
            N = 20_000 # pcd.points.shape[0]  
            lower_bound = -250
            upper_bound = 250
            random_array = np.random.uniform(lower_bound, upper_bound, size=(N, 3))
            points = random_array
            colors = np.random.uniform(0, 1, size=(N, 3))
        elif initialization_points_option == "cluster":  
            points, colors = self.cluster_aux(pcd.colors)
            #colors = pcd.colors
        else: # if initialization_points_option == "pointcloud":
            points = pcd.points
            colors = pcd.colors # min=0 max=1 (P,3)
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # (P,3,16)
        features[:, :3, 0 ] = fused_color # (P,3,1) or (P,3), first coefficient represents the DC
        features[:, 3:, 1:] = 0.0 # (P,0,15)

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(points)).float().cuda()), 0.0000001) # the average distance between point pi to its 3 nearest neighbors
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) 
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # (P,1,3), first coefficient represents the DC
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # (P,15,3), the rest of the coefficients
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if self.uw_flag == "ON":
            # start addition uw
            fused_colorcmed = RGB2SH(torch.tensor(np.asarray([[0.1,0.1,0.1]])).float().cuda())
            featurecmed = torch.zeros((1, 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # (1,3,16)
            featurecmed[:, :3, 0 ] = fused_colorcmed # (1,3,1) or (1,3), first coefficient represents the DC
            featurecmed[:, 3:, 1:] = 0.0 # (1,3,15)
            self._featurescmed_dc = nn.Parameter(featurecmed[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # (1,1,3), first coefficient represents the DC
            self._featurescmed_rest = nn.Parameter(featurecmed[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # (1,15,3), the rest of the coefficients
            
            fused_opacitymed = RGB2SH(torch.tensor(np.asarray([[0.01]])).float().cuda())
            featuresopmed = torch.zeros((fused_point_cloud.shape[0], 1, (self.max_sh_degree + 1) ** 2)).float().cuda() # (P,1,16)
            featuresopmed[:, :1, 0 ] = fused_opacitymed # (P,1,1) or (P,1), first coefficient represents the DC
            featuresopmed[:, 1:, 1:] = 0.0 # (P,0,15)
            self._opacitymed_dc = nn.Parameter(featuresopmed[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # (P,1,1), first coefficient represents the DC
            self._opacitymed_rest = nn.Parameter(featuresopmed[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # (P,15,1), the rest of the coefficients
            # end addition uw
        
        elif self.uw_flag == "HYB":
            # start addition uw
            self._direct = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 3), device="cuda").contiguous().requires_grad_(True)) # (P,3)
            self._binf = nn.Parameter(torch.zeros((1, 3), device="cuda").contiguous().requires_grad_(True)) # (1, 3)
            self._bs = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 3), device="cuda").contiguous().requires_grad_(True)) # (P,3)
            # end addition uw

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if self.uw_flag == "ON":
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                # start addition uw
                {'params': [self._featurescmed_dc], 'lr': training_args.feature_lr, "name": "fcmed_dc"}, # using the same learning rate as f_dc
                {'params': [self._featurescmed_rest], 'lr': training_args.feature_lr / 20.0, "name": "fcmed_rest"}, # using the same learning rate as f_rest
                
                {'params': [self._opacitymed_dc], 'lr': training_args.feature_lr, "name": "opacitymed_dc"}, # using the same learning rate as f_dc (sh representation)
                {'params': [self._opacitymed_rest], 'lr': training_args.feature_lr / 20.0, "name": "opacitymed_rest"}, # using the same learning rate as f_rest (sh representation)
                # end addition uw
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
        elif self.uw_flag == "HYB":
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                # start addition uw
                {'params': [self._direct], 'lr': training_args.feature_lr, "name": "direct"}, # using the same learning rate 
                {'params': [self._binf], 'lr': training_args.feature_lr , "name": "binf"}, # using the same learning rate 
                {'params': [self._bs], 'lr': training_args.feature_lr, "name": "bs"}, # using the same learning rate 
                # end addition uw
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        if self.uw_flag == "ON":
            # start addition uw
            for i in range(self._featurescmed_dc.shape[1]*self._featurescmed_dc.shape[2]):
                l.append('fcmed_dc_{}'.format())
            for i in range(self._featurescmed_rest.shape[1]*self._featurescmed_rest.shape[2]):
                l.append('fcmed_rest_{}'.format(i))
                
            for i in range(self._opacitymed_dc.shape[1]*self._opacitymed_dc.shape[2]):
                l.append('opacitymed_dc_{}'.format())
            for i in range(self._opacitymed_rest.shape[1]*self._opacitymed_rest.shape[2]):
                l.append('opacitymed_rest_{}'.format(i))
            # end addition uw
        elif self.uw_flag == "HYB":
            # start addition uw
            for i in range(self._direct.shape[1]):
                l.append('direct_{}'.format(i))
            for i in range(self._binf.shape[1]):
                l.append('binf_{}'.format(i))
            for i in range(self._bs.shape[1]):
                l.append('bs_{}'.format(i))
            # end addition uw
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        if self.uw_flag == "ON":
            # start addition uw
            fcmed_dc = self._featurescmed_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            fcmed_rest = self._featurescmed_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            
            opacitymed_dc = self._opacitymed_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacitymed_rest = self._opacitymed_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            # end addition uw
        elif self.uw_flag == "HYB":
            # start addition uw
            direct = self._direct.detach().cpu().numpy()
            binf = self._binf.detach().cpu().numpy()
            bs = self._bs.detach().cpu().numpy()
            # end addition uw
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.uw_flag == "ON":
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, fcmed_dc, fcmed_rest, opacitymed_dc, opacitymed_rest, opacities, scale, rotation), axis=1) # with uw addition
        elif self.uw_flag == "HYB":
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, direct, np.broadcast_to(binf, (xyz.shape[0],) + binf.shape[1:]), bs, opacities, scale, rotation), axis=1) # with uw addition
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1) # without uw addition
        
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    # start addition uw
    def reset_opacity_med(self):
        fused_opacitymed = RGB2SH(torch.tensor(np.asarray([[0.01]])).float().cuda())
        featuresopmed = torch.zeros((self.get_xyz.shape[0], 1, (self.max_sh_degree + 1) ** 2)).float().cuda() # (P,1,16)
        featuresopmed[:, :1, 0 ] = fused_opacitymed # (P,1,1) or (P,1), first coefficient represents the DC
        featuresopmed[:, 1:, 1:] = 0.0 # (P,0,15)
        _featuresopmed = torch.cat((featuresopmed[:,:,0:1].transpose(1, 2).contiguous(), featuresopmed[:,:,1:].transpose(1, 2).contiguous()), dim=1)
        opacitymed_gath = torch.where(torch.abs(self.get_opacity_med) < torch.abs(_featuresopmed), self.get_opacity_med, _featuresopmed)
        
        featuresopmed_dc = opacitymed_gath[:,:1,:].transpose(1, 2).contiguous()
        featuresopmed_rest = opacitymed_gath[:,1:,:].transpose(1, 2).contiguous()
        optimizable_tensors = self.replace_tensor_to_optimizer(featuresopmed_dc, "opacitymed_dc")
        self._opacitymed_dc = optimizable_tensors["opacitymed_dc"]
        optimizable_tensors = self.replace_tensor_to_optimizer(featuresopmed_rest, "opacitymed_rest")
        self._opacitymed_rest = optimizable_tensors["opacitymed_rest"]
    def reset_hyb(self):
        direct_new = torch.min(self.get_direct, torch.ones_like(self.get_direct)*0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(direct_new, "direct")
        bs_new = torch.min(self.get_bs, torch.ones_like(self.get_bs)*0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(bs_new, "bs")
        
        self._direct = optimizable_tensors["direct"]
        self._bs = optimizable_tensors["bs"]
    # end addition uw
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if self.uw_flag == "ON":
            # start addition uw
            featurescmed_dc = np.zeros((1, 3, 1))
            featurescmed_dc[:, 0, 0] = np.asarray(plydata.elements[0]["fcmed_dc_0"])
            featurescmed_dc[:, 1, 0] = np.asarray(plydata.elements[0]["fcmed_dc_1"])
            featurescmed_dc[:, 2, 0] = np.asarray(plydata.elements[0]["fcmed_dc_2"])
            extra_f_namescmed = [p.name for p in plydata.elements[0].properties if p.name.startswith("fcmed_rest_")]
            extra_f_namescmed = sorted(extra_f_namescmed, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_namescmed)==3*(self.max_sh_degree + 1) ** 2 - 3
            features_extracmed = np.zeros((1, len(extra_f_namescmed)))
            for idx, attr_name in enumerate(extra_f_namescmed):
                features_extracmed[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extracmed = features_extracmed.reshape((features_extracmed.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
            self._featurescmed_dc = nn.Parameter(torch.tensor(featurescmed_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._featurescmed_rest = nn.Parameter(torch.tensor(features_extracmed, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            
            
            opacitymed_dc = np.zeros((xyz.shape[0], 3, 1))
            opacitymed_dc[:, 0, 0] = np.asarray(plydata.elements[0]["opacitymed_dc_0"])
            extra_omed_namescmed = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacitymed_rest_")]
            extra_omed_namescmed = sorted(extra_omed_namescmed, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_omed_namescmed)==1*(self.extra_omed_namescmed + 1) ** 2 - 1
            omed_extra = np.zeros((1, len(extra_omed_namescmed)))
            for idx, attr_name in enumerate(extra_f_namescmed):
                omed_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            omed_extra = omed_extra.reshape((omed_extra.shape[0], 1, (self.max_sh_degree + 1) ** 2 - 1))
            self._opacitymed_dc = nn.Parameter(torch.tensor(opacitymed_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._opacitymed_rest = nn.Parameter(torch.tensor(omed_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            # end addition uw
        elif self.uw_flag == "HYB":
            # start addition uw
            dierct = np.asarray(plydata.elements[0]["direct"])[..., np.newaxis]
            binf = np.asarray(plydata.elements[0]["binf"])[..., np.newaxis]
            bs = np.asarray(plydata.elements[0]["bs"])[..., np.newaxis]
            self._direct = nn.Parameter(torch.tensor(dierct, dtype=torch.float, device="cuda").requires_grad_(True))
            self._binf = nn.Parameter(torch.tensor(binf, dtype=torch.float, device="cuda").requires_grad_(True))
            self._bs = nn.Parameter(torch.tensor(bs, dtype=torch.float, device="cuda").requires_grad_(True))
            # end addition uw
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    #######################################################################################################################################
#### Densification (pruning, cloning, splitting) doesn't need color med, it is a structure property and not a gaussian property
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "binf": # binf is a general parameter
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        if self.uw_flag == "ON":
            # start addition uw
            self._featurescmed_dc = optimizable_tensors["fcmed_dc"]
            self._featurescmed_rest = optimizable_tensors["fcmed_rest"]
            
            self._opacitymed_dc = optimizable_tensors["opacitymed_dc"]
            self._opacitymed_rest = optimizable_tensors["opacitymed_rest"]
            # end addition uw
        if self.uw_flag == "HYB":
            # start addition uw
            self._direct = optimizable_tensors["direct"]
            # self._binf = optimizable_tensors["binf"]
            self._bs = optimizable_tensors["bs"]
            # end addition uw
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1

            if group["name"] == "binf": # binf is a general parameter
                continue
            
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    def densification_postfix(self, new_xyz, 
                              new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    # start addition uw
    def densification_postfix_uw(self, new_xyz, 
                              new_features_dc, new_features_rest,
                              new_featurescmed_dc, new_featurescmed_rest, # with uw addition
                              new_opacitymed_dc, new_opacitymed_rest,# with uw addition
                              new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        # start addition uw
        "fcmed_dc": new_featurescmed_dc,
        "fcmed_rest": new_featurescmed_rest,
        "opacitymed_dc": new_opacitymed_dc,
        "opacitymed_rest": new_opacitymed_rest,
        # end addition uw
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        # start addition uw
        self._featurescmed_dc = optimizable_tensors["fcmed_dc"]
        self._featurescmed_rest = optimizable_tensors["fcmed_rest"]
        self._opacitymed_dc = optimizable_tensors["opacitymed_dc"]
        self._opacitymed_rest = optimizable_tensors["opacitymed_rest"]
        # end addition uw
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    # end addition uw
    # start addition uw
    def densification_postfix_uwhyb(self, new_xyz, 
                              new_features_dc, new_features_rest,
                              new_direct, new_bs, # with uw addition
                              new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        # start addition uw
        "direct": new_direct,
        #"binf": new_binf,
        "bs": new_bs,
        # end addition uw
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        # start addition uw
        self._direct = optimizable_tensors["direct"]
        #self._binf = optimizable_tensors["binf"]
        self._bs = optimizable_tensors["bs"]
        # end addition uw
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    # end addition uw
    
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        if self.uw_flag == "ON":
            # start addition uw
            new_featurescmed_dc = self._featurescmed_dc[selected_pts_mask].repeat(N,1,1)
            new_featurescmed_rest = self._featurescmed_rest[selected_pts_mask].repeat(N,1,1)
            
            new_opacitymed_dc = self._opacitymed_dc[selected_pts_mask].repeat(N,1,1)
            new_opacitymed_rest = self._opacitymed_rest[selected_pts_mask].repeat(N,1,1)
            # end addition uw
        elif self.uw_flag == "HYB":
            # start addition uw
            new_direct = self._direct[selected_pts_mask].repeat(N,1)
            #new_binf = self._binf
            new_bs = self._bs[selected_pts_mask].repeat(N,1)
            # end addition uw
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        
        if self.uw_flag == "ON":
            self.densification_postfix_uw(new_xyz, new_features_dc, new_features_rest, new_featurescmed_dc, new_featurescmed_rest, new_opacitymed_dc, new_opacitymed_rest, new_opacity, new_scaling, new_rotation) # with uw addition
        elif self.uw_flag == "HYB":
            self.densification_postfix_uwhyb(new_xyz, new_features_dc, new_features_rest, new_direct, new_bs, new_opacity, new_scaling, new_rotation) # with uw addition
        else:
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation) # without uw addition
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        if self.uw_flag == "ON":
            # start addition uw
            #new_featurescmed_dc = self._featurescmed_dc[selected_pts_mask]
            new_featurescmed_dc = self._featurescmed_dc
            #new_featurescmed_rest = self._featurescmed_rest[selected_pts_mask]
            new_featurescmed_rest = self._featurescmed_rest
            
            new_opacitymed_dc = self._opacitymed_dc[selected_pts_mask]
            new_opacitymed_rest = self._opacitymed_rest[selected_pts_mask]
            # end addition uw
        elif self.uw_flag == "HYB":
            # start addition uw
            new_direct = self._direct[selected_pts_mask]
            #new_binf = self._binf
            new_bs = self._bs[selected_pts_mask]
            # end addition uw
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        if self.uw_flag == "ON":
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_featurescmed_dc, new_featurescmed_rest, new_opacitymed_dc, new_opacitymed_rest, new_opacities, new_scaling, new_rotation) # with uw addition
        elif self.uw_flag == "HYB":
            self.densification_postfix_uwhyb(new_xyz, new_features_dc, new_features_rest, new_direct, new_bs, new_opacities, new_scaling, new_rotation) # with uw addition
        else:
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation) # without uw addition
        

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1