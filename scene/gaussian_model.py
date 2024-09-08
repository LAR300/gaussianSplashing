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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, process_mean3d_using_camera_mask
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation,  get_minimum_axis, flip_align_view
from utils.reloc_utils import compute_relocation_cuda


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
        if uw_flag.startswith("HYB"):
            self._direct = torch.empty(0)
            self._binf = torch.empty(0)
            self._featuresbinf_dc = torch.empty(0)
            self._featuresbinf_rest = torch.empty(0)
            self._featuresbs_dc = torch.empty(0)
            self._featuresbs_rest = torch.empty(0)
            self._featuresdirect_dc = torch.empty(0)
            self._featuresdirect_rest = torch.empty(0)
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
        if self.uw_flag.startswith("HYB"):
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
             # start addition uw
            self._direct,
            self._binf,
            self._featuresbinf_dc,
            self._featuresbinf_rest,
            self._featuresbs_dc,
            self._featuresbs_rest,
            self._featuresdirect_dc,
            self._featuresdirect_rest,
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
        if self.uw_flag.startswith("HYB"):
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            # start addition uw
            self._direct,
            self._binf,
            self._featuresbinf_dc,
            self._featuresbinf_rest,
            self._featuresbs_dc,
            self._featuresbs_rest,
            self._featuresdirect_dc,
            self._featuresdirect_rest,
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
    def get_direct(self):
        direct = self._direct
        return direct 
    @property
    def get_binf(self):
        binf = self._binf
        return binf 
    @property
    def get_featuresbinf(self):
        featuresbinf_dc = self._featuresbinf_dc
        featuresbinf_rest = self._featuresbinf_rest
        return torch.cat((featuresbinf_dc, featuresbinf_rest), dim=1)
    @property
    def get_featuresbs(self):
        featuresbs_dc = self._featuresbs_dc
        featuresbs_rest = self._featuresbs_rest
        return torch.cat((featuresbs_dc, featuresbs_rest), dim=1)
    @property
    def get_featuresdirect(self):
        featuresdirect_dc = self._featuresdirect_dc
        featuresdirect_rest = self._featuresdirect_rest
        return torch.cat((featuresdirect_dc, featuresdirect_rest), dim=1)
    @property
    def get_minimum_axis(self):
        return get_minimum_axis(self.get_scaling, self.get_rotation)
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
        
        featuresuw = torch.zeros((1, 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # (1,3,16)

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
        if self.uw_flag.startswith("HYB"):
            # start addition uw
            zero_3 = torch.zeros((1, 3), device="cuda")
            #direct_ini = torch.Tensor([[0.007,0.008,0.008]]).cuda()
            self._direct = nn.Parameter(zero_3).contiguous().requires_grad_(True)  # (1,3)
            #self._direct = nn.Parameter(torch.zeros_like(fused_point_cloud,device="cuda").contiguous().requires_grad_(True))  # (P,3) #########################V2
            # binf_ini = torch.Tensor([0.0611105 , 0.09450259, 0.14404887]).cuda()
            #binf_ini = torch.Tensor([0.15113972, 0.23778467, 0.25      ]).cuda()
            self._binf = nn.Parameter(zero_3).contiguous().requires_grad_(True)  # (1,3)
            self._featuresbinf_dc = nn.Parameter(featuresuw[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # (1,1,3), first coefficient represents the DC
            self._featuresbinf_rest = nn.Parameter(featuresuw[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # (1,15,3), the rest of the coefficients
            self._featuresbs_dc = nn.Parameter(featuresuw[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # (1,1,3), first coefficient represents the DC
            self._featuresbs_rest = nn.Parameter(featuresuw[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # (1,15,3), the rest of the coefficients
            self._featuresdirect_dc = nn.Parameter(featuresuw[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # (1,1,3), first coefficient represents the DC
            self._featuresdirect_rest = nn.Parameter(featuresuw[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # (1,15,3), the rest of the coefficients
            #bs_ini = torch.Tensor([0.25      , 0.06863068, 0.03780112]).cuda()
            #bs_ini = torch.Tensor([0.25, 0.12123746, 0.15481124]).cuda()
            self._bs = nn.Parameter(zero_3.contiguous().requires_grad_(True)) # (1,3)
            # end addition uw

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if self.uw_flag.startswith("HYB"):
            l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # start addition uw
            {'params': [self._direct], 'lr': training_args.feature_lr, "name": "direct"}, # using the same learning rate 
            {'params': [self._binf], 'lr': training_args.feature_lr , "name": "binf"}, # using the same learning rate 
            
            {'params': [self._featuresbinf_dc], 'lr': training_args.feature_lr, "name": "fbinf_dc"},
            {'params': [self._featuresbinf_rest], 'lr': training_args.feature_lr / 20.0, "name": "fbinf_rest"},
            {'params': [self._featuresbs_dc], 'lr': training_args.feature_lr, "name": "fbs_dc"},
            {'params': [self._featuresbs_rest], 'lr': training_args.feature_lr / 20.0, "name": "fbs_rest"},
            {'params': [self._featuresdirect_dc], 'lr': training_args.feature_lr, "name": "fdirect_dc"},
            {'params': [self._featuresdirect_rest], 'lr': training_args.feature_lr / 20.0, "name": "fdirect_rest"},
            
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
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, viewer_fmt=False):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        if self.uw_flag.startswith("HYB"):
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
        if self.uw_flag.startswith("HYB"):

            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1) # with uw addition

            #attributes = np.concatenate((xyz, normals, f_dc, f_rest, direct, np.broadcast_to(binf, (xyz.shape[0],) + binf.shape[1:]), bs, opacities, scale, rotation), axis=1) # with uw addition
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
    def enforce_constraints_hyb(self, direct_range=(0.1,1), binf_range=(0.1,0.8), bs_range=(0.1,1), uw_additions=None):
        self._binf.data.clamp_(binf_range[0], binf_range[1])
        self._bs.data.clamp_(bs_range[0], bs_range[1]) 
        # self._direct.data.clamp_(direct_range[0], direct_range[1]) 
        # clamped = {}
        # clamped['binf'] = torch.clamp(uw_additions['binf'], min=binf_range[0], max=binf_range[1])
        # clamped['bs'] = torch.clamp(uw_additions['bs'], min=bs_range[0], max=bs_range[1])
        # clamped['direct'] = torch.clamp(uw_additions['direct'], min=direct_range[0], max=direct_range[1])
        # if torch.any(clamped['binf'] != uw_additions['binf']):
        #     adjustment_factor = torch.min(clamped['binf'] / ( uw_additions['binf'] + 1e-1))
        #     self._featuresbinf_dc.data = self._featuresbinf_dc.data * adjustment_factor
        #     self._featuresbinf_rest.data = self._featuresbinf_rest.data * adjustment_factor
        # if torch.any(clamped['bs'] != uw_additions['bs']):
        #     adjustment_factor = torch.min(clamped['bs'] / ( uw_additions['bs'] + 1e-1))
        #     self._featuresbs_dc.data = self._featuresbs_dc.data * adjustment_factor
        #     self._featuresbs_rest.data = self._featuresbs_rest.data * adjustment_factor
        # if torch.any(clamped['direct'] != uw_additions['direct']):
        #     adjustment_factor = torch.min(clamped['direct'] / ( uw_additions['direct'] + 1e-1))
        #     self._featuresdirect_dc.data = self._featuresdirect_dc.data * adjustment_factor
        #     self._featuresdirect_rest.data = self._featuresdirect_rest.data * adjustment_factor
        
    
    def reset_hyb(self):
        featuresuw = torch.zeros((1, 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # (1,3,16)
        # fbinf_dc_new = featuresuw[:,:,0:1].transpose(1, 2).contiguous()
        # optimizable_tensors = self.replace_tensor_to_optimizer(fbinf_dc_new, "fbinf_dc")
        # self._featuresbinf_dc= optimizable_tensors["fbinf_dc"]
        
        # fbinf_rest_new = featuresuw[:,:,1:].transpose(1, 2).contiguous()
        # optimizable_tensors = self.replace_tensor_to_optimizer(fbinf_rest_new, "fbinf_rest")
        # self._featuresbinf_rest= optimizable_tensors["fbinf_rest"]
        
        # fbs_dc_new = featuresuw[:,:,0:1].transpose(1, 2).contiguous()
        # optimizable_tensors = self.replace_tensor_to_optimizer(fbs_dc_new, "fbs_dc")
        # self._featuresbs_dc= optimizable_tensors["fbs_dc"]
        
        # fbs_rest_new = featuresuw[:,:,1:].transpose(1, 2).contiguous()
        # optimizable_tensors = self.replace_tensor_to_optimizer(fbs_rest_new, "fbs_rest")
        # self._featuresbs_rest= optimizable_tensors["fbs_rest"]
        
        fdirect_dc_new = featuresuw[:,:,0:1].transpose(1, 2).contiguous()
        optimizable_tensors = self.replace_tensor_to_optimizer(fdirect_dc_new, "fdirect_dc")
        self._featuresdirect_dc= optimizable_tensors["fdirect_dc"]
        
        fdirect_rest_new = featuresuw[:,:,1:].transpose(1, 2).contiguous()
        optimizable_tensors = self.replace_tensor_to_optimizer(fdirect_rest_new, "fdirect_rest")
        self._featuresdirect_rest= optimizable_tensors["fdirect_rest"]
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
        if self.uw_flag.startswith("HYB"):
            # start addition uw
            direct = np.asarray(plydata.elements[0]["direct"])[..., np.newaxis]
            binf = np.asarray(plydata.elements[0]["binf"])[..., np.newaxis]
            bs = np.asarray(plydata.elements[0]["bs"])[..., np.newaxis]
            self._direct = nn.Parameter(torch.tensor(direct, dtype=torch.float, device="cuda").requires_grad_(True))
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
            if group["name"] == "binf" or group["name"] == "direct" or group["name"] == "bs" \
                or group["name"] == "fbinf_dc" or group["name"] == "fbinf_rest" \
                    or group["name"] == "fbs_dc" or group["name"] == "fbs_rest" \
                        or group["name"] == "fdirect_dc" or group["name"] == "fdirect_rest":
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
        #self._direct = optimizable_tensors["direct"] #########################V2

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    def prune_nan(self):
        selected_pts_mask = torch.isnan(self.get_opacity).any(dim=1)
        self.prune_points(selected_pts_mask)
        
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1

            if group["name"] == "binf" or group["name"] == "direct" or group["name"] == "bs" \
                or group["name"] == "fbinf_dc" or group["name"] == "fbinf_rest" \
                    or group["name"] == "fbs_dc" or group["name"] == "fbs_rest" \
                        or group["name"] == "fdirect_dc" or group["name"] == "fdirect_rest":
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
                              new_opacities, new_scaling, new_rotation, reset_params=True):
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

        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    
    # start addition uw
    def densification_postfix_uwhyb(self, new_xyz, 
                              new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation
                              ,reset_params=True):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        # start addition uw
        #"direct": new_direct,
        #"binf": new_binf,
        #"bs": new_bs,
        # end addition uw
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        # start addition uw
        #self._direct = optimizable_tensors["direct"]
        #self._binf = optimizable_tensors["binf"]
        #self._bs = optimizable_tensors["bs"]
        # end addition uw
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2): #################V1
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

        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        
        if self.uw_flag.startswith("HYB"):
            self.densification_postfix_uwhyb(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=True) # with uw addition
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
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        if self.uw_flag.startswith("HYB"):
            self.densification_postfix_uwhyb(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, reset_params=True) # with uw addition
        else:
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation) # without uw addition
        

    def densify_and_prune(self, max_grad, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        torch.cuda.empty_cache()
    
    def densify_and_prune_by_size(self, min_opacity, max_screen_size=False):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > np.percentile(self.max_radii2D.detach().cpu().numpy(), 99.5)
            #big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            big_points_ws = self.get_scaling.max(dim=1).values > np.percentile(self.get_scaling.max(dim=1).values.detach().cpu().numpy(), 99.9)
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def set_requires_grad(self, attrib_name, state: bool):
        getattr(self, f"_{attrib_name}").requires_grad = state
    
    def prune_by_maskingBG(self, viewpoint_stack):
        prune_mask = process_mean3d_using_camera_mask(self.get_xyz, viewpoint_stack)
        self.prune_points(prune_mask)
        
###################################################### MCMC #####################################################################

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {"xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation}

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] == "binf" or group["name"] == "direct" or group["name"] == "bs"\
                 or group["name"] == "fbinf_dc" or group["name"] == "fbinf_rest": # binf is a general parameter
                continue
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"] 

        return optimizable_tensors

    
    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0] + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))

        return self._xyz[idxs], self._features_dc[idxs], self._features_rest[idxs], new_opacity, new_scaling, self._rotation[idxs]


    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio
    

    def relocate_gs(self, dead_mask=None):

        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask 
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = (self.get_opacity[alive_indices, 0]) 
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self._xyz[dead_indices], 
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices] 
        ) = self._update_params(reinit_idx, ratio=ratio)
        
        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx) 
        

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1) 
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz, 
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation 
        ) = self._update_params(add_idx, ratio=ratio)

        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling

        if self.uw_flag.startswith("HYB"):
            self.densification_postfix_uwhyb(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False) # with uw addition
        else:
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False)
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs
