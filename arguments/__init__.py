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

from argparse import ArgumentParser, Namespace
import sys
import os
import json
import yaml

class NestedDict:
        def __init__(self, data):
            self._data = data

        def __getattr__(self, name):
            if name in self._data:
                if isinstance(self._data[name], dict):
                    return NestedDict(self._data[name])
                else:
                    return self._data[name]
            else:
                raise AttributeError(f"'NestedDict' object has no attribute '{name}'")
            
        def __setattr__(self, name, value):
            if name == '_data':
                super().__setattr__(name, value)
            else:
                self._data[name] = value
                
        def __getitem__(self, key):
            return self._data[key]

        def __setitem__(self, key, value):
            self._data[key] = value
            
_YAML_DEFAULTS = None

def _load_yaml_defaults():
    global _YAML_DEFAULTS
    if _YAML_DEFAULTS is not None:
        return
    config_path_env = os.environ.get("GS_CONFIG", None)
    candidate_paths = []
    if config_path_env:
        candidate_paths.append(config_path_env)
    # Try project-level default
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate_paths.append(os.path.join(project_root, "configs", "default.yaml"))
    # Try CWD fallback
    candidate_paths.append(os.path.join(os.getcwd(), "configs", "default.yaml"))
    for path in candidate_paths:
        try:
            if path and os.path.isfile(path):
                with open(path, 'r') as f:
                    _YAML_DEFAULTS = yaml.safe_load(f) or {}
                return
        except Exception:
            pass
    _YAML_DEFAULTS = {}

def _cfg(group_name, key, default_value):
    _load_yaml_defaults()
    try:
        return _YAML_DEFAULTS.get(group_name, {}).get(key, default_value)
    except Exception:
        return default_value

class DotDict(dict):
    """Dictionary class that allows dot notation access to its elements."""
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __setattr__(self, key, value):
        self[key] = value

def write_dictionary_to_file(namespace_dict, filename):
    #namespace_dict = vars(namespace)
    # Write the dictionary to the file using json
    with open(filename, 'w') as file:
        json.dump(dict(namespace_dict._data), file, indent=4)

def read_namespace_from_file(filename):
    # Read the dictionary from the file
    with open(filename, 'r') as file:
        namespace_dict = json.load(file)

    # Convert dictionary to namespace
    namespace = Namespace(**namespace_dict)

    return namespace
     
class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = _cfg('LoadingParameters', 'sh_degree', 3)
        self._source_path = _cfg('LoadingParameters', 'source_path', "")
        self._model_path = _cfg('LoadingParameters', 'model_path', "")
        self._images = _cfg('LoadingParameters', 'images', "images")
        self.initialization_points_option = _cfg('LoadingParameters', 'initialization_points_option', "pointcloud")
        self._resolution = _cfg('LoadingParameters', 'resolution', -1)
        self.underwater_processing = _cfg('LoadingParameters', 'underwater_processing', "OFF")
        self.removal_using_monocular_depth = _cfg('LoadingParameters', 'removal_using_monocular_depth', False)
        self._white_background = _cfg('LoadingParameters', 'white_background', False)
        self.data_device = _cfg('LoadingParameters', 'data_device', "cuda")
        self.densification = _cfg('LoadingParameters', 'densification', "OFF")
        self.cap_max = _cfg('LoadingParameters', 'cap_max', 200000)
        self.eval = _cfg('LoadingParameters', 'eval', True)
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = _cfg('PipelineParameters', 'convert_SHs_python', False)
        self.compute_cov3D_python = _cfg('PipelineParameters', 'compute_cov3D_python', False)
        self.debug = _cfg('PipelineParameters', 'debug', False)
        super().__init__(parser, "Pipeline Parameters")

class UnderwaterParams(ParamGroup):
    def __init__(self, parser):
        
        self.remove_bs_every = _cfg('UnderwaterParameters', 'remove_bs_every', 500)
        self.exp_range_dl = _cfg('UnderwaterParameters', 'exp_range_dl', 0.00001)
        self.exp_range_dh = _cfg('UnderwaterParameters', 'exp_range_dh', 0.00005)
        self.exp_range_bl = _cfg('UnderwaterParameters', 'exp_range_bl', 0.001)
        self.exp_range_bh = _cfg('UnderwaterParameters', 'exp_range_bh', 1)
        self.binf_range_l = _cfg('UnderwaterParameters', 'binf_range_l', 0.01)
        self.binf_range_h = _cfg('UnderwaterParameters', 'binf_range_h', 1)
        super().__init__(parser, "Underwater Parameters")
        
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = _cfg('OptimizationParameters', 'iterations', 30_000)
        self.position_lr_init = _cfg('OptimizationParameters', 'position_lr_init', 0.00016)
        self.position_lr_final = _cfg('OptimizationParameters', 'position_lr_final', 0.0000016)
        self.position_lr_delay_mult = _cfg('OptimizationParameters', 'position_lr_delay_mult', 0.01)
        self.position_lr_max_steps = _cfg('OptimizationParameters', 'position_lr_max_steps', 30_000)
        self.feature_lr = _cfg('OptimizationParameters', 'feature_lr', 0.0025)
        self.opacity_lr = _cfg('OptimizationParameters', 'opacity_lr', 0.05)
        self.scaling_lr = _cfg('OptimizationParameters', 'scaling_lr', 0.005)
        self.rotation_lr = _cfg('OptimizationParameters', 'rotation_lr', 0.001)
        self.percent_dense = _cfg('OptimizationParameters', 'percent_dense', 0.01)
        self.lambda_dssim = _cfg('OptimizationParameters', 'lambda_dssim', 0.3)
        self.lambda_bsest = _cfg('OptimizationParameters', 'lambda_bsest', 0.0)
        
        self.normal_lr = _cfg('OptimizationParameters', 'normal_lr', 0.0002)
        self.lambda_zero_one = _cfg('OptimizationParameters', 'lambda_zero_one', 1e-3)
        self.lambda_predicted_normal = _cfg('OptimizationParameters', 'lambda_predicted_normal', 2e-1)
        self.lambda_delta_reg = _cfg('OptimizationParameters', 'lambda_delta_reg', 1e-3)
        
        self.densification_interval = _cfg('OptimizationParameters', 'densification_interval', 100)
        self.opacity_reset_interval = _cfg('OptimizationParameters', 'opacity_reset_interval', 3000)
        self.masking_interval = _cfg('OptimizationParameters', 'masking_interval', -1)
        self.densify_from_iter = _cfg('OptimizationParameters', 'densify_from_iter', 1500)
        self.densify_until_iter = _cfg('OptimizationParameters', 'densify_until_iter', 15_000)
        self.densify_grad_threshold = _cfg('OptimizationParameters', 'densify_grad_threshold', 0.0002)
        self.remove_min_opacity_threshold  = _cfg('OptimizationParameters', 'remove_min_opacity_threshold', 0.1)
        self.random_background = _cfg('OptimizationParameters', 'random_background', False)
        self.noise_lr = _cfg('OptimizationParameters', 'noise_lr', 5e5)
        self.scale_reg = _cfg('OptimizationParameters', 'scale_reg', 0.02)
        self.opacity_reg = _cfg('OptimizationParameters', 'opacity_reg', 0.02)
        
        self.min_opacity_prune = _cfg('OptimizationParameters', 'min_opacity_prune', 0.005)
        self.large_gaussian_densification_screen_size_threshold = _cfg('OptimizationParameters', 'large_gaussian_densification_screen_size_threshold', 20.0)

        # New regularization parameters for conditional application
        self.lambda_opacity_reg = _cfg('OptimizationParameters', 'lambda_opacity_reg', 0.0)
        self.lambda_scale_reg = _cfg('OptimizationParameters', 'lambda_scale_reg', 0.0)
        
        # Near-camera large Gaussian removal parameters
        self.prune_near_large_gaussians = _cfg('OptimizationParameters', 'prune_near_large_gaussians', True)
        self.near_camera_distance_threshold = _cfg('OptimizationParameters', 'near_camera_distance_threshold', 2.0)
        self.large_gaussian_screen_size_threshold = _cfg('OptimizationParameters', 'large_gaussian_screen_size_threshold', 50.0)
        self.near_large_gaussian_opacity_threshold = _cfg('OptimizationParameters', 'near_large_gaussian_opacity_threshold', 0.8)
        
        super().__init__(parser, "Optimization Parameters")

class WandbParams(ParamGroup):
    def __init__(self, parser):
        self.wb_use_logger = _cfg('WandbParameters', 'wb_use_logger', False)
        self.wb_project = _cfg('WandbParameters', 'wb_project', "gaussian_splattingUW")
        self.wb_group = _cfg('WandbParameters', 'wb_group', "GroupExp1")
        self.wb_name = _cfg('WandbParameters', 'wb_name', "")
        self.wb_dir = _cfg('WandbParameters', 'wb_dir', os.path.join("./logs/"))
        self.wb_resume = _cfg('WandbParameters', 'wb_resume', None)
        self.wb_id = _cfg('WandbParameters', 'wb_id', None)
        super().__init__(parser, "Wandb Parameters")
        
def add_general_params(parser):
    # Add General Parameters
    group_general_args = parser.add_argument_group("General Parameters")
    group_general_args.add_argument('--ip', type=str, default=_cfg('GeneralParameters', 'ip', "0"))
    group_general_args.add_argument('--port', type=int, default=_cfg('GeneralParameters', 'port', -1))
    group_general_args.add_argument('--debug_from', type=int, default=_cfg('GeneralParameters', 'debug_from', -1))
    group_general_args.add_argument('--detect_anomaly', action='store_true', default=_cfg('GeneralParameters', 'detect_anomaly', False))
    group_general_args.add_argument("--test_iterations", nargs="+", type=int, default=_cfg('GeneralParameters', 'test_iterations', [1, 3000, 7000, 15_000, 20_000 , 30_000, 50_000]))
    group_general_args.add_argument("--save_iterations", nargs="+", type=int, default=_cfg('GeneralParameters', 'save_iterations', [75_000]))
    group_general_args.add_argument("--quiet", action="store_true", default=_cfg('GeneralParameters', 'quiet', False))
    group_general_args.add_argument("--checkpoint_iterations", nargs="+", type=int, default=_cfg('GeneralParameters', 'checkpoint_iterations', []))
    group_general_args.add_argument("--start_checkpoint", type=str, default=_cfg('GeneralParameters', 'start_checkpoint', None))
    group_general_args.add_argument('--save_for_metrics', action='store_true', default=_cfg('GeneralParameters', 'save_for_metrics', False))
    group_general_args.add_argument('--save_video', action='store_true', default=_cfg('GeneralParameters', 'save_video', False))
    return parser
    
def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
