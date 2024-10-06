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

import copy
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

def load_mask(filename, code=cv2.IMREAD_GRAYSCALE):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    
       
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

def save_render_depth(img_path,img, bit16=False):
    if bit16:
        #img = cv2.cvtColor(img, cv2.CV_16U)
        cv2.imwrite(img_path, (img*65535).astype(np.uint16))
    else:
        cv2.imwrite(img_path, cv2.cvtColor(img,cv2.COLOR_RGB2BGR)*255)
    
def save_imagePIL(img_path, image_orig):
    if type(image_orig) == torch.Tensor:
        image = image_orig.detach().cpu().numpy()
        try:
            image = image.squeeze(0)
        except:
            pass
    else:
        image = image_orig
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

def transform_points_4x4(points, matrix):
    """
    Transform a set of 3D points using a 4x4 transformation matrix.

    Parameters:
    points (np.ndarray): The 3D points as a numpy array with shape (m, 3).
    matrix (np.ndarray): The 4x4 transformation matrix as a numpy array.

    Returns:
    np.ndarray: The transformed 4D points as a numpy array with shape (m, 4).
    """
    num_points = points.shape[0]
    # Append a column of ones to the points to make them 4D
    points_hom = np.hstack((points, np.ones((num_points, 1))))
    # Perform the matrix multiplication
    transformed = points_hom @ matrix#.T , it is already transposed in cameras.py
    return transformed

def transform_points_4x4_torch(points, matrix):
    """
    Transform a set of 3D points using a 4x4 transformation matrix.

    Parameters:
    points (torch.Tensor): The 3D points as a PyTorch tensor with shape (m, 3).
    matrix (torch.Tensor): The 4x4 transformation matrix as a PyTorch tensor.

    Returns:
    torch.Tensor: The transformed 4D points as a PyTorch tensor with shape (m, 4).
    """
    # Ensure the points tensor has shape (m, 3)
    num_points = points.shape[0]
    
    # Append a column of ones to make the points 4D (homogeneous coordinates)
    ones = torch.ones((num_points, 1), device=points.device, dtype=points.dtype)
    points_hom = torch.cat((points, ones), dim=1)
    
    # Perform matrix multiplication to transform the points
    transformed = torch.matmul(points_hom, matrix)  # Using .T for correct matrix shape

    return transformed

def ndc2pix(v, S):
    """
    Convert a normalized device coordinate (NDC) value to pixel coordinates.

    Parameters:
    v (np.ndarray): The NDC values.
    S (int): The size of the screen (e.g., screen width or height in pixels).

    Returns:
    np.ndarray: The corresponding pixel coordinates.
    """
    return ((v + 1.0) * S - 1.0) * 0.5

def filter_array_by_mask(mean2d, mask, mean2d_mask):
    """
    Update mean2d_mask to 0 for rows in array `mean2d` where the corresponding values in the `mask` are 0.
    Leave rows unchanged if they refer to coordinates outside the bounds of the mask.

    Parameters:
    mean2d (np.ndarray): Input array of shape (m, 2).
    mask (np.ndarray): Mask array of shape (h, w).
    mean2d_mask (np.ndarray): Array of shape (m,) that will be updated.

    Returns:
    None
    """
    # Ensure the values in `mean2d` are within the bounds of `mask`
    indices_x = np.clip(mean2d[:, 0], 0, mask.shape[1] - 1).astype(int)
    indices_y = np.clip(mean2d[:, 1], 0, mask.shape[0] - 1).astype(int)

    # Create a boolean mask where the points are in the image bounds
    valid = (0 <= mean2d[:, 0]) & (mean2d[:, 0] < mask.shape[1]) & (0 <= mean2d[:, 1]) & (mean2d[:, 1] < mask.shape[0])

    # Update mean2d_mask where mask values are 0
    mean2d_mask[valid & (mask[indices_y, indices_x] == 0)] = 1
    #mean2d_mask[valid] = 1
    
def process_means3d_using_cam_for_depth_condition(mean3d, view_matrix):
    transformed_points = transform_points_4x4_torch(mean3d, view_matrix)[:,:3]
    return transformed_points

def process_mean3d_using_camera_mask(mean3d, viewpoint_stack):
    """
    Process all points to transform them using the projection matrix and convert them to image coordinates.

    Parameters:
    mean3d (np.ndarray): Array containing all points with shape (m, 3).
    projmatrix (np.ndarray) (inside viewpoint_stack): The 4x4 projection matrix.
    W (int): The width of the image.
    H (int): The height of the image.

    Returns:
    np.ndarray: The 2D points in image coordinates with shape (m, 2).
    """
    # `mean3d` is a 3D array containing points in a 3D space. The function
    # `process_mean3d_using_camera_mask` processes these points by transforming them using a
    # projection matrix and converting them to image coordinates. The points are transformed using the
    # projection matrix obtained from a camera viewpoint. The function calculates the reciprocal of
    # the homogeneous coordinate, projects the points, and then converts them to image coordinates by
    # applying a normalization operation. Finally, the function filters out points that fall outside
    # the bounds of a mask associated with the camera viewpoint.
    mean3d = mean3d.cpu().numpy()
    mean2d_mask = np.zeros((mean3d.shape[0], 1))
    for viewpoint_cam in viewpoint_stack:
        # if viewpoint_cam.image_name != "T_S04863":
        #     continue
        H = int(viewpoint_cam.image_height)
        W = int(viewpoint_cam.image_width)
        # Transform all points using the projection matrix
        transformed_points = transform_points_4x4(mean3d, viewpoint_cam.full_proj_transform.cpu().numpy())
        
        # Calculate the reciprocal of the homogeneous coordinate
        p_w = 1.0 / (transformed_points[:, 3] + 1e-7)
        
        # Project the points
        p_proj = transformed_points[:, :3] * p_w[:, np.newaxis]
        
        # Convert to image coordinates
        points_image_x = ndc2pix(p_proj[:, 0], W)
        points_image_y = ndc2pix(p_proj[:, 1], H)
        
        # Stack the coordinates to get the final output
        mean2d_cam = np.stack((points_image_x.astype(int), points_image_y.astype(int)), axis=-1)
        
        if viewpoint_cam.mask is not None:
            filter_array_by_mask(mean2d_cam, viewpoint_cam.mask.cpu().numpy().squeeze(0), mean2d_mask)
        #display_on_image(mean2d_cam[(mean2d_mask >= 0.5).astype(bool).squeeze(1)], W, H, viewpoint_cam.image_name)
    
    mean2d_mask = (mean2d_mask >= 0.5).astype(bool).squeeze(1)
    mean2d_mask = torch.tensor(mean2d_mask, dtype=torch.bool, device="cuda")
    return mean2d_mask

def display_on_image(mean2d_cam, W, H, image_name):
    """
    Display the 2D points on an image.

    Parameters:
    mean2d_cam (np.ndarray): The 2D points in image coordinates with shape (m, 2).
    W (int): The width of the image.
    H (int): The height of the image.

    Returns:
    None
    """
    # Create an empty image
    image = np.zeros((H, W, 3), dtype=np.uint8)
    # Draw a circle at each point
    for point in mean2d_cam:
        cv2.circle(image, tuple(point), 2, (0, 255, 0), -1)
    # Display the image
    cv2.imwrite('utils/projected_'+image_name + '.png', image)

    print(f"Image saved as {image_name}")
    
