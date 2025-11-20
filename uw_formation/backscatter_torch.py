import torch
import cv2
from scipy.optimize import minimize
import numpy as np
import os
from depthmap import depth_estimation
from torchvision.transforms import Resize


def parameter_fun_Bc_diff2(x, z, Ic):
    Bc_inf = x[0]
    bcb = x[1]
    y = torch.norm(Ic - (Bc_inf * (1 - torch.exp(-bcb * z))))
    return y

def estimateBackscatter(Ism, zsm, pdark=0.1, intervals_num=20, resized_highet=None):
    if resized_highet is not None:
         # HWC
        resized_size_image = (resized_highet, int(resized_highet / Ism.shape[1] * Ism.shape[0]))
        resized_size_depth = (resized_highet, int(resized_highet / Ism.shape[1] * Ism.shape[0]))
        Ism = chw2hwc(Ism, invers=True)
        Ism = Resize(resized_size_image, antialias=True)(Ism)
        Ism = chw2hwc(Ism)
        zsm = Resize(resized_size_depth, antialias=True)(torch.unsqueeze(zsm, 0)).squeeze(0)
    Ism[Ism < 0] = 0
    zsm[zsm < 0] = 0
    darkZ, Bc = getBackscatterByCurveFittingMultipleImages(Ism, zsm, pdark)
    ints = torch.linspace(torch.min(darkZ), torch.max(darkZ), intervals_num) # intervals of depth values
    minvals = torch.full((len(ints) - 1, 3), float('nan')) # minvalues for each interval
    for k in range(3):
        for i in range(len(ints) - 1):
            ind = (darkZ > ints[i]) & (darkZ <= ints[i + 1])
            if torch.sum(ind) == 0:
                continue
            else:
                minvals[i, k] = torch.min(Bc[ind, k])
    
    thisz = ints[:-1] # all the bins
    exclude = torch.isnan(minvals[:, 0]) | torch.isnan(minvals[:, 1]) |  torch.isnan(minvals[:, 2])
    thisz = thisz[~exclude] # excluding the nan or inf values
    minvals = minvals[~exclude, :]
    thisz = torch.cat((torch.tensor([0]), thisz))
    minvals = torch.cat((torch.tensor([[0, 0, 0]]), minvals))
    
    x0 = torch.tensor([torch.rand(1).item(), torch.rand(1).item() * 5])
    lb = torch.tensor([0, 1])
    ub = torch.tensor([0, 5])
    out = torch.zeros((2, 3))
    
    options = {
    'disp': False,
    'maxiter': 1000,
    'maxfun': 100000,
    'ftol': 1e-6
    }
    x0 = torch.tensor([torch.rand(1).item(), torch.rand(1).item() * 5])  
    for k in range(3):
        f = lambda x: parameter_fun_Bc_diff2(x, thisz, minvals[:, k])
        problem = {'fun': f, 'x0': x0, 'bounds': [lb.numpy(), ub.numpy()], 'options': options}
        result = minimize(**problem)
        out[:, k] = torch.tensor(result.x)

    Bc_inf = out[0, :]
    bcb = out[1, :]
    
    # Fix color channel ordering to match RGB convention (like numpy version)
    # Swap R and B channels: BGR -> RGB
    Bc_inf_rgb = torch.stack([Bc_inf[2], Bc_inf[1], Bc_inf[0]], dim=0)
    bcb_rgb = torch.stack([bcb[2], bcb[1], bcb[0]], dim=0)
    
    # Validate estimated parameters
    print(f"Torch Estimated B_infinity (RGB): [{Bc_inf_rgb[0]:.3f}, {Bc_inf_rgb[1]:.3f}, {Bc_inf_rgb[2]:.3f}]")
    print(f"Torch Estimated betac_b (RGB): [{bcb_rgb[0]:.3f}, {bcb_rgb[1]:.3f}, {bcb_rgb[2]:.3f}]")
    
    params = {'Bc_inf': Bc_inf_rgb, 'betac_b': bcb_rgb, 'pdark': pdark}
    return params # RGB (fixed from BGR)

def getBackscatterByCurveFittingMultipleImages(I, dm, pdark=1, rhoflag=0, edges_num=10):
    if rhoflag:
        sigmaVal = 1 * (max(I.shape) / 2)
        Ec = 2 * cv2.GaussianBlur(I.numpy(), (0, 0), sigmaVal)
        rho = I / Ec
        rho = torch.clamp(rho, 0, 1)
    else:
        rho = I

    edges = torch.linspace(torch.min(dm), torch.max(dm), edges_num)
    zcluster = clusterRange(dm, edges)
    maskRho = torch.zeros(dm.shape)
    for i in range(len(edges) - 1):
        thisMask = (zcluster == i)
        if torch.sum(thisMask) > 0:
            thisRho = cropImgToDepthMap(rho, thisMask)
            thisMaskRho = findDarkestPixels(thisRho, pdark)
            maskRho += thisMaskRho

    maskRho = maskRho.bool()
    darkZ, Bc = extractDarkestPixels(I, maskRho, dm)
    exclude2 = torch.isnan(darkZ) | torch.isinf(darkZ)
    darkZ = darkZ[~exclude2]
    Bc = Bc[~exclude2.flatten(), :]
    exclude = torch.any(Bc < 0, dim=1)
    darkZ = darkZ[~exclude]
    Bc = Bc[~exclude, :]
    return darkZ, Bc

def clusterRange(dm, edges):
    bins = len(edges) - 1
    cluster = torch.zeros(dm.shape)
    for i in range(bins):
        cluster[(dm >= edges[i]) & (dm < edges[i+1])] = i
    return cluster

def extractDarkestPixels(I, mask, dm):
    s = I.shape
    if len(s) == 3:
        Ires = I.reshape(s[0] * s[1], 3)
        maskres = mask.reshape(s[0] * s[1], 1)
        dmres = dm.reshape(s[0] * s[1], 1)
        darkPix = Ires[maskres[:, 0]]
        darkZ = dmres[maskres[:, 0]]
    else:
        Ires = I.reshape(s[0] * s[1], 1)
        maskres = mask.reshape(s[0] * s[1], 1)
        dmres = dm.reshape(s[0] * s[1], 1)
        darkPix = Ires[maskres[:, 0]]
        darkZ = dmres[maskres[:, 0]]
    return darkZ, darkPix

def cropImgToDepthMap(rho, mask, mask_value=-1):
    # Apply the mask to the image
    cropped_image = torch.where(mask.unsqueeze(-1), rho, mask_value)
    return cropped_image

def findDarkestPixels(thisRho, pdark, too_dark_flag=False):
    # Initialize an empty mask for each channel
    masks = []
    # Iterate over each channel
    for channel in range(3):
        # Sort the pixels in the channel in ascending order
        sorted_pixels = torch.sort(thisRho[:,:,channel].flatten())[0]
        sorted_pixels = sorted_pixels[sorted_pixels >= 0]
        # Compute the threshold value for the lower pdark percentile
        threshold_value = 0
        pdark_c = pdark
        if too_dark_flag:
            pdark_c = pdark - 0.005
            while threshold_value == 0:
                pdark_c = pdark_c + 0.005
                threshold_index = int(len(sorted_pixels) * pdark_c) if pdark_c < 1 else len(sorted_pixels) - 1
                threshold_value = sorted_pixels[threshold_index]
        else:
            threshold_index = int(len(sorted_pixels) * pdark_c) if pdark_c < 1 else len(sorted_pixels) - 1
            threshold_value = sorted_pixels[threshold_index]

        # Create a mask where pixels below the threshold value and pixels greater than or equal to zero are True for this channel
        mask_channel = (thisRho[:,:,channel] <= threshold_value) & (thisRho[:,:,channel] >= 0)
        masks.append(mask_channel)
    
    # Stack masks for each channel along the last axis to form the final mask
    stacked_masks = torch.stack(masks, dim=-1)
    mask = torch.all(stacked_masks, dim=-1)

    return mask

def create_backscatter_image(z, bs_dict):
    Bc_inf = bs_dict['Bc_inf']
    bcb = bs_dict['betac_b']
    z_3 = torch.stack([z,z,z], dim=-1)
    y = Bc_inf * (1 - torch.exp(-bcb * z_3))
    return y

def clean_bs_image(I, bs_image):
    return I - bs_image

def load_image(filename, code=cv2.IMREAD_COLOR, depth=False):
    if depth:
        return torch.from_numpy(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
    else:
        image = cv2.imread(filename, code)
        if code == cv2.IMREAD_COLOR or code == cv2.IMREAD_UNCHANGED:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR->RGB (match numpy version)
        if np.max(image) > 1:
            image = image / 255.0
        return torch.from_numpy(image)
    
def save_image(filename, image):
    if torch.max(image) <= 1:
        image = image * 255.0
    image = chw2hwc(image)
    # Convert RGB to BGR for cv2.imwrite (match numpy version)
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)

def chw2hwc(image, invers=False):
    if invers:
        # HWC -> CHW
        image.shape[2] in [1, 2, 3]
        return image.permute(2, 0, 1)
    if image.shape[0] in [1, 2, 3]:
        # CHW -> HWC
        return image.permute(1, 2, 0)
    return image

def preform_full_bs_removal(image, image_depth=None):
    image = chw2hwc(image)
    if image_depth is None or image_depth.shape != image.shape[:2]:
        image_depth = depth_estimation(image)  # using midas depth estimation
    bs_dict = estimateBackscatter(image, image_depth, pdark=0.1)
    bs_image = create_backscatter_image(image_depth, bs_dict)
    clean_from_bs = clean_bs_image(image, bs_image)
    bs_image = chw2hwc(bs_image, invers=True)
    clean_from_bs = chw2hwc(clean_from_bs, invers=True)
    return clean_from_bs, bs_image, bs_dict

def preform_full_bs_removal_path(input_image_path, image_depth=None):
    image = load_image(input_image_path)
    image = chw2hwc(image)
    if image_depth is None or image_depth.shape != image.shape[:2]:
        print('Estimating depth')
        image_depth = depth_estimation(image)
        image_depth = torch.from_numpy(image_depth)
    bs_dict = estimateBackscatter(image, image_depth, pdark=0.1)
    bs_image = create_backscatter_image(image_depth, bs_dict)
    clean_from_bs = clean_bs_image(image, bs_image)
    bs_image = chw2hwc(bs_image, invers=True)
    clean_from_bs = chw2hwc(clean_from_bs, invers=True)
    
    return image, clean_from_bs, bs_image, bs_dict, image_depth

def preform_bs_on_dir(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            input_image = os.path.join(input_dir, file)
            output_image = os.path.join(output_dir, file)
            preform_full_bs_removal_path(input_image, output_image)
            print(f'Processing {file} done')



if __name__ == "__main__":
    whiteblancing_dir = os.path.join('uw_formation', 'white_balancing')
    bs_dir = os.path.join('uw_formation', 'bs_estimation')
    depth_dir = os.path.join('uw_formation', 'depth_estimation')
    clean_dir = os.path.join('uw_formation', 'clean_bs')
    depth_gs_dir = os.path.join('uw_formation', 'depth_gs')
    input_image_name = "MTN_1108"
    ext = ".png"
    input_depth = None
    input_image = f"{input_image_name}{ext}"
    #input_depth = os.path.join(depth_dir,f"{input_image_name}_depth.png")
    #input_depth = os.path.join(depth_gs_dir,f"train_{input_image_name}_30000.png")
    output_image_name_depth = f"{input_image_name}_depth.png"
    output_image_name_cleanbs = f"{input_image_name}_nonbs.png"
    output_image_name_bs = f"{input_image_name}_bs.png"
    image, clean_from_bs, bs_image, bs_dict, image_depth = preform_full_bs_removal_path(os.path.join(whiteblancing_dir,input_image), image_depth=None)
    save_image(os.path.join(clean_dir,output_image_name_cleanbs), clean_from_bs)
    save_image(os.path.join(depth_dir,output_image_name_depth), image_depth)
    save_image(os.path.join(bs_dir,output_image_name_bs), bs_image)
    print(bs_dict)
    print(f'Processing {input_image_name} done')
    #preform_full_bs_removal_path(input_image, os.path.join(bs_dir,output_image_name_cleanbs), image_depth=load_image(input_depth, depth=True))

    
    # seathru_nerf_data_dir = os.path.join('data', 'SeathruNeRF_dataset')
    # preform_bs_on_dir(os.path.join(seathru_nerf_data_dir, 'Panama', 'images_wb'), os.path.join(seathru_nerf_data_dir, 'Panama', 'Panama_images_without_bs'))
    
