import cv2
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.ndimage import gaussian_filter
import os
try:
    from uw_formation.depthmap import depth_estimation
except ImportError:
    from depthmap import depth_estimation
from scipy.optimize import curve_fit

def parameter_fun_Bc_diff2(x, z, Ic):
    z = np.array(z)
    Ic = np.array(Ic)
    Bc_inf = x[0]
    bcb = x[1]
    #y = np.linalg.norm(Ic - (Bc_inf * (1 - np.exp(-bcb * z))))
    y = np.sum((Ic - Bc_inf * (1 - np.exp(-bcb * z)))**2)
    return y



def estimateBackscatter(Ism, zsm, pdark=0.01, intervals_num=25, resized_highet=300):
    resized_highet = Ism.shape[1]
    Ism = cv2.resize(Ism, (resized_highet, int(resized_highet / Ism.shape[1] * Ism.shape[0])))
    Ism[Ism < 0] = 0
    zsm = cv2.resize(zsm, (resized_highet, int(resized_highet / zsm.shape[1] * zsm.shape[0])))
    zsm[zsm < 0] = 0
    darkZ, Bc = getBackscatterByCurveFittingMultipleImages(Ism, zsm, pdark)
    ints = np.linspace(np.min(darkZ), np.max(darkZ), intervals_num) # intervals of depth values
    min_values_depth = [[],[],[]]
    minvals = [[],[],[]] # minvalues for each interval
    for k in range(3):
        for i in range(len(ints) - 1):
            ind = (darkZ > ints[i]) & (darkZ <= ints[i + 1])
            if np.sum(ind) == 0 or np.sum(np.isnan(Bc[ind, k])) == len(Bc[ind, k]):
                continue
            else:
                minvals[k].append(np.min(Bc[ind, k]))
                #minvals[k].append(ints[i]))
                index_min = np.argmin(Bc[ind, k], axis=0)
                min_values_depth[k].append(darkZ[ind][index_min])
    
    #thisz = ints[:-1] # all the bins
    #exclude = np.isnan(minvals[:, 0]) | np.isnan(minvals[:, 1]) |  np.isnan(minvals[:, 2])
    #thisz = thisz[~exclude] # excluding the nan or inf values
    #minvals = minvals[~exclude, :]
    #thisz = np.concatenate([[0], thisz])
    #minvals = np.vstack([[0, 0, 0], minvals])
    scale = 0.25
    # x0 = [np.random.rand()*scale, np.random.rand()*scale]
    # lb = [0, 0]
    # ub = [scale, scale]
    b1_bounds = (0, scale)  # Example bounds for b1
    b2_bounds = (0, scale)   # Example bounds for b2
    out = np.zeros((2, 3))
    def model_func(z, b_inf, bcb):
        return b_inf * (1 - np.exp(-bcb * z))
    
    for k in range(3):
        f = lambda x: parameter_fun_Bc_diff2(x, min_values_depth[k][3:-3], minvals[k][3:-3])
        #problem = {'fun': f, 'x0': x0, 'bounds': (lb, ub)}
        #result = minimize(**problem)
        #result = least_squares(f, x0, bounds=[lb, ub])
        params, covariance = curve_fit(model_func, xdata=min_values_depth[k], ydata=minvals[k], bounds=([b1_bounds[0], b2_bounds[0]], [b1_bounds[1], b2_bounds[1]]))
        #out[:, k] = result.x
        out[:, k] = params
    Bc_inf = out[0, :]
    bcb = out[1, :]
    params = {'Bc_inf': np.array(Bc_inf), 'betac_b': np.array(bcb), 'pdark': pdark}
    return params # RGB

def getBackscatterByCurveFittingMultipleImages(I, dm, pdark=0.01, rhoflag=0, edges_num=10):
    if rhoflag:
        sigmaVal = 1 * (max(I.shape) / 2)
        Ec = 2 * gaussian_filter(I, sigma=sigmaVal)
        rho = I / Ec
        rho = np.clip(rho, 0, 1)
    else:
        rho = I

    edges = np.linspace(np.min(dm), np.max(dm), edges_num)
    zcluster = clusterRange(dm, edges)
    maskRho = np.zeros(dm.shape)
    for i in range(len(edges) - 1):
        thisMask = (zcluster == i)
        if np.sum(thisMask) > 0:
            thisRho = cropImgToDepthMap(rho, thisMask)
            thisMaskRho = findDarkestPixels(thisRho, pdark)
            numFound = np.sum(thisMaskRho)
            maskRho += thisMaskRho

    maskRho = maskRho.astype(bool)
    darkZ, Bc = extractDarkestPixels(I, maskRho, dm)
    exclude2 = np.isnan(darkZ) | np.isinf(darkZ)
    darkZ = darkZ[~exclude2]
    Bc = Bc[~exclude2.flatten(), :]
    exclude = np.any(Bc < 0, axis=1)
    darkZ = darkZ[~exclude]
    Bc = Bc[~exclude, :]
    return darkZ, Bc


def clusterRange(dm, edges):
    bins = len(edges) - 1
    cluster = np.zeros(dm.shape)
    for i in range(bins):
        cluster[(dm >= edges[i]) & (dm < edges[i+1])] = i
    return cluster


def extractDarkestPixels(I, mask, dm):
    s = I.shape
    if len(s) == 3:
        Ires = np.reshape(I, (s[0] * s[1], 3))
        maskres = np.reshape(mask, (s[0] * s[1], 1))
        dmres = np.reshape(dm, (s[0] * s[1], 1))
        darkPix = Ires[maskres[:, 0]]
        darkZ = dmres[maskres[:, 0]]
    else:
        Ires = np.reshape(I, (s[0] * s[1], 1))
        maskres = np.reshape(mask, (s[0] * s[1], 1))
        dmres = np.reshape(dm, (s[0] * s[1], 1))
        darkPix = Ires[maskres[:, 0]]
        darkZ = dmres[maskres[:, 0]]
    return darkZ, darkPix

def cropImgToDepthMap(rho, mask, mask_value=-1):
    # Apply the mask to the image
    cropped_image = np.where(np.expand_dims(mask,axis=-1), rho, mask_value)
    return cropped_image

def findDarkestPixels(thisRho, pdark, too_dark_flag=False):
    # Initialize an empty mask for each channel
    masks = []
    # Iterate over each channel
    for channel in range(3):
        # Sort the pixels in the channel in ascending order
        sorted_pixels = np.sort(thisRho[:,:,channel].flatten())
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
    stacked_masks = np.stack(masks, axis=-1)
    mask = np.all(stacked_masks, axis=-1)

    return mask

def create_backscatter_image(z, bs_dict):
    Bc_inf = bs_dict['Bc_inf']
    bcb = bs_dict['betac_b']
    y = [Bc_inf[i] * (1 - np.exp(-bcb[i] * z)) for i in range(Bc_inf.shape[0])]
    y = np.stack(y, axis=-1)
    return y

def clean_bs_image(I, bs_image):
    return I - bs_image

def add_bs_image(Dc, bs_image):
    return Dc + bs_image


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
    
    
def save_image(filename, image, depth=False):
    if depth:
        bits = 1
        grayscale = True
        if not grayscale:
            bits = 1

        if not np.isfinite(depth).all():
            depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            print("WARNING: Non-finite depth values present")

        depth_min = image.min()
        depth_max = image.max()

        max_val = (2**(8*bits))-1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (image - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(image.shape, dtype=image.dtype)

        if not grayscale:
            out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

        if bits == 1:
            cv2.imwrite(filename, out.astype("uint8"))
        elif bits == 2:
            cv2.imwrite(filename, out.astype("uint16"))
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

def preform_full_bs_addition(Dc, bs_dict, image_depth=None):
    transpose_chw = False
    if Dc.shape[0] in [1,2,3]:
        transpose_chw = True
        Dc = np.transpose(Dc, (1, 2, 0)) # transpose CHW -> HWC
    if image_depth is None or image_depth.shape != Dc.shape[:2]:
        image_depth = depth_estimation(Dc) # using midas depth estimation
    bs_image = create_backscatter_image(image_depth, bs_dict)
    Ic = add_bs_image(Dc, bs_image)
    if transpose_chw:
        # transpose HWC -> CHW
        Ic = np.transpose(Ic, (2, 0, 1))
    return Ic

def preform_full_bs_removal(image, bs_dict=None, image_depth=None, return_depth=False):
    transpose_chw = False
    if image.shape[0] in [1,2,3]:
         # transpose CHW -> HWC
        transpose_chw = True
        image = np.transpose(image, (1, 2, 0))
    if image_depth is None or image_depth.shape != image.shape[:2]:
        image_depth = depth_estimation(image) # using midas depth estimation
    if bs_dict is None:
        bs_dict = estimateBackscatter(image, image_depth, pdark=0.01)
    bs_image = create_backscatter_image(image_depth, bs_dict)
    clean_from_bs = clean_bs_image(image, bs_image)
    if transpose_chw:
        # transpose HWC -> CHW
        bs_image = np.transpose(bs_image, (2, 0, 1))
        clean_from_bs = np.transpose(clean_from_bs, (2, 0, 1))
    if return_depth:
        return clean_from_bs, bs_image, bs_dict, image_depth
    return clean_from_bs, bs_image, bs_dict

    
def preform_full_bs_removal_path(input_image_path, output_image_path, image_depth=None):
    image = load_image(input_image_path)
    if image_depth is None or image_depth.shape != image.shape[:2]:
        print('Estimating depth')
        image_depth = depth_estimation(image)
    bs_dict = estimateBackscatter(image, image_depth, pdark=0.01)
    bs_image = create_backscatter_image(image_depth, bs_dict)
    clean_from_bs = clean_bs_image(image, bs_image)
    save_image(output_image_path, clean_from_bs)
    


def preform_bs_on_dir(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            input_image = os.path.join(input_dir,file)
            output_image = os.path.join(output_dir,file)
            preform_full_bs_removal_path(input_image, output_image)
            print(f'Processing {file} done')
            
if __name__ == "__main__":
    whiteblancing_dir = os.path.join('uw_formation', 'white_balancing')
    bs_dir = os.path.join('uw_formation', 'bs_estimation')
    depth_dir = os.path.join('uw_formation', 'depth_estimation')
    clean_dir = os.path.join('uw_formation', 'clean_bs')
    depth_gs_dir = os.path.join('uw_formation', 'depth_gs')
    input_image_name = "MTN_1107"
    ext = ".png"
    # input_image_name = "DSC01784"
    # ext = ".jpg"
    input_depth = None
    input_image = os.path.join(whiteblancing_dir,f"{input_image_name}{ext}")
    #input_depth = os.path.join(depth_dir,f"{input_image_name}_depth.png")
    #input_depth = os.path.join(depth_gs_dir,f"train_{input_image_name}_30000.png")
    output_image_name_depth = f"{input_image_name}_depth.png"
    output_image_name_cleanbs = f"{input_image_name}_nonbs.png"
    output_image_name_bs = f"{input_image_name}_bs.png"
    #output_image_name_cleanbs = f"{input_image_name}_bs_clean.png"
    image = load_image(input_image)#load_image(input_image)
    image_depth = None
    image_depth = load_image(os.path.join(depth_gs_dir, f"{input_image_name}_gs10000_depth{ext}"),depth=True)
    clean_from_bs, bs_image, bs_dict, image_depth = preform_full_bs_removal(image, bs_dict=None, image_depth=image_depth, return_depth=True)

    save_image(os.path.join(clean_dir,output_image_name_cleanbs), clean_from_bs)
    save_image(os.path.join(depth_dir,output_image_name_depth), image_depth)
    save_image(os.path.join(bs_dir,output_image_name_bs), bs_image)

    print(f'bs_dict: {bs_dict}')
    print(f'Processing {input_image_name} done')
    #preform_full_bs_removal_path(input_image, os.path.join(bs_dir,output_image_name_cleanbs), image_depth=load_image(input_depth, depth=True))

    
    # seathru_nerf_data_dir = os.path.join('data', 'SeathruNeRF_dataset')
    # preform_bs_on_dir(os.path.join(seathru_nerf_data_dir, 'Panama', 'images_wb'), os.path.join(seathru_nerf_data_dir, 'Panama', 'Panama_images_without_bs'))