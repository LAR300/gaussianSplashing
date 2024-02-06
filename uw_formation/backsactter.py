import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from PIL import Image

def parameter_fun_Bc_diff2(x, z, Ic):
    Bc_inf = x[0]
    bcb = x[1]
    y = np.linalg.norm(Ic - (Bc_inf * (1 - np.exp(-bcb * z))))
    return y

# // params = [];
# // params = estimateBackscatter(RGB,z,params,10);

def estimateBackscatter(I, z, pdark=10):
    Ism = np.array(Image.open(I).resize((300, np.nan)))
    Ism[Ism < 0] = 0
    zsm = np.array(Image.open(z).resize((300, np.nan)))
    zsm[zsm < 0] = np.nan
    darkZ, Bc, _, _ = getBackscatterByCurveFittingMultipleImages(Ism, zsm, pdark)
    ints = np.linspace(np.min(darkZ), np.max(darkZ), 20)
    minvals = np.full((len(ints) - 1, 3), np.nan)
    for k in range(3):
        for i in range(len(ints) - 1):
            ind = (darkZ > ints[i]) & (darkZ <= ints[i + 1])
            this_val = np.min(Bc[ind, k])
            if ~np.isnan(this_val):
                minvals[i, k] = this_val
    thisz = ints[:-1]
    exclude = np.isnan(minvals[:, 0])
    thisz = thisz[~exclude]
    minvals = minvals[~exclude, :]
    thisz = np.concatenate([[0], thisz])
    minvals = np.vstack([[0, 0, 0], minvals])

    x0 = np.random.randn(2)
    lb = [0, 0]
    ub = [1, 1]
    out = np.zeros((2, 3))
    for k in range(3):
        thisVal = thisz
        f = lambda x: parameter_fun_Bc_diff2(x, thisz, minvals[:, k])
        problem = {'fun': f, 'x0': x0, 'bounds': [lb, ub]}
        result = minimize(**problem)
        out[:, k] = result.x

    Bc_inf = out[0]
    bcb = out[1]
    params = {'Bc_inf': Bc_inf, 'betac_b': bcb, 'pdark': pdark}
    return params

def getBackscatterByCurveFittingMultipleImages(I, dm, pdark=1, rhoflag=0):
    if rhoflag:
        sigmaVal = 1 * (max(I.shape) / 2)
        Ec = 2 * gaussian_filter(I, sigma=sigmaVal)
        rho = I / Ec
        rho = np.clip(rho, 0, 1)
    else:
        rho = I

    edges = np.linspace(np.min(dm), np.max(dm), 10)
    zcluster = clusterRange(dm, edges)
    maskRho = np.zeros(I.shape[:2])
    for i in range(len(edges) - 1):
        thisMask = (zcluster >= edges[i]) & (zcluster < edges[i + 1])
        if np.sum(thisMask) > 0:
            thisRho = cropImgToDepthMap(rho, ~thisMask)
            thisMaskRho = findDarkestPixels(thisRho, pdark)
            numFound = np.sum(thisMaskRho)
            maskRho += thisMaskRho
    maskRho = maskRho.astype(bool)
    darkZ, Bc = extractDarkestPixels(I, maskRho, dm)
    exclude2 = np.isnan(darkZ) | np.isinf(darkZ)
    darkZ = darkZ[~exclude2]
    Bc = Bc[~exclude2, :]
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

def cropImgToDepthMap(rho, mask):
    # Invert the mask
    inverted_mask = ~mask
    
    # Apply the inverted mask to the image
    cropped_image = np.where(inverted_mask, rho, 0)
    
    return cropped_image

def findDarkestPixels(thisRho, pdark):
    # Flatten the image to a 1D array
    flattened_image = thisRho.flatten()
    
    # Sort the pixels in ascending order
    sorted_pixels = np.sort(flattened_image)
    
    # Calculate the threshold index based on the percentile pdark
    threshold_index = int(len(sorted_pixels) * pdark)
    
    # Get the threshold value
    threshold_value = sorted_pixels[threshold_index]
    
    # Create a mask where pixels below the threshold are True
    mask = thisRho <= threshold_value
    
    return mask