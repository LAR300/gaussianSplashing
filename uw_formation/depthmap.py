import cv2
import torch
import os
import numpy as np

def depth_estimation(image):
    if type(image) == torch.Tensor:
        image = image.numpy()
    transpose_chw = False
    if image.shape[0] in [1,2,3]:
         # transpose CHW -> HWC
        transpose_chw = True
        image = np.transpose(image, (1, 2, 0))
        
    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo='check')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform(image).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    # output depth is opposite 
    output = output.max() - output + 0.5
    if transpose_chw:
        # transpose HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
    
    
    return output

if __name__ == "__main__":
    whiteblancing_dir = os.path.join('uw_formation', 'white_balancing')
    depthdir = os.path.join('uw_formation', 'depth_estimation')
    input_image_name = "MTN_1097"
    input_image_name_ext = f"{input_image_name}.png"
    output_image_name = f"{input_image_name}_depth.png"
    image = cv2.imread(os.path.join(whiteblancing_dir,input_image_name_ext), cv2.IMREAD_COLOR)
    d_map = depth_estimation(image)
    print(f'min: {d_map.min()}, max: {d_map.max()}.. max lines up {d_map[0:10].max()}, max lines down {d_map[-10:-1].max()}, shape: {d_map.shape}')
    cv2.imwrite(os.path.join(depthdir,output_image_name), d_map)