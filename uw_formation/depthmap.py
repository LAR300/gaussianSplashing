import cv2
import torch
import os

def depth_estimation(image):
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
    return output
if __name__ == "__main__":
    whiteblancing_dir = os.path.join('uw_formation', 'white_balancing')
    depthdir = os.path.join('uw_formation', 'depth_estimation')
    input_image_name = "MTN_1103"
    input_image_name_ext = f"{input_image_name}.png"
    output_image_name = f"{input_image_name}_depth.png"
    image = cv2.imread(os.path.join(whiteblancing_dir,input_image_name_ext), cv2.IMREAD_COLOR)
    d_map = depth_estimation(image)
    cv2.imwrite(os.path.join(depthdir,output_image_name), d_map)