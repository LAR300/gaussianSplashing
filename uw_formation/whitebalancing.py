import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
from PIL import Image
from skimage import exposure

def compensate_RB(image, flag):
    """
    Compensates for the red and blue channel lost in underwater images.

    Args:
        image: The underwater image as a cv2 image object.
        flag: A flag that determines how to compensate for the lost channels.
            0: Compensate for both red and blue channels using the green channel.
            1: Compensate for the red channel only using the green channel.

    Returns:
        A new Image object with the compensated colors.
    """

    # Split the image into R, G, and B components
    imageB, imageG, imageR = cv2.split(image)

    # Get the maximum and minimum pixel values for each channel
    minR, maxR = np.min(imageR), np.max(imageR)
    minG, maxG = np.min(imageG), np.max(imageG)
    minB, maxB = np.min(imageB), np.max(imageB)

    # Convert the image arrays to NumPy arrays
    imageR = imageR.astype(np.float64)
    imageG = imageG.astype(np.float64)
    imageB = imageB.astype(np.float64)

    # Normalize the pixel values to the range [0, 1]
    imageR = (imageR - minR) / (maxR - minR)
    imageG = (imageG - minG) / (maxG - minG)
    imageB = (imageB - minB) / (maxB - minB)

    # Get the mean of each channel
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)

    # Compensate for red and blue channels
    if flag == 0:
        imageR = (imageR + (meanG - meanR) * (1 - imageR) * imageG) * maxR
        imageB = (imageB + (meanG - meanB) * (1 - imageB) * imageG) * maxB
        imageG *= maxG
    elif flag == 1:
        imageR = (imageR + (meanG - meanR) * (1 - imageR) * imageG) * maxR
        imageB *= maxB
        imageG *= maxG

    # Convert the compensated image arrays back to uint8 type
    imageR = np.clip(imageR, 0, 255).astype(np.uint8)
    imageG = np.clip(imageG, 0, 255).astype(np.uint8)
    imageB = np.clip(imageB, 0, 255).astype(np.uint8)

    # Merge the channels back together
    compensateIM = cv2.merge([imageR, imageG, imageB])

    return compensateIM
def gray_world(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of input image files
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for image_file in image_files:
        # Open image
        img_path = os.path.join(input_dir, image_file)
        img = Image.open(img_path)

        # Convert image to numpy array
        img_array = np.array(img)

        # Apply histogram equalization to improve contrast
        img_array_eq = exposure.equalize_hist(img_array)

        scaled_r = img_array_eq[:,:,0]
        scaled_g = img_array_eq[:,:,1]
        scaled_b = img_array_eq[:,:,2]

        # Stack scaled channels into RGB image
        balanced_img_array = np.stack([scaled_r, scaled_g, scaled_b], axis=2)

        # Clip values to [0, 255] range
        balanced_img_array = np.clip(balanced_img_array, 0, 255)
        balanced_img_array = (balanced_img_array * 255).astype(np.uint8)
        # Convert numpy array back to image
        balanced_img = Image.fromarray(balanced_img_array.astype(np.uint8))

        # Save balanced image
        output_path = os.path.join(output_dir, image_file)
        balanced_img.save(output_path)

def white_balance_gray_avg(image):
    """
    Performs white balancing on an image, normalizing channels for gray average.

    Args:
        image: The input image as a NumPy array.

    Returns:
        The white-balanced image as a NumPy array.
    """

    print(f"Image shape: {image.shape}, image min: {image.min()}, image max: {image.max()}, image type: {image.dtype}")
    # Convert image to float format
    image_float = image.astype("float32") / 255.0
    percentiles = np.percentile(image_float, [0.5, 99.5], axis=(0, 1))
    clipped_image = np.clip(image_float, percentiles[0], percentiles[1]) * 255.0  # Convert back to uint8 after clipping
    image_float = image.astype("uint8")
    print(f"Image shape: {clipped_image.shape}, image min: {clipped_image.min()}, image max: {clipped_image.max()}, image type: {clipped_image.dtype}")
    # Calculate grayscale image and overall mean
    grayscale_image = cv2.cvtColor(image_float, cv2.COLOR_BGR2GRAY)
    mean_gray = np.mean(grayscale_image)

    # Calculate mean intensities for each channel
    mean_b, mean_g, mean_r = cv2.split(image_float)[0].mean(), cv2.split(image_float)[1].mean(), cv2.split(image_float)[2].mean()

    # Calculate individual scaling factors for each channel
    scale_b = mean_gray / mean_b
    scale_g = mean_gray / mean_g
    scale_r = mean_gray / mean_r

    # Apply scaling factors to each channel
    white_balanced_image = np.zeros_like(image_float)
    white_balanced_image[:, :, 0] = image_float[:, :, 0] * scale_b
    white_balanced_image[:, :, 1] = image_float[:, :, 1] * scale_g
    white_balanced_image[:, :, 2] = image_float[:, :, 2] * scale_r

    # Convert back to uint8 format
    white_balanced_image = (white_balanced_image * 255.0).astype("uint8")

    return white_balanced_image

def wb_precentile(imagesPath, outImagesPath, precentile=99.8):
    imageList = os.listdir(imagesPath)
    imageNum = len(imageList)
    scale = (1384, 918)
    imageVec = np.zeros((scale[1], scale[0], 3, imageNum))
    imageVecNum = imageVec.shape[-1]

    for imageIdx in range(imageVecNum):
        img = Image.open(os.path.join(imagesPath, imageList[imageIdx]))
        img = img.resize(scale)
        img = np.array(img) / 255.0  # Convert image to double
        imageVec[:, :, :, imageIdx] = img

    wb = np.percentile(imageVec, precentile, axis=(0, 1, 3))

    for imageIdx in range(imageVecNum):
        currImage = imageVec[:, :, :, imageIdx] / wb
        currImage = (currImage * 255).astype(np.uint8)  # Convert back to uint8
        img = Image.fromarray(currImage)
        img.save(os.path.join(outImagesPath, f"{imageList[imageIdx]}"))
        
def process_images(input_folder, output_folder, binary=False):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Open the image file
            img_path = os.path.join(input_folder, filename)
            with Image.open(img_path) as img:
                # Resize the image
                resized_img = img.resize((1365, 904))

                # Convert to binary (1-bit pixels, black and white)
                if binary:
                    resized_img = resized_img.convert('1')

                # Invert black and white
                #inverted_img = Image.eval(binary_img, lambda x: 255 - x)

                # Save the processed image to the output folder
                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)

if __name__ == "__main__":
    # whiteblancing_dir = os.path.dirname(__file__)
    # input_image_name = "MTN_1288"
    # input_image_name_ext = f'{input_image_name}.png'
    # output_image_name = f"{input_image_name}_wb.png"

    # image = cv2.imread(os.path.join(whiteblancing_dir,input_image_name_ext), cv2.IMREAD_COLOR)


    # balanced_image = compensate_RB(image, 0)
    # balanced_image = gray_world(balanced_image)


    #balanced_image = white_balance_gray_avg(balanced_image)
    
    imagesPath = os.path.join("data", "D3", "imagesRaw")
    outImagesPath = os.path.join("data", "D3", "images")
    
    # imagesPath = os.path.join("data", "SeathruNeRF_dataset", "JapaneseGardens-RedSea-notWB" , "images")
    # outImagesPath = os.path.join("data", "SeathruNeRF_dataset", "JapaneseGardens-RedSea-notWB" , "images_wb")
    process_images(imagesPath, outImagesPath)