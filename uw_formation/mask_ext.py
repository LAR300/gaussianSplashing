import cv2
import numpy as np
import os

def create_smooth_mask(image, threshold=3, kernel_size=9):
    """Create a smooth mask based on high variations in the image."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the gradient (variation) of the image
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate the magnitude of the gradient
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Normalize the magnitude to the range [0, 255]
    normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply a binary threshold to identify high variation areas
    _, mask = cv2.threshold(normalized_magnitude, threshold, 255, cv2.THRESH_BINARY)
    
    # Convert to uint8
    mask = np.uint8(mask)
    
    # Apply morphological closing to fill small holes in the mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Optionally apply Gaussian blur to smooth the edges of the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask

def process_images(input_folder, output_folder, threshold=30):
    """Process all images in the folder to create smooth variation masks."""
    # Get all image file paths
    image_paths = [os.path.join(input_folder, fname) for fname in os.listdir(input_folder) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    # Sort the paths to ensure they are in a specific order
    image_paths.sort()
    
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each image in the folder
    for image_path in image_paths:
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error reading image: {image_path}")
            continue
        
        # Create a smooth variation mask for the current image
        #mask = create_smooth_mask(image, threshold)
        mask = np.zeros_like(image)
        mask[:] = 255
        
        # Get the filename and prepare the mask output path
        filename = os.path.basename(image_path)
        mask_filename = f"{filename}"
        mask_path = os.path.join(output_folder, mask_filename)
        
        # Save the mask as an image
        cv2.imwrite(mask_path, mask)
        print(f"Saved smooth mask: {mask_path}")
     
if __name__ == "__main__":
    input_folder = "/home/nirmu/projects/gaussian_splattingUW/data/D3/images"  # Replace with your input folder path
    output_folder = "/home/nirmu/projects/gaussian_splattingUW/data/D3/masks"  # Replace with your output folder path
    
    process_images(input_folder, output_folder)
