import os
import cv2

def create_video_from_images(image_dir, output_video_path, fps=20):
    # Get list of all image files in the directory
    images = [img for img in os.listdir(image_dir) if img.endswith((".png", ".jpg", ".jpeg"))]
    
    # Sort images by their names
    images.sort()
    
    # Get the full path for each image
    image_paths = [os.path.join(image_dir, img) for img in images]
    
    # Read the first image to get the frame size
    frame = cv2.imread(image_paths[0])
    height, width, layers = frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4 files
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Iterate over sorted images and write them to the video
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        video.write(frame)
    
    # Release the video writer
    video.release()
    print(f"Video saved in {os.path.dirname(output_video_path)} as {os.path.basename(output_video_path)}")
    
if __name__ == "__main__":
    source_dir = 'output/GroupExDebug/GroupExDebug__D3_wbSM_HYB_iter30000_00000'
    # Example usage
    image_directory = f'{source_dir}/train/video_renders'
    output_video_file = f'{source_dir}/train/video_renders/video_renders.mp4'
    create_video_from_images(image_directory, output_video_file, fps=30)

    # image_directory = 'output/GroupExDebug/GroupExDebug__D3_HYB_iter15000_00000/video_rendered_J'
    # output_video_file = 'output/GroupExDebug/GroupExDebug__D3_HYB_iter15000_00000/video_renders/video_rendered_J.mp4'
    # create_video_from_images(image_directory, output_video_file, fps=30)

    image_directory = f'{source_dir}/train/video_depth'
    output_video_file = f'{source_dir}/train/video_depth/video_depth.mp4'
    create_video_from_images(image_directory, output_video_file, fps=30)