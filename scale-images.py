"""
Fetch images from directory images/train/ and scale them to 256x256.
Store the scaled images in directory images/train/scaled-images/
"""
import os
from PIL import Image

def crop_images(source_dir, target_dir, size=(256, 256)):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for filename in os.listdir(source_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"): # Add more file types if needed
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            
            with Image.open(source_path) as img:
                img = img.resize(size)
                img.save(target_path)

source_directory = "./images/train/"
target_directory = "./images/train/cropped-images/"
crop_images(source_directory, target_directory)
