"""
For each image in the images/train directory, randomly crop 50 256x256 images.
Store the cropped images in directory images/train/cropped-images/
"""
import os
import random
from PIL import Image

def random_crop_images(source_dir, target_dir, crop_size=(256, 256), num_crops=50):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for filename in os.listdir(source_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more file types if needed
            source_path = os.path.join(source_dir, filename)
            img = Image.open(source_path)
            img_width, img_height = img.size
            
            crops = []
            for _ in range(num_crops):
                left = random.randint(0, img_width - crop_size[0])
                top = random.randint(0, img_height - crop_size[1])
                right = left + crop_size[0]
                bottom = top + crop_size[1]
                
                cropped_img = img.crop((left, top, right, bottom))
                crops.append(cropped_img)
            
            # Save cropped images
            for i, crop in enumerate(crops):
                crop_filename = f"{os.path.splitext(filename)[0]}_crop_{i}.png"
                crop_path = os.path.join(target_dir, crop_filename)
                crop.save(crop_path)

source_directory = "./images/train/"
target_directory = "./images/train/cropped-images/"
random_crop_images(source_directory, target_directory)





