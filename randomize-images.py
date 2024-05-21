"""
got to directory images/train/cropped-images/.
There are around 30,000 images there. Split these images into random chunks of 500 images each. Store each chunk in a separate directory, named images/train/cropped-images/chunk-0, images/train/cropped-images/chunk-1, and so on.
"""
import os
import random
import shutil

def split_images_into_chunks(source_dir, chunk_size=500):
    # Ensure the source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    # Get all image files in the directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    random.shuffle(files)  # Shuffle to randomize the order

    # Create chunks and corresponding directories
    for i in range(0, len(files), chunk_size):
        chunk_dir = os.path.join(source_dir, f"chunk-{i // chunk_size}")
        os.makedirs(chunk_dir, exist_ok=True)  # Create the chunk directory if it doesn't exist

        # Move files to the new chunk directory
        for file in files[i:i + chunk_size]:
            shutil.move(os.path.join(source_dir, file), os.path.join(chunk_dir, file))

# Define the source directory
source_directory = "images/train/cropped-images/"

# Call the function to split images into chunks
split_images_into_chunks(source_directory)
