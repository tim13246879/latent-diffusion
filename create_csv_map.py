"""
Create a csv file with 3 columns: 'chunk', 'index', 'image_name'. Chunk refers to the number x in ./images/cropped-images/chunk-x. Index refers to the index of the image in the chunk. Image_name refers to the name of the image file.
"""
import csv
import os

def generate_csv():
    base_dir = './images/cropped-images'
    output_file = 'image_map.csv'
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['chunk', 'index', 'image_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # Iterate over each chunk directory
        for chunk_dir in sorted(os.listdir(base_dir), key=lambda x: int(x.split('-')[-1])):
            if os.path.isdir(os.path.join(base_dir, chunk_dir)):
                chunk_index = chunk_dir.split('-')[-1]  # Extract the chunk index from the folder name
                
                # List all images in the chunk directory
                images = os.listdir(os.path.join(base_dir, chunk_dir))
                for index, image_name in enumerate(images):
                    writer.writerow({'chunk': chunk_index, 'index': index, 'image_name': image_name})

# Call the function to generate the CSV
generate_csv()
