
import torch
from ldm.models.autoencoder import AutoencoderKL
from torchvision import transforms
from PIL import Image
import os
import yaml
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr



# Path to your checkpoint file
ckpt_path = "./models/first_stage_models/kl-f4/model.ckpt"

with open('models/first_stage_models/kl-f4/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

lossconfig = config['model']['params']['lossconfig']
ddconfig = config['model']['params']['ddconfig']
embed_dim = config['model']['params']['embed_dim']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create an instance of AutoencoderKL
autoencoder = AutoencoderKL(ddconfig=ddconfig, lossconfig=lossconfig, embed_dim=embed_dim, ckpt_path=ckpt_path)
autoencoder.to(device)

print(f'Using device: {device}')

cropped_images = './images/cropped-images/'

CHUNK_SIZE = 20

def batch_encode(batch_tensor):
    processed_tensors = []
    for i in range(0, len(batch_tensor), CHUNK_SIZE):
        batch_slice = batch_tensor[i:i+CHUNK_SIZE]
        with torch.no_grad():
            posterior = autoencoder.encode(batch_slice)
            z = posterior.sample()
            processed_tensors.append(z)
            del batch_slice, posterior, z
    final_tensor = torch.cat(processed_tensors, dim=0)    
    return final_tensor


for chunk in os.listdir(cropped_images):
    image_chunk = []
    for image_path in os.listdir(os.path.join(cropped_images, chunk)):
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            img = Image.open(os.path.join(cropped_images, chunk, image_path)).convert('RGB')
            image_chunk.append(img)
    tensors = [transforms.ToTensor()(image) for image in image_chunk]
    batch_tensor = torch.stack(tensors).to(device)
    print("batch shape: ", batch_tensor.shape)
    
    z = batch_encode(batch_tensor)

    # Save the tensor 'z' to a file
    torch.save(z, f'./encoded_images/{chunk}.pt')
    print("Latent tensor saved")



### TODO: encode the batch_tensor, separated into smaler chunks if needed.

# # Save the first original image to a file
# first_original_image = image_chunk[0]
# first_original_image.save('./original_image.png')
# print("Original image saved as 'original_image.png'")




# # Decode the sampled z and convert the output tensors to images
# decoded_images = autoencoder.decode(z)
# decoded_images = decoded_images.cpu().detach()
# print("decoded images shape: ", decoded_images.shape)
# print("decoded images: ", decoded_images)

# # Convert tensors to PIL Images and display them
# to_pil = transforms.ToPILImage()
# decoded_pil_images = []
# autoencoder.image_key = "segmentation"
# for img in decoded_images:
#     img = img.clamp(0, 1)
#     img = img * 255
#     img = img.type(torch.uint8)
#     decoded_pil_images.append(to_pil(img))

# # Save the first decoded image to a file
# first_decoded_image = decoded_pil_images[0]
# first_decoded_image.save('./decoded_image.png')
# print("Image saved as 'decoded_image.png'")


# # Load the images
# decoded_image1 = Image.open('./decoded_image.png')
# decoded_image2 = Image.open('./decoded_image2.png')
# original_image = Image.open('./original_image.png')

# # Convert images to numpy arrays
# decoded_image1_np = np.array(decoded_image1)
# decoded_image2_np = np.array(decoded_image2)
# original_image_np = np.array(original_image)

# # Calculate PSNR
# psnr_value = psnr(original_image_np, decoded_image1_np)
# print(f"PSNR between original image and decoded_image2: {psnr_value}")
