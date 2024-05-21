import torch
import yaml
from PIL import Image
import os
from torchvision import transforms
from ldm.models.autoencoder import AutoencoderKL

class Decoder:
    """
    something
    """
    def __init__(self, config_path=None, ckpt_path=None, device=None):            
        config_path = '~/ae-compress/latent_diffusion/models/first_stage_models/kl-f4/config.yaml'
        ckpt_path = '~/ae-compress/latent_diffusion/models/first_stage_models/kl-f4/model.ckpt'
        with open(os.path.expanduser(config_path), 'r') as file:
            config = yaml.safe_load(file)
        lossconfig = config['model']['params']['lossconfig']
        ddconfig = config['model']['params']['ddconfig']
        embed_dim = config['model']['params']['embed_dim']
        
        self.config_path = config_path
        self.ckpt_path = ckpt_path

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.autoencoder = AutoencoderKL(ddconfig=ddconfig, lossconfig=lossconfig, embed_dim=embed_dim, ckpt_path=os.path.expanduser(ckpt_path))
        self.autoencoder.to(self.device)
        self.CHUNK_SIZE = 30

    def decode(self, latent_tensor):
        processed_tensors = []
        for i in range(0, len(latent_tensor), self.CHUNK_SIZE):
            batch_slice = latent_tensor[i:i+self.CHUNK_SIZE]
            with torch.no_grad():
                decoded_images = self.autoencoder.decode(batch_slice)
                processed_tensors.append(decoded_images)
                del batch_slice, decoded_images
        final_tensor = torch.cat(processed_tensors, dim=0)
        return final_tensor

# Usage example
if __name__ == "__main__":
    encoded_images = '~/ae-compress/latent_diffusion/latent/decoded_latents/'
    cropped_images = '~/ae-compress/latent_diffusion/images/cropped-images/'
    decoded_images = '~/ae-compress/latent_diffusion/latent/decoded_latent_images/'

    decoder = Decoder()

    for chunk in os.listdir(encoded_images):
        image_chunk = decoder.decode(torch.load(os.path.join(encoded_images, chunk)))
        print("decoded images shape: ", image_chunk.shape)
        chunk_name = chunk.split('.')[0]
        image_list = os.listdir(os.path.join(cropped_images, chunk_name))
        for i, img in enumerate(image_chunk):
            image_name = image_list[i]
            img = img.clamp(0, 1)
            img = img * 255
            img = img.type(torch.uint8)
            output_dir = os.path.join(decoded_images, chunk_name)
            os.makedirs(output_dir, exist_ok=True)
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(output_dir, image_name))
            print(f"Image saved as '{image_name}'")



# # Path to your checkpoint file
# ckpt_path = "./models/first_stage_models/kl-f4/model.ckpt"

# with open('models/first_stage_models/kl-f4/config.yaml', 'r') as file:
#     config = yaml.safe_load(file)

# lossconfig = config['model']['params']['lossconfig']
# ddconfig = config['model']['params']['ddconfig']
# embed_dim = config['model']['params']['embed_dim']

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Create an instance of AutoencoderKL
# autoencoder = AutoencoderKL(ddconfig=ddconfig, lossconfig=lossconfig, embed_dim=embed_dim, ckpt_path=ckpt_path)
# autoencoder.to(device)

# CHUNK_SIZE = 30

# def batch_decode(batch_tensor):
#     processed_tensors = []
#     for i in range(0, len(batch_tensor), CHUNK_SIZE):
#         batch_slice = batch_tensor[i:i+CHUNK_SIZE]
#         with torch.no_grad():
#             decoded_images = autoencoder.decode(batch_slice)
#             processed_tensors.append(decoded_images)
#             del batch_slice, decoded_images
#     final_tensor = torch.cat(processed_tensors, dim=0)
#     return final_tensor

# encoded_images = './latent/decoded_latents/'
# cropped_images = './images/cropped-images/'
# decoded_images = './latent/decoded_latent_images/'

# for chunk in os.listdir(encoded_images):
#     image_chunk = batch_decode(torch.load(os.path.join(encoded_images, chunk)))
#     print("decoded images shape: ", image_chunk.shape)
#     chunk_name = chunk.split('.')[0]
#     image_list = os.listdir(os.path.join(cropped_images, chunk_name))
#     for i, img in enumerate(image_chunk):
#         image_name = image_list[i]
#         img = img.clamp(0, 1)
#         img = img * 255
#         img = img.type(torch.uint8)
#         output_dir = os.path.join(decoded_images, chunk_name)
#         os.makedirs(output_dir, exist_ok=True)
#         img = transforms.ToPILImage()(img)
#         img.save(os.path.join(output_dir, image_name))
#         print(f"Image saved as '{image_name}'")

#############################################################################################################


# z = torch.load('./encoded_images/chunk-0.pt')
# decoded_images = autoencoder.decode(z[:4])
# print("decoded images shape: ", decoded_images.shape)

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

# # Load the first image from the specified directory
# first_image_path = './images/cropped-images/chunk-0/'
# first_image_filename = os.listdir(first_image_path)[0]  # Assuming there is at least one image in the directory
# first_image = Image.open(os.path.join(first_image_path, first_image_filename)).convert('RGB')

# # Save the first image to a new file
# first_image.save('./original_image.png')
# print("Original image saved as 'original_image.png'")

