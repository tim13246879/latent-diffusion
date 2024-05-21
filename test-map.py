import torch
import yaml
from PIL import Image
import os
from torchvision import transforms
from ldm.models.autoencoder import AutoencoderKL

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

path = './encoded_images/chunk-54.pt'
z = torch.load(path)
decoded_image = autoencoder.decode(z[389:390])
decoded_image = decoded_image.clamp(0, 1)
decoded_image = decoded_image * 255
decoded_image = decoded_image.type(torch.uint8)
img = transforms.ToPILImage()(decoded_image[0])
img.save(f'./image-test-map.png')
print(f"Image saved as 'image-test-map.png'")