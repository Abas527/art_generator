import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import VAE
import streamlit as st

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_image(model,num_images=1,latent_dim=512):

    with torch.no_grad():
        z=torch.randn(num_images,latent_dim).to(device)
        samples=model.decoder(z)
        samples=(samples+1)/2
        samples=samples.to("cpu")
        return samples


def main():
    #defining and loading model

    latent_dim = 512
    img_channels = 3
    img_size = 128

    model = VAE(latent_dim=latent_dim, img_channels=img_channels, img_size=img_size).to(device)
    model.load_state_dict(torch.load("models/model.pth"))
    model.eval()


    #ui
    st.title("Image Generator")
    num_images = st.number_input("Number of images to generate", min_value=1, max_value=5, value=1)
    
    if(st.button("Generate")):
        images = generate_image(model,num_images=num_images,latent_dim=latent_dim)
        
        for i in range(num_images):
            image=to_pil_image(images[i])
            st.image(image)


if __name__ == "__main__":
    main()