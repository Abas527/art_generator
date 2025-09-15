import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Encoder

class Encoder(nn.Module):
    def __init__(self, latent_dim=256, img_channels=3, img_size=128):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),    # 128 -> 64
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),             # 64 -> 32
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),            # 32 -> 16
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),            # 16 -> 8
            nn.ReLU(True),
            nn.Conv2d(512, 512, 4, 2, 1),            # 8 -> 4
            nn.ReLU(True),
        )

        # Dynamically compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, img_channels, img_size, img_size)
            conv_out = self.conv(dummy)
            self.flattened_size = conv_out.view(1, -1).size(1)

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Decoder

class Decoder(nn.Module):
    def __init__(self, latent_dim=256, img_channels=3):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512*4*4)  # start from 512x4x4

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),  # 4 -> 8
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8 -> 16
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16 -> 32
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32 -> 64
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),  # 64 -> 128
            nn.Tanh(),  # Output in [-1,1]
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 512, 4, 4)
        x_recon = self.deconv(h)
        return x_recon

# -----------------------------
# VAE
# -----------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=256, img_channels=3, img_size=128):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, img_channels, img_size)
        self.decoder = Decoder(latent_dim, img_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, perceptual_model=None):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (recon_loss + kl_loss) / x.size(0)  # average per batch


if __name__ == "__main__":
    model = VAE(latent_dim=256, img_channels=3, img_size=128)
    dummy = torch.randn(4, 3, 128, 128)
    recon, mu, logvar = model(dummy)
    print("Input:", dummy.shape)
    print("Reconstructed:", recon.shape)
    print("Mu:", mu.shape)
    print("Logvar:", logvar.shape)
