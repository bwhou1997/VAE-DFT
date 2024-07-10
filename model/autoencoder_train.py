import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


class Autoencoder_charge(nn.Module):
    def __init__(self, latent_dim=10):
        super(Autoencoder_charge, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(40, 24, kernel_size=3, padding=1),  # Adjust parameters as needed
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(392, latent_dim*2),
            # nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 392),
            nn.ReLU(),
            nn.Unflatten(1, (8, 7, 7)),
            nn.ConvTranspose2d(8, 24, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 40, kernel_size=2, stride=2),
            nn.Sigmoid()  # Sigmoid for values between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x = self.decoder(z)
        return x, mu, logvar

    def get_latent_space(self, x):
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    # def get_latent_space(self):
    #     return self.latent_space
    def no_grad(self):
        for param in self.parameters():
            param.requires_grad = False
        return self

reconstruction_function = nn.MSELoss(reduction='sum')
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss
    BCE = reconstruction_function(recon_x, x)

    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

