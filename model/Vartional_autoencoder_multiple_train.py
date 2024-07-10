import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn.functional as F
from torch.optim import Adam

class VAE(nn.Module):
    def __init__(self, latent_dim=30):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(40, 60, kernel_size=3, padding=1),  # Adjust parameters as needed
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(60, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(392*4, latent_dim*2),
            # nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 392*4),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 60, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(60, 40, kernel_size=2, stride=2),
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


def train_vae(model, dataloader ,epochs=10, learning_rate=1e-3, beta=1.0):
    model = model()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        train_loss = 0
        for batch_data in dataloader:
            # If your dataset provides data in the form (data, labels), use batch_data[0]
            # If not, just use batch_data
            data = batch_data[0].float()

            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, beta)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(dataloader.dataset)}")




