import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch.optim import Adam


class RotationalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1,padding_mode='circular'):
        super(RotationalConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode)

    def forward(self, x):
        # Apply convolution to original, 90, 180, and 270 degree rotated inputs
        x0 = self.conv(x)
        x90 = self.conv(torch.rot90(x, k=1, dims=[2, 3]))
        x180 = self.conv(torch.rot90(x, k=2, dims=[2, 3]))
        x270 = self.conv(torch.rot90(x, k=3, dims=[2, 3]))
        
        # Combine the results (you can choose different ways to combine, here we use max)
        x_combined = torch.max(torch.max(x0, x90), torch.max(x180, x270))
        
        return x_combined



class VAE(nn.Module):
    def __init__(self, latent_dim=60):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            RotationalConv2d(40, 480, kernel_size=3, padding=1, padding_mode='circular'), # Adjust parameters as needed
            # Comment: the performance of VAE will increase obviously when increasing kernel_size, try using 3, 5, 7,...
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(480, 480, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            # nn.Linear(392*4, latent_dim*2),
            nn.Linear(480, latent_dim*2),
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
        # 0. Input size (preprocess)
        N, C, H, W = x.size()

        # Encode
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x = self.decoder(z)


        # 4 .Up-sampling (adaptive postprocess)
        upsampling = nn.Upsample(size=(H, W))
        x = upsampling(x)
        
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


def train_vae(model, dataloader ,epochs=10, learning_rate=0.01, beta=1.0):
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

