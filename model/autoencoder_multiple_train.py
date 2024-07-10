import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, encode_layer=30, info_layer=1, quantum_number_later=1):
        super(Autoencoder, self).__init__()

        self.info_layer = info_layer
        self.encode_layer = encode_layer
        self.quantum_number_layer = quantum_number_later

        self.encoder = nn.Sequential(
            nn.Conv2d(40, 32, kernel_size=3, padding=1),  # Adjust parameters as needed
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(392*3, encode_layer),
            nn.ReLU()
        )

        self.encoder_info = nn.Sequential(
            nn.Linear(3, 8),  # Encode crucial information to the same size as the latent space
            nn.ReLU(),
            nn.Linear(8, info_layer)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encode_layer, 392*3),
            nn.ReLU(),
            nn.Unflatten(1, (24, 7, 7)),
            nn.ConvTranspose2d(24, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 40, kernel_size=2, stride=2),
            nn.Sigmoid()  # Sigmoid for values between 0 and 1
        )

        self.decoder_info = nn.Sequential(
            nn.Linear(info_layer, 8),  # Decode to the size of crucial information
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x_image, x_info):

        ####### partition info layer
        reciprocal_lattice = x_info[...,:3]
        quantum_number = x_info[...,3:]
        ####### partition info layer

        x_image = self.encoder(x_image)
        reciprocal_lattice = self.encoder_info(reciprocal_lattice)

        # print('x_image.shape', x_image.shape)
        # print('x_info.shape', x_info.shape)
        self.combined_latent = torch.cat((quantum_number, reciprocal_lattice, x_image), dim=1)
        # print('combined_latent.shape', combined_latent.shape)

        ####### partition latent space
        info_latent = self.combined_latent[:, self.quantum_number_layer:self.quantum_number_layer+self.info_layer]
        image_latent = self.combined_latent[:,self.info_layer+self.quantum_number_layer:]


        decoded_info = self.decoder_info(info_latent)  # You can also decode the image-encoded info if needed
        decoded_image = self.decoder(image_latent)

        ###### merge info output
        decoded_info = torch.cat((decoded_info, quantum_number), dim=1)

        return decoded_image, decoded_info

    def get_latent_space(self):
        return self.combined_latent


