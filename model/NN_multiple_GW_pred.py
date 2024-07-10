import h5py
import numpy as np
import torch
import random  # Add this import
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import h5py as h5
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#from torchsummary import summary
from autoencoder_multiple_train import Autoencoder
from autoencoder_train import Autoencoder_charge
import os
import shutil
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from NN_multiple_GW import GW_network
from Vartional_autoencoder_multiple_train import VAE
from autoencoder_super_train import Autoencoder_super

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load autoencoder to get my latent space
    loaded_autoencoder = Autoencoder()
    loaded_autoencoder.load_state_dict(torch.load("autoencoder_multiple_model.pth", map_location=torch.device('cpu')))
    loaded_autoencoder.eval()  # Set the model to evaluation mode\

    loaded_autoencoder_charge = Autoencoder_charge()
    loaded_autoencoder_charge.load_state_dict(torch.load("autoencoder_model.pth", map_location=torch.device('cpu')))
    loaded_autoencoder_charge.eval()  # Set the model to evaluation mode\

    loaded_autoencoder_super = Autoencoder_super()
    loaded_autoencoder_super.load_state_dict(torch.load("autoencoder_model_super.pth", map_location=torch.device('cpu')))
    loaded_autoencoder_super.eval()

    loaded_VAE = VAE()
    loaded_VAE.load_state_dict(torch.load('VAE_model.pth', map_location=torch.device('cpu')))
    loaded_VAE.eval()


    # f = h5.File('wfs_merged_pred.h5','r')
    # bnd_index = 1
    #
    #
    # data = f['data'][:]
    # energy = f['energy'][:]
    # kpt = f['information'][:]
    # information = np.zeros((len(energy), 4))
    # information[...,-1] = energy
    # information[...,:3] = kpt

    nbd_each_mat = 5
    state_index = range(0,300)
    f = h5.File('../data/wfs_merged_pred.h5','r')
    energy = f['energy'][state_index]
    data = f['data'][state_index]
    information = f['information'][state_index]
    charge_density = f['charge_density'][state_index]
    super_state = f['super_band'][state_index]
    information[..., -1] = energy
    f.close()

################### prepare input
    max_image = np.max(data, axis=(1, 2, 3))
    data = data / max_image[:, np.newaxis, np.newaxis, np.newaxis]

    # max_image_charge = np.max(charge_density, axis=(1, 2, 3))
    charge_density_1 = charge_density /  np.max(charge_density, axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
    charge_density_2 = charge_density**2 / np.max(charge_density**2, axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
    charge_density_3 = charge_density ** 3 / np.max(charge_density**3, axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
    charge_density_4 = charge_density ** 4 / np.max(charge_density ** 4, axis=(1, 2, 3))[:, np.newaxis, np.newaxis,np.newaxis]

    max_image_charge = torch.tensor(np.max(charge_density, axis=(1, 2, 3))[:, np.newaxis], dtype=torch.float32).to(device)
    data = torch.tensor(data, dtype=torch.float32)
    charge_density_1 = torch.tensor(charge_density_1, dtype=torch.float32)
    charge_density_2 = torch.tensor(charge_density_2, dtype=torch.float32)
    charge_density_3 = torch.tensor(charge_density_3, dtype=torch.float32)
    charge_density_4 = torch.tensor(charge_density_4, dtype=torch.float32)
    information = torch.tensor(information, dtype=torch.float32)

    super_state_1 = super_state[:, 0] / np.max(super_state[:, 0], axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
    super_state_2 = super_state[:, 1] / np.max(super_state[:, 1], axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
    super_state_3 = super_state[:, 2] / np.max(super_state[:, 2], axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]

    super_state_1 = torch.tensor(super_state_1, dtype=torch.float32)
    super_state_2 = torch.tensor(super_state_2, dtype=torch.float32)
    super_state_3 = torch.tensor(super_state_3, dtype=torch.float32)

    output_super1_VAE, mu_charge, logvar_charge = loaded_autoencoder_super(super_state_1)
    latent_space_super_1 = torch.cat((mu_charge, logvar_charge), dim=1)

    output_super2_VAE, mu_charge, logvar_charge = loaded_autoencoder_super(super_state_2)
    latent_space_super_2 = torch.cat((mu_charge, logvar_charge), dim=1)

    output_super3_VAE, mu_charge, logvar_charge = loaded_autoencoder_super(super_state_3)
    latent_space_super_3 = torch.cat((mu_charge, logvar_charge), dim=1)

    loss = nn.MSELoss()



    # energy
    output = loaded_autoencoder(data, information[...,9:])
    latent_space = loaded_autoencoder.get_latent_space()

    # charge
    output_charge_VAE, mu_charge, logvar_charge = loaded_autoencoder_charge(charge_density_1)
    latent_space_charge = torch.cat((mu_charge, logvar_charge), dim=1)
    # latent_space_charge = mu_charge

    # charge ^2 # todo
    output_charge2_VAE, mu_charge, logvar_charge = loaded_autoencoder_charge(charge_density_2)
    latent_space_charge2 = torch.cat((mu_charge, logvar_charge), dim=1)

    # charge ^3 # todo
    output_charge3_VAE, mu_charge, logvar_charge = loaded_autoencoder_charge(charge_density_3)
    latent_space_charge3 = torch.cat((mu_charge, logvar_charge), dim=1)

    # charge ^4 # todo
    output_charge4_VAE, mu_charge, logvar_charge = loaded_autoencoder_charge(charge_density_4)
    latent_space_charge4 = torch.cat((mu_charge, logvar_charge), dim=1)

    # wavefunction
    output_VAE, mu, logvar = loaded_VAE(data)
    latent_space_VAE = torch.cat((mu, logvar), dim=1)
    # latent_space_VAE = mu

    # latent_space[..., 2:] = torch.tensor(np.random.random(size=latent_space[..., 2:].shape), dtype=torch.float32) # misguide machine: this is for test

    energy = torch.tensor(energy, dtype=torch.float32)
    # r^2 test:


    del charge_density_1; del charge_density_2; del charge_density_3; del charge_density_4


    energy = energy.to(device)
    latent_space = torch.tensor(latent_space.detach().cpu().numpy(), dtype=torch.float32)[...,:].to(device)
    latent_space_VAE = torch.tensor(latent_space_VAE.detach().cpu().numpy(), dtype=torch.float32).to(device)
    latent_space_charge = torch.tensor(latent_space_charge.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)
    latent_space_charge2 = torch.tensor(latent_space_charge2.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)
    latent_space_charge3 = torch.tensor(latent_space_charge3.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)
    latent_space_charge4 = torch.tensor(latent_space_charge4.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)

    latent_space_super_1 = torch.tensor(latent_space_super_1.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)
    latent_space_super_2 = torch.tensor(latent_space_super_2.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)
    latent_space_super_3 = torch.tensor(latent_space_super_3.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)

    latent_space = torch.cat((max_image_charge, latent_space[...,:2], latent_space_VAE,
                              latent_space_charge, latent_space_charge2, latent_space_charge3, latent_space_charge4,
                              latent_space_super_1, latent_space_super_2, latent_space_super_3), dim=1) # charge latent space + |nk>

############################### predict

    NN_model = GW_network(latent_space.shape[1])
    NN_model.load_state_dict(torch.load("NN_multiple_GW_600000.pth", map_location=torch.device('cpu')))
    NN_model.to(device)

    criterion = nn.MSELoss()  # Use Mean Squared Error for regression
    optimizer = optim.SGD(NN_model.parameters(), lr=0.001)
    latent_space = torch.tensor(latent_space.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)
#################################


    # max_image = np.max(data, axis=(1, 2, 3))
    # data = data / max_image[:, np.newaxis, np.newaxis, np.newaxis]
    #
    #
    # data = torch.tensor(data, dtype=torch.float32)
    # information = torch.tensor(information, dtype=torch.float32)



    # output = loaded_autoencoder(data, information)
    # latent_space = loaded_autoencoder.get_latent_space()
    # output_VAE, mu, logvar = loaded_VAE(data)
    # latent_space_VAE = torch.cat((mu, logvar), dim=1)


    # latent_space = torch.tensor(latent_space.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)
    # latent_space_VAE = torch.tensor(latent_space_VAE.detach().cpu().numpy(), dtype=torch.float32).to(device)
    # latent_space = torch.cat((latent_space[..., :2], latent_space_VAE), dim=1)
    # criterion = nn.MSELoss()  # Use Mean Squared Error for regression


    # NN_model = GW_network(latent_space.shape[1])
    # NN_model.load_state_dict(torch.load("NN_multiple_GW.pth"))
    # NN_model.to(device)
    #
    # optimizer = optim.SGD(NN_model.parameters(), lr=0.01)

    NN_model.eval()
    res = NN_model(latent_space)
    res = res.detach().cpu().numpy()
    energy = energy.detach().cpu().numpy()

    plt.figure()



    dft_max = max(energy.reshape(60,nbd_each_mat)[...,0])
    gw_max = max((res.reshape(60,nbd_each_mat)+energy.reshape(60,nbd_each_mat))[...,0])

    plt.plot(energy.reshape(60,nbd_each_mat)-dft_max, color='blue',linestyle='dashed',)
    plt.plot(res.reshape(60,nbd_each_mat)+energy.reshape(60,nbd_each_mat)-gw_max, color='red')
  #  plt.legend()

    plt.savefig('../results/MoS2.png')

    print('GW bandstructures saved!')