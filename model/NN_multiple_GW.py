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
# from torchsummary import summary
from autoencoder_multiple_train import Autoencoder
from autoencoder_train import Autoencoder_charge
from autoencoder_super_train import  Autoencoder_super
from Vartional_autoencoder_multiple_train import  VAE
import os
import shutil
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class GW_network(nn.Module):
    def __init__(self, input_size):
        super(GW_network, self).__init__()
        self.dftlayer = nn.Sequential(
            nn.Linear(input_size, 120),
            nn.ReLU(),
            nn.Linear(120, 80),
            # nn.ReLU(),
            # nn.Linear(120, 80),
            nn.ReLU(),
            nn.Linear(80,1),
        )

    def forward(self, x):
        out = self.dftlayer(x)
        return out



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load autoencoder to get my latent space
    loaded_autoencoder = Autoencoder()
    loaded_autoencoder.load_state_dict(torch.load("autoencoder_multiple_model.pth"))
    loaded_autoencoder.eval()  # Set the model to evaluation mode\

    loaded_autoencoder_charge = Autoencoder_charge()
    loaded_autoencoder_charge.load_state_dict(torch.load("autoencoder_model.pth"))
    loaded_autoencoder_charge.eval()  # Set the model to evaluation mode\

    loaded_autoencoder_super = Autoencoder_super()
    loaded_autoencoder_super.load_state_dict(torch.load("autoencoder_model_super.pth"))
    loaded_autoencoder_super.eval()

    loaded_VAE = VAE()
    loaded_VAE.load_state_dict(torch.load('VAE_model.pth'))
    loaded_VAE.eval()



    f = h5.File('wfs_merged.h5')
    GWenergy = f['GWenergy'][()]
    energy = f['energy'][()]
    GWenergy_index = list(np.where((GWenergy != -1000) & (GWenergy - energy > -5) & (GWenergy - energy < 4.7))[0])

    GWenergy = GWenergy[GWenergy_index]
    energy = energy[GWenergy_index]
    data = f['data'][GWenergy_index]
    information = f['information'][GWenergy_index]
    charge_density = f['charge_density'][GWenergy_index]
    super_state = f['super_band'][GWenergy_index]
    ######### find states near Fermi Level #todo think about a good way to do this!

    # GWenergy_index = list(np.where((GWenergy > -10) )[0])
    # GWenergy = GWenergy[GWenergy_index]
    # energy = energy[GWenergy_index]
    # data = data[GWenergy_index]
    # charge_density= charge_density[GWenergy_index]
    # information = information[GWenergy_index]
    # super_state = super_state[GWenergy_index]

    information[..., -1] = energy
    ######### find states near Fermi Level
    f.close()

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


    ########## super band 12/3/2023
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
    print('super1 r^2:', 1-  loss(super_state_1, output_super1_VAE) / np.var(super_state_1.detach().cpu().numpy()))
    print('super2 r^2:', 1 - loss(super_state_2, output_super2_VAE) / np.var(super_state_2.detach().cpu().numpy()))
    print('super3 r^2:', 1 - loss(super_state_3, output_super3_VAE) / np.var(super_state_3.detach().cpu().numpy()))

    del super_state_1; del super_state_2; del super_state_3
    ###############################


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
    print('energy r^2:', 1-loss(latent_space[...,0], energy)/np.var(energy.detach().cpu().numpy()))
    print('charge r^2:', 1-loss(charge_density_1, output_charge_VAE) / np.var(charge_density_1.detach().cpu().numpy()))
    print('charge2 r^2:', 1-loss(charge_density_2, output_charge2_VAE) / np.var(charge_density_2.detach().cpu().numpy()))
    print('charge3 r^2:', 1-loss(charge_density_3, output_charge3_VAE) / np.var(charge_density_3.detach().cpu().numpy()))
    print('charge4 r^2:', 1 - loss(charge_density_4, output_charge4_VAE) / np.var(charge_density_4.detach().cpu().numpy()))
    print('data r^2:', 1-loss(output_VAE, data)/np.var(data.detach().cpu().numpy()))

    del charge_density_1; del charge_density_2; del charge_density_3; del charge_density_4
    ######################################## transfer all train data from autoencoder to current model

    energy = energy.to(device)
    GWenergy = torch.tensor(GWenergy, dtype=torch.float32).to(device)
    latent_space = torch.tensor(latent_space.detach().cpu().numpy(), dtype=torch.float32)[...,:].to(device)
    latent_space_VAE = torch.tensor(latent_space_VAE.detach().cpu().numpy(), dtype=torch.float32).to(device)
    latent_space_charge = torch.tensor(latent_space_charge.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)
    latent_space_charge2 = torch.tensor(latent_space_charge2.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)
    latent_space_charge3 = torch.tensor(latent_space_charge3.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)
    latent_space_charge4 = torch.tensor(latent_space_charge4.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)

    latent_space_super_1 = torch.tensor(latent_space_super_1.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)
    latent_space_super_2 = torch.tensor(latent_space_super_2.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)
    latent_space_super_3 = torch.tensor(latent_space_super_3.detach().cpu().numpy(), dtype=torch.float32)[..., :].to(device)

    # latent_space = torch.cat((max_image_charge, latent_space[...,:2], latent_space_VAE,
    #                         latent_space_charge, latent_space_charge2, latent_space_charge3, latent_space_charge4), dim=1) # charge latent space + |nk>

    latent_space = torch.cat((max_image_charge, latent_space[...,:2], latent_space_VAE,
                              latent_space_charge, latent_space_charge2, latent_space_charge3, latent_space_charge4,
                              latent_space_super_1, latent_space_super_2, latent_space_super_3), dim=1) # charge latent space + |nk>
    
    # latent_space = torch.cat((max_image_charge, latent_space[...,:2], latent_space_VAE,
    #                           latent_space_charge, latent_space_charge2, latent_space_charge3, latent_space_charge4), dim=1) # charge latent space + |nk>
    # latent_space = torch.cat((latent_space[...,:2], latent_space_VAE), dim=1) # charge latent space + |nk>
    # latent_space = torch.cat((latent_space[...,:2], latent_space_VAE, latent_space_charge), dim=1) # charge latent space + |nk>
    # latent_space = torch.cat((latent_space[...,:2], latent_space_VAE), dim=1)
    # latent_space = latent_space[...,:2]
    # latent_space = latent_space_VAE

    f.close()

    ################################## Create Train and Test Data

    ####### save 10% random states for test
    X_train, X_test, y_train, y_test = train_test_split(latent_space, GWenergy - energy, test_size=0.1, random_state=41) # we need to learn correction!
    print('X_train size:', X_train.shape)
    print('y_train size:', y_train.shape)
    print('X_test size:', X_test.shape)
    print('y_test size:', y_test.shape)

    ####### save the last 10% materials for test
    # total_samples = len(GWenergy - energy)
    # split_index = int(0.90 * total_samples)

    # X_train, X_test = latent_space[:split_index], latent_space[split_index:]
    # y_train, y_test = (GWenergy - energy)[:split_index], (GWenergy - energy)[split_index:]
    # print('X_train size:', X_train.shape)
    # print('y_train size:', y_train.shape)
    # print('X_test size:', X_test.shape)
    # print('y_test size:', y_test.shape)



#### here, I comment below


    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=600, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=600, shuffle=False)

    input_size = latent_space.shape[1]  # The dimension of your data
    # hidden_size = 64
    # output_size = 1  # For regression, output size is typically 1
    model = GW_network(input_size).to(device)
    criterion = nn.MSELoss()  # Use Mean Squared Error for regression
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    num_epochs =58000

    # summary(model, input_size=input_size)
    train_error = np.zeros(num_epochs)
    test_error = np.zeros(num_epochs)
    print('starting training DFT NN')

    # train_error = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.view(-1, 1)
            loss = criterion(outputs, targets)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_error[epoch] = train_loss / len(train_loader)
        print(f"Epoch Train [{epoch + 1}/{num_epochs}], loss: {train_error[epoch] :.4f}")


        # Evaluate the model on the test data
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                targets = targets.view(-1, 1)
                test_loss += criterion(outputs, targets).item()

        test_error[epoch] = test_loss / len(test_loader)
        print(f"Epoch Test [{epoch + 1}/{num_epochs}], loss: {test_error[epoch] :.4f}")

    print('Training and testing complete.')

    var_train = np.var(y_train.detach().cpu().numpy())
    var_test = np.var(y_test.detach().cpu().numpy())
    plt.plot(1-train_error/var_train, label='train r^2')
    plt.plot(1 - test_error / var_test, label='test r^2')
    plt.ylim(0., 1)
    plt.legend()
    plt.savefig('r2.png')

    torch.save(model.state_dict(), "NN_multiple_GW.pth")

############### plot
    plt.figure()
    n_train = X_train.shape[0]
    res_train = []
    outputs_train = model(X_train)
    outputs_train = outputs_train.detach().cpu().numpy()
    targets_train = y_train.detach().cpu().numpy()
    for i in range(n_train):
        res_train.append([targets_train[i], outputs_train[i][0]])
    res_train.sort()
    res_train = np.array(res_train)
    plt.scatter(res_train[:,0], res_train[:,1], label='Train',s=3)

    n_test = X_test.shape[0]
    res_test = []
    outputs_test = model(X_test)
    outputs_test = outputs_test.detach().cpu().numpy()
    targets_test = y_test.detach().cpu().numpy()
    for i in range(n_test):
        res_test.append([targets_test[i], outputs_test[i][0]])
    res_test.sort()
    res_test = np.array(res_test)
    plt.scatter(res_test[:,0], res_test[:,1], label='Test',s=3)

    plt.xlabel('Calculated GW corrections [eV]', fontsize=12)
    plt.ylabel('Predicted GW corrections [eV]', fontsize=12)
    # plt.xlim(-1., 5)
    # plt.ylim(-1., 5)
    plt.legend()
    plt.savefig('predvsreal.png')

    print('GW MAE:',np.mean(np.sqrt(test_error[-1000:])))
    print('R2:',np.mean((1 - test_error / var_test)[-1000:]))

    # 11/02/2024: it seems outlier doesn't affect prediction too much. 
