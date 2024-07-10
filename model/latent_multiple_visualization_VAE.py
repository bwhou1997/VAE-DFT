import numpy as np
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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
from torchsummary import summary
# from autoencoder_multiple_train import Autoencoder
# import phate
import os
import shutil
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from Vartional_autoencoder_multiple_rot_ti_pbc_sota import VAE

# Load a sample dataset (Iris dataset in this case)





if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load autoencoder to get my latent space
    loaded_autoencoder = VAE()
    loaded_autoencoder.load_state_dict(torch.load("VAE_model_rot_ti_pbc.pth"))
    loaded_autoencoder.eval()  # Set the model to evaluation mode\

    f = h5.File('wfs_merged.h5')
    data = f['data'][()]
    energy = f['energy'][()]
    information = f['information'][:]
    GWenergy = f['GWenergy'][:]
    ######### find states near Fermi Level #todo think about a good way to do this!

    # GWenergy_index = list(np.where(GWenergy != -1000)[0])
    GWenergy_index = list(np.where((GWenergy != -1000) & (GWenergy - energy > -3) & (GWenergy - energy < 3))[0])
    GWenergy = GWenergy[GWenergy_index]
    energy = energy[GWenergy_index]
    data = data[GWenergy_index]
    information = information[GWenergy_index]

    # energy_index = list(np.where((energy < 3) & (-3 < energy))[0])
    # energy = energy[energy_index]
    # data = data[energy_index]
    # information = information[energy_index]
    ######### find states near Fermi Level

    data = torch.tensor(data, dtype=torch.float32)
    information = torch.tensor(information, dtype=torch.float32)

    con_batch, latent_space, logvar= loaded_autoencoder(data)
    # con_batch, latent_space, logvar = loaded_autoencoder.get_latent_space()

    ######################################## transfer all train data from autoencoder to current model

    latent_space = latent_space.detach().cpu().numpy()[...,2:]
    f.close()

    


############ TSNE
    # print('start tsne')
    # tsne = TSNE(n_components=2, perplexity=800, random_state=42)
    # latent_reduced = tsne.fit_transform(latent_space)
    # print('start plotting')
    # plt.figure(figsize=(8, 6))
    # plt.scatter(latent_reduced[:, 0], latent_reduced[:, 1], s=2, alpha=0.9, c=GWenergy-energy, cmap='bwr') # lacation in k space
    # plt.colorbar(label='E_GW - E_DFT')
    # plt.title('t-SNE Visualization of Iris Dataset')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.show()

    # np.savetxt('l.dat',np.array([latent_reduced[:, 0],latent_reduced[:, 1],GWenergy-energy]).T)
############ TSNE-3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting
    
    number_data = 40000

    print('start tsne')
    tsne = TSNE(n_components=3, 
            perplexity=18000, 
            random_state=42, 
            init='pca',
            verbose=2,
            n_iter=20000,
            learning_rate=1
            )  # Set n_components to 3 for 3D t-SNE
    # tsne = TSNE(n_components=3, 
    #             perplexity=15000, 
    #             random_state=42, 
    #             init='pca',
    #             verbose=2,
    #             n_iter=20000
    #             )  # Set n_components to 3 for 3D t-SNE
    latent_reduced = tsne.fit_transform(latent_space[:number_data])
    # latent_reduced = tsne.fit_transform(data.detach().cpu().numpy().reshape(data.shape[0], -1))
    print('start plotting')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
    
    # Scatter plot in 3D
    scatter = ax.scatter(latent_reduced[:, 0], latent_reduced[:, 1], latent_reduced[:, 2], s=2, alpha=0.9, c= (GWenergy - energy)[:number_data],
                         cmap='bwr')
    
    # Add a color bar
    cbar = plt.colorbar(scatter, label='E_GW - E_DFT')
    
    ax.set_title('3D t-SNE Visualization of Iris Dataset')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    
    plt.show()
    np.savetxt('l.dat',np.array([latent_reduced[:, 0],latent_reduced[:, 1],latent_reduced[:, 2], (GWenergy-energy)[:number_data]]).T)
############ Phate
    # print('start tsne')
    # phate_operator = phate.PHATE(n_components=2, decay = 20, knn=1000, t=100)
    # tree_phate = phate_operator.fit_transform(latent_space)
    # plt.scatter(tree_phate[:, 0], tree_phate[:, 1], s=1, c=energy, cmap='viridis')
    # plt.colorbar(label='Target Class')
    # plt.title('PHATE Visualization of Iris Dataset')
    # plt.xlabel('phate Dimension 1')
    # plt.ylabel('phate Dimension 2')
