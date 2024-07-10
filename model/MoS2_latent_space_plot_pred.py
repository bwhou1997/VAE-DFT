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
from MoS2_latent_space_plot_train import VAE
from Vartional_autoencoder_multiple_ROT_TI_GAP import VAE_rot
import os
import shutil
# from torchsummary import summary
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.manifold import TSNE

# you can turn on this to see CNN model which is invariant to traslational invariance, PBC, and rotation
# loaded_VAE = VAE_rot()
# # Load the saved state dictionary into the model
# loaded_VAE.load_state_dict(torch.load("VAE_model_rot_ti_pbc.pth", map_location=torch.device('cpu')))
# loaded_VAE.eval()  # Set the model to evaluation mode


loaded_VAE = VAE()
# Load the saved state dictionary into the model
loaded_VAE.load_state_dict(torch.load("MoS2_model.pth", map_location=torch.device('cpu')))
loaded_VAE.eval()  # Set the model to evaluation mode

f = h5.File('../data/wfs_merged_TMD.h5')
data = f['data'][()]
energy = f['energy'][()]
information = f['information'][:]

######### find states near Fermi Level #todo think about a good way to do this!
energy_index = list(np.where((energy < 0) & (-3 < energy))[0])
energy = energy[energy_index]
data = data[energy_index]
information = information[energy_index]
information[...,-1] = energy
######### find states near Fermi Level
f.close()

# Modify the code to pick 50 random figures
n =data.shape[0]
random_indices = random.sample(range(n), 1)

# Loop through the selected figures and plot them

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_VAE.to(device)
# summary(loaded_VAE, input_size=(data.shape)[1:])


if os.path.isdir('../results/TMD_VAE'): shutil.rmtree('../results/TMD_VAE')
os.makedirs('../results/TMD_VAE', exist_ok=True)
loss = nn.MSELoss()
for idx in range(20,40):
    print('plotting figures-%s' % idx)
    new_data = data[idx]
    new_data = new_data[np.newaxis, ...]
    new_data = torch.tensor(new_data, dtype=torch.float32)

    new_data = new_data / new_data.max()
    new_data = new_data.to(device)

    reconstructed_data, mu, logvar = loaded_VAE(new_data)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    im1 = ax1.imshow(new_data.detach().cpu().numpy()[0].sum(axis=0), cmap='viridis')
    ax1.set_title(f"Original Data - Figure {idx} energy %.2f"%energy[idx])
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(reconstructed_data.detach().cpu().numpy()[0].sum(axis=0), cmap='viridis')
    ax2.set_title(f"Reconstructed Data")
    fig.colorbar(im2, ax=ax2)


    mu = mu.detach().cpu().numpy()
    logvar = logvar.detach().cpu().numpy()
    ax3.bar(range(len(mu[0])),mu[0])
    ax3.set_title(f"Latent Space mu")

    ax4.bar(range(len(logvar[0])),logvar[0])
    ax4.set_title(f"Latent Space sigma")

    # Save the figure as a 'png' in the 'autoencoder_pred' directory
    if idx < 10:
        plt.savefig(f'../results/TMD_VAE/figure_0{idx}.png')
    else:
        plt.savefig(f'../results/TMD_VAE/figure_{idx}.png')

    plt.close()  # Close the figure to release resources

f.close()


############### latent space

fig1 = plt.figure()
new_all_data = torch.tensor(data, dtype=torch.float32)
new_all_data = new_all_data.to(device)
reconstructed_data, mu, logvar = loaded_VAE(new_all_data)

all_data = data
new_all_data = torch.tensor(all_data, dtype=torch.float32)
mu = mu.detach().cpu().numpy()
# plt.scatter(mu[:,0], mu[:,2], s=1)
# plt.close()  # Close the figure to release resources

# fig1.savefig(f'../results/TMD_VAE/latent.png')


################ t-SNE of latent space
###### 2D
# print('start tsne')
# tsne = TSNE(n_components=2, perplexity=150, random_state=42)
# latent_reduced = tsne.fit_transform(mu)
# print('start plotting')
# fig2 = plt.figure(figsize=(8, 6))
# plt.plot(latent_reduced[:60, 0], latent_reduced[:60, 1], linewidth=1,color='black', alpha=1)  # lacation in k space
# # ax = fig.add_subplot(111, projection='3d')
# plt.scatter(latent_reduced[:, 0], latent_reduced[:, 1], c=energy)  # lacation in k space

# plt.text(latent_reduced[0, 0], latent_reduced[0, 1]+0.1, r'$\Gamma$', fontsize=18, ha='center', va='center')
# plt.text(latent_reduced[19, 0], latent_reduced[19, 1]-0.1, 'M', fontsize=18, ha='center', va='center')
# plt.text(latent_reduced[33, 0], latent_reduced[33, 1]-0.1, 'K', fontsize=18,  ha='center', va='center')
# # plt.text(latent_reduced[, 0], latent_reduced[0, 1], '%/Gamma%', fontsize=12, color='red', ha='center', va='center')

# plt.colorbar(label='DFT Energy')
# plt.title('t-SNE Visualization of Iris Dataset')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.show()
# fig2.savefig(f'../results/TMD_VAE/tSNE.png')
# plt.close()

##### 3D
print('start tsne')
tsne = TSNE(n_components=3, perplexity=175, random_state=20)
latent_reduced = tsne.fit_transform(mu)
print('start plotting')
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with larger dots
scatter = ax.scatter(latent_reduced[:, 0], latent_reduced[:, 1], latent_reduced[:, 2], c=energy, s=70)  # Adjust the 's' parameter as needed

# Connect dots with lines
for i in range(179):
    if 0 <= i< 60:
        ax.plot([latent_reduced[i, 0], latent_reduced[i + 1, 0]],
                [latent_reduced[i, 1], latent_reduced[i + 1, 1]],
                [latent_reduced[i, 2], latent_reduced[i + 1, 2]],
                color='blue', alpha=1,linestyle='--')  # Adjust color and alpha as needed
    elif 60 <= i <120:
        ax.plot([latent_reduced[i, 0], latent_reduced[i + 1, 0]],
                [latent_reduced[i, 1], latent_reduced[i + 1, 1]],
                [latent_reduced[i, 2], latent_reduced[i + 1, 2]],
                color='darkgreen', alpha=1,linestyle='--')  # Adjust color and alpha as needed
    else:
        ax.plot([latent_reduced[i, 0], latent_reduced[i + 1, 0]],
                [latent_reduced[i, 1], latent_reduced[i + 1, 1]],
                [latent_reduced[i, 2], latent_reduced[i + 1, 2]],
                color='red', alpha=1,linestyle='--')  # Adjust color and alpha as needed
ax.set_xlabel('t-SNE Dimension 1', fontsize=15)  # Adjust the fontsize as needed
ax.set_ylabel('t-SNE Dimension 2', fontsize=15)  # Adjust the fontsize as needed
ax.set_zlabel('t-SNE Dimension 3', fontsize=15)  # Adjust the fontsize as needed

plt.show()
fig.savefig('../results/TMD_VAE/tSNE.png')


################# save as a GIF
# import glob
# import contextlib
# from PIL import Image
# # filepaths
# fp_in = "./../results/TMD_VAE/figure_*.png"
# fp_out = "./../results/TMD_VAE/image.gif"
# # use exit stack to automatically close opened images
# with contextlib.ExitStack() as stack:
#     # lazily load images
#     imgs = (stack.enter_context(Image.open(f))
#             for f in sorted(glob.glob(fp_in)))
#     # extract  first image from iterator
#     img = next(imgs)
#     # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
#     img.save(fp=fp_out, format='GIF', append_images=imgs,
#              save_all=True, duration=200, loop=0)