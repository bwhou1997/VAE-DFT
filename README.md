Unsupervised Learning for GW (autoencoder)
==========

# Package requirements

The Python packages below are required:
- [gpaw](https://wiki.fysik.dtu.dk/gpaw/) 21.1.1
- [ase](https://wiki.fysik.dtu.dk/ase/) 3.22
- [numpy](https://numpy.org/) 1.19.4
- [pandas](https://pandas.pydata.org/) 1.1.4
- h5py

## DFT-GW:

`conda` has some difficulties to correctly and automatically 
figure out the dependencies of `ase`.
It is suggested to run this project in a separate Anaconda environment:
```shell
conda create -n gpaw python=3.9.5 -y
conda activate gpaw
conda install numpy=1.22.0 -c conda-forge
conda install scipy -y
conda install -c conda-forge ase=3.22.1 -y
conda install libxc=4.3.4 -c conda-forge -y
conda install gpaw -c conda-forge -y
conda install h5py
```

## Machine learning:

The machine learning packages required by this project can be installed 
by running the follows:
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge torchinfo
conda install -c anaconda scikit-learn
pip install e3nn ase numpy
pip install git+https://github.com/muhrin/mrs-tutorial.git
```
Note that the version of CUDA should be determined by 
what is supported on your platform; 
it can be checked by `nvcc --version`.
The CUDA devices can be checked by `nvidia-smi`.


# Demo

- Following instruction in `./model/run`
- You should expect to see files in `./output`