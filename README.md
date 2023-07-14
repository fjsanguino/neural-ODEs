# Neural-ODEs
We reproduce the results of Neural Ordinary Differential Equations (Ricky T.Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud, 2018). Our code implementation is independent of the torchdiffeq implementation of the authors, making use only of the AdamsBashforthMoulton integrator (https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/fixed_adams.py) for compatibility with PyTorch.

## Code
* `model.py` contains implementations for MLPs, ResNets, and ODENets.
* `MNIST.py` runs the MNIST task.
* `cifar10.py` runs the CIFAR task.
* `MNIST_denoise.py` runs the Image Denoising task.

Each task contains a variable `MODEL` which can be set to MLP, Paper, or ODENet to train the MLP, ResNet, or ODENet models, respectively.