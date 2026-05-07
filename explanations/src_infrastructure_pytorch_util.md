# `src/infrastructure/pytorch_util.py`

## What this file does
This file contains **PyTorch helper utilities** used throughout the project.

## Main parts
- **`build_mlp(...)`**:
  - builds a feedforward neural network with a chosen number of layers and hidden size
- **`init_gpu(...)`**:
  - chooses CPU or GPU and stores the device in a global variable
- **`set_device(...)`**:
  - manually sets the CUDA device
- **`from_numpy(...)`**:
  - converts NumPy arrays into float torch tensors on the correct device
- **`to_numpy(...)`**:
  - converts torch tensors back to NumPy arrays

## In plain words
This file is the project’s **PyTorch toolbox**. It keeps the code cleaner by centralizing common tasks like building networks and moving data between NumPy and PyTorch.

