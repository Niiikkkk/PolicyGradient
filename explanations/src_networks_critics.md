# `src/networks/critics.py`

## What this file does
This file defines the **value network** (also called the critic or baseline). It estimates how good a state is.

## Main parts
- **`ValueCritic.__init__(...)`**:
  - builds a neural network that maps observations to a single scalar value `V(s)`
  - creates an optimizer
- **`forward(...)`**:
  - runs the network on observations and returns predicted values
- **`update(...)`**:
  - compares predicted values to target Q-values
  - computes MSE loss
  - backpropagates and updates the network

## In plain words
The critic does **not** choose actions. Its job is to predict the expected return from a state, which helps reduce the noise in policy gradient training.

