# `src/networks/policies.py`

## What this file does
This file defines the **policy network** (also called the actor). It decides what actions the agent should take.

## Main parts
- **`MLPPolicy.__init__(...)`**:
  - builds the neural network for the policy
  - uses a different output depending on whether the action space is discrete or continuous
  - creates the optimizer
- **`get_action(...)`**:
  - takes one observation
  - converts it to a tensor
  - runs the policy
  - samples an action from the returned distribution
  - converts it back to NumPy for the environment
- **`forward(...)`**:
  - returns a distribution over actions
  - `Categorical` for discrete actions
  - `Normal` for continuous actions
- **`MLPPolicyPG.update(...)`**:
  - computes the policy gradient loss
  - uses `log_prob(actions) * advantages`
  - backpropagates and updates the network

## In plain words
This file is the part that says **“given this state, what should I do?”**. During training, it also learns to make good actions more likely.

