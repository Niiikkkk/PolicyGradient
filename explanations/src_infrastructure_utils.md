# `src/infrastructure/utils.py`

## What this file does
This file handles **sampling rollouts** from the environment and turning them into training data.

## Main parts
- **`sample_trajectory(...)`**:
  - resets the environment
  - asks the policy for an action at each step
  - sends the action to the environment with `env.step(...)`
  - stores observations, actions, rewards, next observations, and terminal flags
  - stops when the episode ends or when `max_length` is reached
- **`sample_trajectories(...)`**:
  - collects multiple trajectories until enough timesteps are gathered for one training batch
- **`sample_n_trajectories(...)`**:
  - collects a fixed number of trajectories, usually for evaluation or videos
- **`compute_metrics(...)`**:
  - computes logging statistics like average return and average episode length
- **`convert_listofrollouts(...)`**:
  - concatenates trajectory data into flat arrays
- **`get_traj_length(...)`**:
  - returns the length of a trajectory

## In plain words
This file is the **data collector**. It turns the policy’s actions into trajectories, and trajectories into batches that the agent can learn from.

