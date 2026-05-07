# `src/infrastructure/log_utils.py`

## What this file does
This file handles **experiment logging and saving**.

## Main parts
- **`Logger`**:
  - writes metrics to a CSV file
  - sends metrics to Weights & Biases
  - can log rollout videos
- **`dump_log(...)`**:
  - saves the config, logging data, and trained agent parameters to disk
- **`setup_wandb(...)`**:
  - initializes a W&B run for tracking experiments
- **`get_wandb_video(...)`** and **`reshape_video(...)`**:
  - combine multiple rendered rollouts into a single video for logging
- **`remove_functions(...)`** and **`get_flag_dict(...)`**:
  - help clean up configuration dictionaries before saving

## In plain words
This file is the **record keeper**. It makes sure training results, saved models, and videos are stored properly so you can analyze them later.

