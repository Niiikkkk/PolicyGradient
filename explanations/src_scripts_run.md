# `src/scripts/run.py`

## What this file does
This is the **main entry point** for training. It starts the experiment, creates the environment, builds the agent, and runs the training loop.

## Main parts
- **Imports and path setup**: makes sure Python can import modules from `src/`.
- **`run_training_loop(logger, args)`**:
  - sets random seeds
  - chooses CPU or GPU
  - creates the Gym environment
  - figures out observation/action sizes
  - builds the `PGAgent`
  - repeatedly collects trajectories and trains the agent
  - logs training and evaluation results
  - optionally saves videos
- **`setup_arguments()`**: defines command-line flags like environment name, learning rate, batch size, and whether to use a baseline.
- **`main(args)`**: creates the logging directory, starts W&B logging, and launches training.

## In plain words
Think of this file as the **experiment controller**. It does not learn the policy itself; it tells the rest of the code when to collect data, when to train, and when to log results.

