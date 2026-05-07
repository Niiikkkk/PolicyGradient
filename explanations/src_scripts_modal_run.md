# `src/scripts/modal_run.py`

## What this file does
This file lets you run the training code on **Modal**, which is a cloud execution platform.

## Main parts
- **Imports**:
  - loads `main` and `setup_arguments` from `scripts.run`
- **Modal settings**:
  - defines the app name, volume, default GPU/CPU/memory, and container paths
- **`load_gitignore_patterns()`**:
  - reads `.gitignore` and turns entries into Modal ignore patterns
- **Image setup**:
  - builds a Docker-like container image with dependencies and project files
  - optionally copies your `.netrc` file for W&B authentication
- **`hw2_modal_remote(...)`**:
  - the Modal function that parses CLI arguments and launches training remotely
  - commits the volume after the run finishes

## In plain words
This file is the **cloud launcher**. It wraps the normal training script so you can run the same experiment on a remote machine instead of your laptop.

