#!/bin/bash

# Example script to run pretraining with a limited number of files
# using an optimized config for small runs

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate moment_2

# Set parameters
CONFIG_PATH="configs/pretraining/pretrain_small.yaml"
GPU_ID=0
MAX_FILES=10  # Set to the number of files you want to use

# Run the pretraining script with the max_files parameter and optimized config
python scripts/pretraining/pretraining.py \
  --config ${CONFIG_PATH} \
  --gpu_id ${GPU_ID} \
  --max_files ${MAX_FILES} 