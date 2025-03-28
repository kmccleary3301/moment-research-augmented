#!/bin/bash

# Example script to run pretraining with a limited number of files
# This helps when you want to quickly test the pretraining process
# without using the entire dataset

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate moment_2

# Set parameters
CONFIG_PATH="configs/pretraining/pretrain.yaml"
GPU_ID=0
MAX_FILES=10  # Set to the number of files you want to use

# Run the pretraining script with the max_files parameter
python scripts/pretraining/pretraining.py \
  --config ${CONFIG_PATH} \
  --gpu_id ${GPU_ID} \
  --max_files ${MAX_FILES} 