#!/bin/bash

# Base configuration path shared across modes and dataset
BASE_CONFIG_PATH="configs/animal_2d_keypoint/animalvitpose/ap10k_ab"
 

configurations=(
    "${BASE_CONFIG_PATH}/cspnext-t-210e_ap10k-256x256.py"
    "${BASE_CONFIG_PATH}/cspnext-s-210e_ap10k-256x256.py"
    "${BASE_CONFIG_PATH}/cspnext-m-210e_ap10k-256x256.py"
    "${BASE_CONFIG_PATH}/cspnext-l-210e_ap10k-256x256.py"
    "${BASE_CONFIG_PATH}/shufflenetv1_8xb64-210e_ap10k-256x256.py"
    "${BASE_CONFIG_PATH}/shufflenetv2_8xb64-210e_ap10k-256x256.py"
)

# Execute training based on the selected mode's configurations
for config in "${configurations[@]}"; do
    # Extracting configuration file name without path and extension for work_dir
    config_name=$(basename -- "$config")
    config_name="${config_name%.*}" # Removing the file extension
    work_dir="./work_dirs/animalvitpose/ap10k_ab/${config_name}"

    python tools/train.py "$config" --work-dir "$work_dir" --amp --auto-scale-lr
done