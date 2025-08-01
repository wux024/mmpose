#!/bin/bash

# Base configuration path shared across modes and dataset
BASE_CONFIG_PATH="configs/animal_2d_keypoint/animalvitpose/ap10k_ab"
 

configurations=(
    "${BASE_CONFIG_PATH}/hrnet-w32_8xb64-210e_ap10k-256x256.py"
    "${BASE_CONFIG_PATH}/hrnet-w48_8xb64-210e_ap10k-256x256.py"
    "${BASE_CONFIG_PATH}/res50_8xb64-210e_ap10k-256x256.py"
    "${BASE_CONFIG_PATH}/res101_8xb64-210e_ap10k-256x256.py"
    "${BASE_CONFIG_PATH}/res152_8xb32-210e_ap10k-256x256.py"
    "${BASE_CONFIG_PATH}/swin-t-p4-w7_8xb64-210e_ap10k-256x256.py"
    "${BASE_CONFIG_PATH}/swin-b-p4-w7_8xb64-210e_ap10k-256x256.py"
    "${BASE_CONFIG_PATH}/swin-l-p4-w7_8xb64-210e_ap10k-256x256.py"
)

# Execute training based on the selected mode's configurations
for config in "${configurations[@]}"; do
    # Extracting configuration file name without path and extension for work_dir
    config_name="${config%.*}"
    config_path=${BASE_CONFIG_PATH}/${config_name}/${config}
    checkpoint_path=$(find "$BASE_CONFIG_PATH/$config_name" -type f -name "*.pth" | sort -r | head -n 1)
    env $CUDA_COMMAND python tools/test.py "$config_path" "$checkpoint_path"
done