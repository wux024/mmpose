#!/bin/bash


# Base configuration path shared across modes and dataset
BASE_CONFIG_PATH="work_dirs/animalvitpose/ap10k_gamma"
 

configurations=(
    "animalvitpose-small_8xb64-210e_ap10k_gamma0.5-256x256.py"
    "animalvitpose-small_8xb64-210e_ap10k_gamma1.0-256x256.py"
    "animalvitpose-small_8xb64-210e_ap10k_gamma1.5-256x256.py"
    "animalvitpose-small_8xb64-210e_ap10k_gamma2.5-256x256.py"
    "animalvitpose-small_8xb64-210e_ap10k_gamma3.0-256x256.py"
    )


# Execute training based on the selected mode's configurations
for config in "${configurations[@]}"; do
    # Extracting configuration file name without path and extension for work_dir
    config_name="${config%.*}"
    config_path=${BASE_CONFIG_PATH}/${config_name}/${config}
    checkpoint_path=$(find "$BASE_CONFIG_PATH/$config_name" -type f -name "*.pth" | sort -r | head -n 1)
    python tools/test.py "$config_path" "$checkpoint_path"
done