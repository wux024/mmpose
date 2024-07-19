#!/bin/bash


# Base configuration path shared across modes and dataset
BASE_CONFIG_PATH="work_dirs/animalvitpose/ap10k"
 

configurations=(
    "animalvitpose-small_8xb64-210e_ap10k-256x256.py"
    "animalvitpose-base_8xb64-210e_ap10k-256x256.py"
    "animalvitpose-large_8xb64-210e_ap10k-256x256.py"
    "animalvitpose-huge_8xb64-210e_ap10k-256x256.py"
    "animalvitpose-small_8xb64-210e_ap10k-384x384.py"
    "animalvitpose-base_8xb64-210e_ap10k-384x384.py"
    "animalvitpose-large_8xb64-210e_ap10k-384x384.py"
    "animalvitpose-huge_8xb64-210e_ap10k-384x384.py"
    "animalvitpose-small_8xb64-210e_ap10k-512x512.py"
    "animalvitpose-base_8xb64-210e_ap10k-512x512.py"
    "animalvitpose-large_8xb64-210e_ap10k-512x512.py"
    "animalvitpose-huge_8xb64-210e_ap10k-512x512.py"
    )


# Execute training based on the selected mode's configurations
for config in "${configurations[@]}"; do
    # Extracting configuration file name without path and extension for work_dir
    config_name="${config%.*}"
    config_path=${BASE_CONFIG_PATH}/${config_name}/${config}
    checkpoint_path=$(find "$BASE_CONFIG_PATH/$config_name" -type f -name "*.pth" | sort -r | head -n 1)
    python tools/test.py "$config_path" "$checkpoint_path"
done