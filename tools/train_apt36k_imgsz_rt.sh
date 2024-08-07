#!/bin/bash

# Base configuration path shared across modes and dataset
BASE_CONFIG_PATH="configs/animal_2d_keypoint/rtmpose/apt36k"

imgsz=256

while [[ $# -gt 0 ]]; do
    case "$1" in
        --imgsz)
            imgsz="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

echo "Training with image size: $imgsz"
 

configurations=(
    "${BASE_CONFIG_PATH}/rtmpose-t_8xb256-420e_apt36k-${imgsz}x${imgsz}.py"
    "${BASE_CONFIG_PATH}/rtmpose-s_8xb256-420e_apt36k-${imgsz}x${imgsz}.py"
    "${BASE_CONFIG_PATH}/rtmpose-m_8xb256-420e_apt36k-${imgsz}x${imgsz}.py"
    "${BASE_CONFIG_PATH}/rtmpose-l_8xb256-420e_apt36k-${imgsz}x${imgsz}.py"
)

# Execute training based on the selected mode's configurations
for config in "${configurations[@]}"; do
    # Extracting configuration file name without path and extension for work_dir
    config_name=$(basename -- "$config")
    config_name="${config_name%.*}" # Removing the file extension
    work_dir="./work_dirs/rtmpose/apt36k/${config_name}"

    python tools/train.py "$config" --work-dir "$work_dir" --amp --auto-scale-lr
done