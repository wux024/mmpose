#!/bin/bash

# Base configuration path shared across modes and dataset


dataset=ap10k
imgsz=256

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            dataset="$2"
            shift 2
            ;;
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

BASE_CONFIG_PATH="configs/animal_2d_keypoint/rtmpose/${dataset}"
 

configurations=(
    "${BASE_CONFIG_PATH}/rtmpose-t_8xb256-420e_${dataset}-${imgsz}x${imgsz}.py"
    "${BASE_CONFIG_PATH}/rtmpose-s_8xb256-420e_${dataset}-${imgsz}x${imgsz}.py"
    "${BASE_CONFIG_PATH}/rtmpose-m_8xb256-420e_${dataset}-${imgsz}x${imgsz}.py"
    "${BASE_CONFIG_PATH}/rtmpose-l_8xb256-420e_${dataset}-${imgsz}x${imgsz}.py"
)

# Execute training based on the selected mode's configurations
for config in "${configurations[@]}"; do
    # Extracting configuration file name without path and extension for work_dir
    config_name=$(basename -- "$config")
    config_name="${config_name%.*}" # Removing the file extension
    work_dir="./work_dirs/rtmpose/${dataset}/${config_name}"

    python tools/train.py "$config" --work-dir "$work_dir" --amp --auto-scale-lr
done