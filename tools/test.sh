#!/bin/bash

DATASET_NAME="ap10k"
GPU_NUMBER="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset=*)
            DATASET_NAME="${1#*=}"
            ;;
        --gpu=*)
            GPU_NUMBER="${1#*=}"
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset=<dataset_name>] [--gpu=<gpu_number>]"
            exit 1
            ;;
    esac
    shift
done

WORK_DIR="work_dirs"
base_config_path="${WORK_DIR}/td-hm_"

config_bases=(
  "res50_8xb64-210e_${DATASET_NAME}-256x256"
  "res101_8xb64-210e_${DATASET_NAME}-256x256"
  "res152_8xb32-210e_${DATASET_NAME}-256x256"
  "hrnet-w32_8xb64-210e_${DATASET_NAME}-256x256"
  "hrnet-w48_8xb64-210e_${DATASET_NAME}-256x256"
  "cspnext-m_udp_8xb64-210e_${DATASET_NAME}-256x256"
)

for config_base in "${config_bases[@]}"; do
    config_path="${base_config_path}${config_base}/${config_base}.py"
    dir_path="${base_config_path}${config_base}"
    CHECKPOINT_FILE=$(find "$dir_path" -maxdepth 1 -type f -name "*.pth" || true)
    
    if [ -n "$CHECKPOINT_FILE" ]; then
        CUDA_VISIBLE_DEVICES="$GPU_NUMBER" python tools/test.py "$config_path" "$CHECKPOINT_FILE"
    else
        echo "No .pth file found for ${config_base}"
    fi
done
