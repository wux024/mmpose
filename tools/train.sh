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

configurations=(
  "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_res50_8xb64-210e_${DATASET_NAME}-256x256.py"
  "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_res101_8xb64-210e_${DATASET_NAME}-256x256.py"
  "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_res152_8xb32-210e_${DATASET_NAME}-256x256.py"
  "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_hrnet-w32_8xb64-210e_${DATASET_NAME}-256x256.py"
  "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_hrnet-w48_8xb64-210e_${DATASET_NAME}-256x256.py"
  "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/cspnext-m_udp_8xb64-210e_${DATASET_NAME}-256x256.py"
)

for config in "${configurations[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU_NUMBER python tools/train.py "$config" --amp --auto-scale-lr
done

