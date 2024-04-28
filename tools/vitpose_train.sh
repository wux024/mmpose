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

# Determine CUDA command prefix based on GPU_NUMBER
if [[ "$GPU_NUMBER" != "0" && -n "$GPU_NUMBER" ]]; then
    CUDA_COMMAND="CUDA_VISIBLE_DEVICES=$GPU_NUMBER "
else
    CUDA_COMMAND="" # No need to set CUDA_VISIBLE_DEVICES for default or invalid values
fi


configurations=(
  "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-small_8xb64-210e_${DATASET_NAME}-256x192.py"
  "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-small-simple_8xb64-210e_${DATASET_NAME}-256x192.py"
  "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-base_8xb64-210e_${DATASET_NAME}-256x192.py"
  "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-base-simple_8xb64-210e_${DATASET_NAME}-256x192.py"
  "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-large_8xb64-210e_${DATASET_NAME}-256x192.py"
  "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-large-simple_8xb64-210e_${DATASET_NAME}-256x192.py"
  "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-huge_8xb64-210e_${DATASET_NAME}-256x192.py"
  "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-huge-simple_8xb64-210e_${DATASET_NAME}-256x192.py"
  )

for config in "${configurations[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU_NUMBER python tools/train.py "$config" --amp --auto-scale-lr
done

