#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset_name> <gpu_number>"
    echo "Using default values:"
    DATASET_NAME=ap10k
    gpu_number=0
else
    DATASET_NAME=$1
    GPU_NUMBER=$2
fi

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

