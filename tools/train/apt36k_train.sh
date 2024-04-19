#!/bin/bash

configurations=(
  "configs/animal_2d_keypoint/topdown_heatmap/apt36k/td-hm_res50_8xb64-210e_apt36k-256x256.py"
  "configs/animal_2d_keypoint/topdown_heatmap/apt36k/td-hm_res101_8xb64-210e_apt36k-256x256.py"
  "configs/animal_2d_keypoint/topdown_heatmap/apt36k/td-hm_res152_8xb32-210e_apt36k-256x256.py"
  "configs/animal_2d_keypoint/topdown_heatmap/apt36k/td-hm_hrnet-w32_8xb64-210e_apt36k-256x256.py"
  "configs/animal_2d_keypoint/topdown_heatmap/apt36k/td-hm_hrnet-w48_8xb64-210e_apt36k-256x256.py"
  "configs/animal_2d_keypoint/topdown_heatmap/apt36k/cspnext-m_udp_8xb64-210e_apt36k-256x256.py"
)

for config in "${configurations[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python tools/train.py "$config" --amp --auto-scale-lr
done
