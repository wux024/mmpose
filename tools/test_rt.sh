#!/bin/bash

dataset="ap10k"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            dataset="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Base configuration path shared across modes and dataset
BASE_CONFIG_PATH="work_dirs/rt/${dataset}"

configurations=(
    "cspnext-t_udp_8xb256-210e_${dataset}-640x640.py"
    "cspnext-s_udp_8xb256-210e_${dataset}-640x640.py"
    "cspnext-m_udp_8xb64-210e_${dataset}-640x640.py"
    "cspnext-l_udp_8xb64-210e_${dataset}-640x640.py"
    "td-hm_mobilenetv2_8xb64-210e_${dataset}-640x640.py"
    "td-hm_shufflenetv1_8xb64-210e_${dataset}-640x640.py"
    "td-hm_shufflenetv2_8xb64-210e_${dataset}-640x640.py"
    "td-hm_vipnas-mbv3_8xb64-210e_${dataset}-640x640.py"
    "td-hm_vipnas-res50_8xb64-210e_${dataset}-640x640.py"
    "td-reg_mobilenetv2_rle-8xb64-210e_${dataset}-640x640.py.py"
    "simcc_mobilenetv2_wo-deconv-8xb64-210e_${dataset}-640x640.py"
    "simcc_vipnas-mbv3_8xb64-210e_${dataset}-640x640.py"
)

# Execute training based on the selected mode's configurations
for config in "${configurations[@]}"; do
    # Extracting configuration file name without path and extension for work_dir
    config_name="${config%.*}" # Removing the file extension
    config_path=${BASE_CONFIG_PATH}/${config_name}/${config}
    checkpoint_path=$(find "$BASE_CONFIG_PATH/$config_name" -type f -name "*.pth" | sort -r | head -n 1)
    python tools/test.py "$config_path" "$checkpoint_path"
done