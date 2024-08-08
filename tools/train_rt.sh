#!/bin/bash

dataset="${dataset}"
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
BASE_CONFIG_PATH="configs/animal_2d_keypoint/rt/${dataset}"

configurations=(
    "cspnext-t_udp_8xb64-210e_${dataset}-640x640.py"
    "cspnext-s_udp_8xb64-210e_${dataset}-640x640.py"
    "cspnext-m_udp_8xb64-210e_${dataset}-640x640.py"
    "cspnext-l_udp_8xb64-210e_${dataset}-640x640.py"
    "td-hm_litehrnet-18_8xb64-210e_${dataset}-640x640.py"
    "td-hm_litehrnet-30_8xb64-210e_${dataset}-640x640.py"
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
    config_name=$(basename -- "$config")
    config_name="${config_name%.*}" # Removing the file extension
    work_dir="./work_dirs/rt/${dataset}/${config_name}"

    python tools/train.py "$config" --work-dir "$work_dir" --amp --auto-scale-lr
done