#!/bin/bash

DATASET_NAME="ap10k"
GPU_NUMBER="0"
MODE="topdown_heatmap"


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            shift
            DATASET_NAME="$1"
            ;;
        --gpu)
            shift
            GPU_NUMBER="$1"
            ;;
        --mode)
            shift
            MODE="$1"
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset <dataset_name>] [--gpu <gpu_number>] [--mode {rtmo|rtmpose|bottomup|topdown_heatmap|topdown_regression|yoloxpose|vitpose}]"
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

# Base configuration path shared across modes and dataset
BASE_CONFIG_PATH="work_dirs/${MODE}/${DATASET_NAME}"
 
case $MODE in
    bottomup)
    echo "Preparing for bottomup processing..."
    configurations=(
        # "ae_hrnet-w32_8xb32-300e_${DATASET_NAME}-512x512.py"
        # "cid_hrnet-w32_8xb32-140e_${DATASET_NAME}-512x512.py"
        # "cid_hrnet-w48_8xb32-140e_${DATASET_NAME}-512x512.py"
        "dekr_hrnet-w32_8xb32-140e_${DATASET_NAME}-512x512.py"
        "dekr_hrnet-w48_8xb32-140e_${DATASET_NAME}-640x640.py"
    )
    ;;
    rtmo)
    echo "Preparing for RTMO processing..."
    configurations=(
        "rtmo-s_8xb16-600e_${DATASET_NAME}-640x640.py"
        "rtmo-m_16xb16-600e_${DATASET_NAME}-640x640.py"
        "rtmo-l_16xb16-600e_${DATASET_NAME}-640x640.py"
    )
    ;;
    rtmpose)
    echo "Preparing for RTMPose processing..."
    configurations=(
        "rtmpose-t_8xb256-420e_${DATASET_NAME}-256x256.py"
        "rtmpose-s_8xb256-420e_${DATASET_NAME}-256x256.py"
        "rtmpose-m_8xb256-420e_${DATASET_NAME}-256x256.py"
        "rtmpose-l_8xb256-420e_${DATASET_NAME}-256x256.py"
    )
    ;;
    topdown_heatmap)
    echo "Preparing for topdown heatmap processing..."
    configurations=(
        "td-hm_res50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "td-hm_res101_8xb64-210e_${DATASET_NAME}-256x256.py"
        "td-hm_res152_8xb32-210e_${DATASET_NAME}-256x256.py"
        "td-hm_hrnet-w32_8xb64-210e_${DATASET_NAME}-256x256.py"
        "td-hm_hrnet-w48_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    topdown_others)
    echo "Preparing for topdown regression processing..."
    configurations=(
        "ipr_res50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "td-reg_res50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "td-reg_res50_rle-8xb64-210e_${DATASET_NAME}-256x256.py"
        "td-reg_res101_8xb64-210e_${DATASET_NAME}-256x256.py"
        "td-reg_res101_rle-8xb64-210e_${DATASET_NAME}-256x256.py"
        "td-reg_res152_8xb64-210e_${DATASET_NAME}-256x256.py"
        "td-reg_res152_rle-8xb64-210e_${DATASET_NAME}-256x256.py" 
        "simcc_res50_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    yoloxpose)
    echo "Preparing for YOLOX-Pose processing..."
    configurations=(
        "yoloxpose_l_8xb32-300e_${DATASET_NAME}-640.py"
        "yoloxpose_m_8xb32-300e_${DATASET_NAME}-640.py"
        "yoloxpose_s_8xb32-300e_${DATASET_NAME}-640.py"
        "yoloxpose_t_4xb64-300e_${DATASET_NAME}-416.py"
    )
    ;;
    vitpose)
    echo "Preparing for ViTPose processing..."
    configurations=(
        "td-hm_ViTPose-small_8xb64-210e_${DATASET_NAME}-256x256.py"
        "td-hm_ViTPose-base_8xb64-210e_${DATASET_NAME}-256x256.py"
        "td-hm_ViTPose-large_8xb64-210e_${DATASET_NAME}-256x256.py"
        "td-hm_ViTPose-huge_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    animalvitpose)
    echo "Preparing for AnimalViTPose processing..."
    configurations=(
        "animalvitpose-small_8xb64-210e_${DATASET_NAME}-256x256.py"
        "animalvitpose-base_8xb64-210e_${DATASET_NAME}-256x256.py"
        "animalvitpose-large_8xb64-210e_${DATASET_NAME}-256x256.py"
        "animalvitpose-huge_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    *)
    echo "Unsupported mode: $MODE. Please choose a valid mode."
    exit 1
    ;;
esac

# Execute training based on the selected mode's configurations
for config in "${configurations[@]}"; do
    config_name="${config%.*}"
    config_path=${BASE_CONFIG_PATH}/${config_name}/${config}
    checkpoint_path=$(find "$BASE_CONFIG_PATH/$config_name" -type f -name "*.pth" | sort -r | head -n 1)
    env $CUDA_COMMAND python tools/test.py "$config_path" "$checkpoint_path"
done