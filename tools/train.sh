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
BASE_CONFIG_PATH="configs/animal_2d_keypoint/${MODE}/${DATASET_NAME}"
 
case $MODE in
    bottomup)
    echo "Preparing for bottomup processing..."
    configurations=(
        #"${BASE_CONFIG_PATH}/ae_hrnet-w32_8xb32-300e_${DATASET_NAME}-512x512.py"
        "${BASE_CONFIG_PATH}/cid_hrnet-w32_8xb32-140e_${DATASET_NAME}-512x512.py"
        "${BASE_CONFIG_PATH}/cid_hrnet-w48_8xb32-140e_${DATASET_NAME}-512x512.py"
        "${BASE_CONFIG_PATH}/dekr_hrnet-w32_8xb32-140e_${DATASET_NAME}-512x512.py"
        "${BASE_CONFIG_PATH}/dekr_hrnet-w48_8xb32-140e_${DATASET_NAME}-640x640.py"
    )
    ;;
    rtmo)
    echo "Preparing for RTMO processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/${MODE}-s_8xb16-600e_${DATASET_NAME}-640x640.py"
        "${BASE_CONFIG_PATH}/${MODE}-m_16xb16-600e_${DATASET_NAME}-640x640.py"
        "${BASE_CONFIG_PATH}/${MODE}-l_16xb16-600e_${DATASET_NAME}-640x640.py"
    )
    ;;
    rtmpose)
    echo "Preparing for RTMPose processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/${MODE}-t_8xb256-420e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/${MODE}-s_8xb256-420e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/${MODE}-m_8xb256-420e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/${MODE}-l_8xb256-420e_${DATASET_NAME}-256x256.py"
    )
    ;;
    topdown_heatmap)
    echo "Preparing for topdown heatmap processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_res50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_res101_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_res152_8xb32-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_hrnet-w32_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_hrnet-w48_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    topdown_others)
    echo "Preparing for topdown regression processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/ipr_res50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/simcc_res50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-reg_res50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-reg_res50_rle-8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-reg_res101_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-reg_res101_rle-8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-reg_res152_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-reg_res152_rle-8xb64-210e_${DATASET_NAME}-256x256.py" 

    )
    ;;
    yoloxpose)
    echo "Preparing for YOLOX-Pose processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/${MODE}_t_4xb64-300e_${DATASET_NAME}-416.py"
        "${BASE_CONFIG_PATH}/${MODE}_s_8xb32-300e_${DATASET_NAME}-640.py"
        "${BASE_CONFIG_PATH}/${MODE}_m_8xb32-300e_${DATASET_NAME}-640.py"
        "${BASE_CONFIG_PATH}/${MODE}_l_8xb32-300e_${DATASET_NAME}-640.py"
    )
    ;;
    vitpose)
    echo "Preparing for ViTPose processing..."
    configurations=(
        "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-small-simple_8xb64-210e_${DATASET_NAME}-256x256.py"
        "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-base-simple_8xb64-210e_${DATASET_NAME}-256x256.py"
        "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-large-simple_8xb64-210e_${DATASET_NAME}-256x256.py"
        "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-huge-simple_8xb64-210e_${DATASET_NAME}-256x256.py"
        "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-small_8xb64-210e_${DATASET_NAME}-256x256.py"
        "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-base_8xb64-210e_${DATASET_NAME}-256x256.py"
        "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-large_8xb64-210e_${DATASET_NAME}-256x256.py"
        "configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}/td-hm_ViTPose-huge_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    animalvitpose)
    echo "Preparing for AnimalViTPose processing with 256x256 input size..."
    configurations=(
        "configs/animal_2d_keypoint/animalvitpose/${DATASET_NAME}/animalvitpose-small_8xb64-210e_${DATASET_NAME}-256x256.py"
        "configs/animal_2d_keypoint/animalvitpose/${DATASET_NAME}/animalvitpose-base_8xb64-210e_${DATASET_NAME}-256x256.py"
        "configs/animal_2d_keypoint/animalvitpose/${DATASET_NAME}/animalvitpose-large_8xb64-210e_${DATASET_NAME}-256x256.py"
        "configs/animal_2d_keypoint/animalvitpose/${DATASET_NAME}/animalvitpose-huge_8xb64-210e_${DATASET_NAME}-256x256.py"
    ) 
    ;;
    *)
    echo "Unsupported mode: $MODE. Please choose a valid mode."
    exit 1
    ;;
esac

# Execute training based on the selected mode's configurations
for config in "${configurations[@]}"; do
    # Extracting configuration file name without path and extension for work_dir
    config_name=$(basename -- "$config")
    config_name="${config_name%.*}" # Removing the file extension
    work_dir="./work_dirs/${MODE}/${DATASET_NAME}/${config_name}"

    env $CUDA_COMMAND python tools/train.py "$config" --work-dir "$work_dir" --amp --auto-scale-lr
done