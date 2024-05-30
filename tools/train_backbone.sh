#!/bin/bash

# bacbone training script for animal pose estimation topdown heatmap models

DATASET_NAME="ap10k"
GPU_NUMBER="0"
BACKBONE="resnet"


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
        --backbone)
            shift
            BACKBONE="$1"
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset <dataset_name>] [--gpu <gpu_number>] [--backbone {resnet|vit}]"
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
BASE_CONFIG_PATH="configs/animal_2d_keypoint/topdown_heatmap/${DATASET_NAME}"
 
case $BACKBONE in
    cspnext)
    echo "Preparing for CSPNeXt processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/cspnext-t_udp_8xb256-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/cspnext-s_udp_8xb256-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/cspnext-m_udp_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/cspnext-l_udp_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    mspn)
    echo "Preparing for MSPN processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_mspn50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_2xmspn50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_3xmspn50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_4xmspn50_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    rsn)
    echo "Preparing for RSNet processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_rsn18_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_rsn50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_2xrsn50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_3xrsn50_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    alexnet)
    echo "Preparing for AlexNet processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_alexnet_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    cpm)
    echo "Preparing for CPM processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_cpm_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    hourglass)
    echo "Preparing for Hourglass processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_hourglass52_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    hrformer)
    echo "Preparing for HRFormer processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_hrformer-small_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_hrformer-base_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    hrnet)
    echo "Preparing for HRNet processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_hrnet-w32_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_hrnet-w32_coarsedropout-8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_hrnet-w32_dark-8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_hrnet-w32_gridmask-8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_hrnet-w32_photometric-8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_hrnet-w32_udp-8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_hrnet-w32_udp-regress-8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_hrnet-w48_8xb32-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_hrnet-w48_dark-8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_hrnet-w48_udp-8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    litehrnet)
    echo "Preparing for LiteHRNet processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_litehrnet-18_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_litehrnet-30_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    mobilenetv2)
    echo "Preparing for MobileNetV2 processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_mobilenetv2_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    pvt)
    echo "Preparing for PVT processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_pvt-s_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_pvtv2-b2_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    resnet)
    echo "Preparing for ResNet processing..."
    configurations=(        
        "${BASE_CONFIG_PATH}/td-hm_res50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_res50_dark-8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_res101_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_res101_dark-8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_res152_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_res152_dark-8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    resnest)
    echo "Preparing for ResNeSt processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_resnest50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_resnest101_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_resnest200_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_resnest269_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    resnext)
    echo "Preparing for ResNeXt processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_resnext50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_resnext101_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_resnext152_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    resnetv1d)
    echo "Preparing for ResNetV1d processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_resnetv1d50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_resnetv1d101_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_resnetv1d152_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    scnet)
    echo "Preparing for SCNet processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_scnet50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_scnet101_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    seresnet)
    echo "Preparing for SEResNet processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_seresnet50_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_seresnet101_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_seresnet152_8xb32-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    shufflenet)
    echo "Preparing for ShuffleNet processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_shufflenetv1_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_shufflenetv2_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    swin)
    echo "Preparing for Swin processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_swin-t-p4-w7_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_swin-b-p4-w7_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_swin-l-p4-w7_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    vgg16)
    echo "Preparing for VGG16 processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_vgg16-bn_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    vipnas)
    echo "Preparing for VIPNAS processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_vipnas-mbv3_8xb64-210e_${DATASET_NAME}-256x256.py"
         "${BASE_CONFIG_PATH}/td-hm_vipnas-res50-8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    vitpose)
    echo "Preparing for ViTPose processing..."
    configurations=(
        "${BASE_CONFIG_PATH}/td-hm_ViTPose-small_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_ViTPose-small-simple_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_ViTPose-base_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_ViTPose-base-simple_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_ViTPose-large_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_ViTPose-large-simple_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_ViTPose-huge_8xb64-210e_${DATASET_NAME}-256x256.py"
        "${BASE_CONFIG_PATH}/td-hm_ViTPose-huge-simple_8xb64-210e_${DATASET_NAME}-256x256.py"
    )
    ;;
    *)
    echo "Unsupported mode: $BACKBONE. Please choose a valid mode."
    exit 1
    ;;
esac

# Execute training based on the selected mode's configurations
for config in "${configurations[@]}"; do
    # Extracting configuration file name without path and extension for work_dir
    config_name=$(basename -- "$config")
    config_name="${config_name%.*}" # Removing the file extension
    work_dir="./work_dirs/${BACKBONE}/${DATASET_NAME}/${config_name}"

    env $CUDA_COMMAND python tools/train.py "$config" --work-dir "$work_dir" --amp --auto-scale-lr
done