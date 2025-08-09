# Base configuration path shared across modes and dataset
$BASE_CONFIG_PATH = "configs/animal_2d_keypoint/animalvitpose/ap10k_ab"

$configurations = @(
    "${BASE_CONFIG_PATH}/animalvitpose-small_8xb64-210e_ap10k-256x256-nodeconv.py",
    "${BASE_CONFIG_PATH}/animalvitpose-small_8xb64-210e_ap10k-256x256-nooneconv.py",
    "${BASE_CONFIG_PATH}/animalvitpose-small_8xb64-210e_ap10k-256x256-nodeconvandoneconv.py",
    "${BASE_CONFIG_PATH}/cspnext-t-210e_ap10k-256x256.py",
    "${BASE_CONFIG_PATH}/cspnext-s-210e_ap10k-256x256.py",
    "${BASE_CONFIG_PATH}/cspnext-m-210e_ap10k-256x256.py",
    "${BASE_CONFIG_PATH}/cspnext-l-210e_ap10k-256x256.py",
    "${BASE_CONFIG_PATH}/hrnet-w32_8xb64-210e_ap10k-256x256.py",
    "${BASE_CONFIG_PATH}/hrnet-w48_8xb64-210e_ap10k-256x256.py",
    "${BASE_CONFIG_PATH}/res50_8xb64-210e_ap10k-256x256.py",
    "${BASE_CONFIG_PATH}/res101_8xb64-210e_ap10k-256x256.py",
    "${BASE_CONFIG_PATH}/res152_8xb32-210e_ap10k-256x256.py",
    "${BASE_CONFIG_PATH}/swin-t-p4-w7_8xb64-210e_ap10k-256x256.py",
    "${BASE_CONFIG_PATH}/swin-b-p4-w7_8xb64-210e_ap10k-256x256.py",
    "${BASE_CONFIG_PATH}/swin-l-p4-w7_8xb64-210e_ap10k-256x256.py"
)

# Execute FLOPs calculation for each configuration
foreach ($config in $configurations) {
    # Display current configuration being processed
    Write-Host "Processing configuration: $config"
    # Run the Python script with current configuration
    python tools/analysis_tools/get_flops.py $config --input-shape 256 256
}
