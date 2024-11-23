import argparse
import os
import subprocess


def prepare_configurations(mode, dataset_name):
    base_config_path = f"configs/animal_2d_keypoint/{mode}/{dataset_name}"

    configurations = []

    if mode == 'bottomup':
        print("Preparing for bottomup processing...")
        configurations.extend([
            # f"{base_config_path}/ae_hrnet-w32_8xb32-300e_{dataset_name}-512x512.py",
            f"{base_config_path}/cid_hrnet-w32_8xb32-140e_{dataset_name}-512x512.py",
            f"{base_config_path}/cid_hrnet-w48_8xb32-140e_{dataset_name}-512x512.py",
            f"{base_config_path}/dekr_hrnet-w32_8xb32-140e_{dataset_name}-512x512.py",
            f"{base_config_path}/dekr_hrnet-w48_8xb32-140e_{dataset_name}-640x640.py"
        ])
    elif mode == 'rtmo':
        print("Preparing for RTMO processing...")
        configurations.extend([
            f"{base_config_path}/{mode}-s_8xb16-600e_{dataset_name}-640x640.py",
            f"{base_config_path}/{mode}-m_16xb16-600e_{dataset_name}-640x640.py",
            f"{base_config_path}/{mode}-l_16xb16-600e_{dataset_name}-640x640.py"
        ])
    elif mode == 'rtmpose':
        print("Preparing for RTMPose processing...")
        configurations.extend([
            f"{base_config_path}/{mode}-t_8xb256-420e_{dataset_name}-256x256.py",
            f"{base_config_path}/{mode}-s_8xb256-420e_{dataset_name}-256x256.py",
            f"{base_config_path}/{mode}-m_8xb256-420e_{dataset_name}-256x256.py",
            f"{base_config_path}/{mode}-l_8xb256-420e_{dataset_name}-256x256.py"
        ])
    elif mode == 'topdown_heatmap':
        print("Preparing for topdown heatmap processing...")
        configurations.extend([
            f"{base_config_path}/td-hm_res50_8xb64-210e_{dataset_name}-256x256.py",
            f"{base_config_path}/td-hm_res101_8xb64-210e_{dataset_name}-256x256.py",
            f"{base_config_path}/td-hm_res152_8xb32-210e_{dataset_name}-256x256.py",
            f"{base_config_path}/td-hm_hrnet-w32_8xb64-210e_{dataset_name}-256x256.py",
            f"{base_config_path}/td-hm_hrnet-w48_8xb64-210e_{dataset_name}-256x256.py"
        ])
    elif mode == 'topdown_others':
        print("Preparing for topdown regression processing...")
        configurations.extend([
            f"{base_config_path}/ipr_res50_8xb64-210e_{dataset_name}-256x256.py",
            f"{base_config_path}/simcc_res50_8xb64-210e_{dataset_name}-256x256.py",
            f"{base_config_path}/td-reg_res50_8xb64-210e_{dataset_name}-256x256.py",
            f"{base_config_path}/td-reg_res50_rle-8xb64-210e_{dataset_name}-256x256.py",
            f"{base_config_path}/td-reg_res101_8xb64-210e_{dataset_name}-256x256.py",
            f"{base_config_path}/td-reg_res101_rle-8xb64-210e_{dataset_name}-256x256.py",
            f"{base_config_path}/td-reg_res152_8xb64-210e_{dataset_name}-256x256.py",
            f"{base_config_path}/td-reg_res152_rle-8xb64-210e_{dataset_name}-256x256.py"
        ])
    elif mode == 'yoloxpose':
        print("Preparing for YOLOX-Pose processing...")
        configurations.extend([
            f"{base_config_path}/{mode}_t_4xb64-300e_{dataset_name}-416.py",
            f"{base_config_path}/{mode}_s_8xb32-300e_{dataset_name}-640.py",
            f"{base_config_path}/{mode}_m_8xb32-300e_{dataset_name}-640.py",
            f"{base_config_path}/{mode}_l_8xb32-300e_{dataset_name}-640.py"
        ])
    elif mode == 'vitpose':
        print("Preparing for ViTPose processing...")
        configurations.extend([
            f"configs/animal_2d_keypoint/topdown_heatmap/{dataset_name}/td-hm_ViTPose-small-simple_8xb64-210e_{dataset_name}-256x256.py",
            f"configs/animal_2d_keypoint/topdown_heatmap/{dataset_name}/td-hm_ViTPose-base-simple_8xb64-210e_{dataset_name}-256x256.py",
            f"configs/animal_2d_keypoint/topdown_heatmap/{dataset_name}/td-hm_ViTPose-large-simple_8xb64-210e_{dataset_name}-256x256.py",
            f"configs/animal_2d_keypoint/topdown_heatmap/{dataset_name}/td-hm_ViTPose-huge-simple_8xb64-210e_{dataset_name}-256x256.py",
            f"configs/animal_2d_keypoint/topdown_heatmap/{dataset_name}/td-hm_ViTPose-small_8xb64-210e_{dataset_name}-256x256.py",
            f"configs/animal_2d_keypoint/topdown_heatmap/{dataset_name}/td-hm_ViTPose-base_8xb64-210e_{dataset_name}-256x256.py",
            f"configs/animal_2d_keypoint/topdown_heatmap/{dataset_name}/td-hm_ViTPose-large_8xb64-210e_{dataset_name}-256x256.py",
            f"configs/animal_2d_keypoint/topdown_heatmap/{dataset_name}/td-hm_ViTPose-huge_8xb64-210e_{dataset_name}-256x256.py"
        ])
    elif mode == 'animalvitpose':
        print("Preparing for AnimalViTPose processing...")
        configurations.extend([
            f"configs/animal_2d_keypoint/animalvitpose/{dataset_name}/animalvitpose-small_8xb64-210e_{dataset_name}-256x256.py",
            #f"configs/animal_2d_keypoint/animalvitpose/{dataset_name}/animalvitpose-base_8xb64-210e_{dataset_name}-256x256.py",
            #f"configs/animal_2d_keypoint/animalvitpose/{dataset_name}/animalvitpose-large_8xb64-210e_{dataset_name}-256x256.py",
            #f"configs/animal_2d_keypoint/animalvitpose/{dataset_name}/animalvitpose-huge_8xb64-210e_{dataset_name}-256x256.py"
        ])
    else:
        print(f"Unsupported mode: {mode}. Please choose a valid mode.")
        return None

    return configurations

def build_output_dir(
    base_dir, 
    optical_field_sizes=None, 
    sub_optical_field_sizes=None, 
    window_size=None, 
    seed=None, 
    inverse=False, 
    imgsz_hadamard=None,
    aliasing=False
):
    """Build the save directory based on the provided arguments."""
    base_dir = f"{base_dir}"
    
    if optical_field_sizes is not None:
        base_dir += f"-{optical_field_sizes}x{optical_field_sizes}"
    
    if sub_optical_field_sizes is not None:
        base_dir += f"-{sub_optical_field_sizes}x{sub_optical_field_sizes}"
    
    if window_size is not None:
        base_dir += f"-{window_size[0]}x{window_size[1]}"

    if inverse:
        base_dir += "-inverse"
    
    if aliasing:
        base_dir += "-aliasing"
    
    if imgsz_hadamard is not None:
        base_dir += f"-{imgsz_hadamard}"
    
    if seed is not None:
        base_dir += f"-{seed}"
    
    return base_dir

def rename_dataset_directory(original_name, temp_name):
    """Rename the dataset directory."""
    if os.path.exists(original_name):
        os.rename(original_name, temp_name)
    else:
        print(f"Dataset directory {original_name} does not exist.")

def main():
    parser = argparse.ArgumentParser(description='Train models with different configurations.')
    parser.add_argument('--dataset', type=str, default='ap10k', help='Dataset name')
    parser.add_argument('--gpu', type=str, default='0', help='GPU number to use')
    parser.add_argument('--mode', type=str, default='topdown_heatmap', choices=[
        'bottomup', 'rtmo', 'rtmpose', 'topdown_heatmap', 'topdown_regression', 'yoloxpose', 'vitpose', 'animalvitpose'
    ], help='Mode of operation')
    parser.add_argument("--optical-field-size", type=int, default=None, help="Optical field size for the entire image.")
    parser.add_argument("--sub-optical-field-size", type=int, default=None, help="Optical field size for sub-regions of the image.")
    parser.add_argument("--window-size", nargs=2, type=int, default=None, help="Window size for sub-regions of the image.")
    parser.add_argument("--inverse", action="store_true", help="Order the images by their size before splitting into sub-regions.")
    parser.add_argument("--imgsz-hadamard", type=int, default=None, help="Image size for the Hadamard transform. If not provided, it will be set to imgsz.")
    parser.add_argument("--aliasing", action="store_true", help="Use aliasing for the Hadamard transform.")

    args = parser.parse_args()

    DATASET_NAME = args.dataset
    GPU_NUM = args.gpu
    MODE = args.mode
    OPTICAL_FIELD_SIZE = args.optical_field_size
    SUB_OPTICAL_FIELD_SIZE = args.sub_optical_field_size
    WINDOW_SIZE = args.window_size
    INVERSE = args.inverse
    IMGSZ_HADAMARD = args.imgsz_hadamard
    ALIASING = args.aliasing

    original_dataset_dir = build_output_dir(
            f"data/{DATASET_NAME}/images",
            optical_field_sizes=OPTICAL_FIELD_SIZE,
            sub_optical_field_sizes=SUB_OPTICAL_FIELD_SIZE,
            window_size=WINDOW_SIZE,
            inverse=INVERSE,
            imgsz_hadamard=IMGSZ_HADAMARD,
            aliasing=ALIASING
        )
    _, split_original_dataset_dir = original_dataset_dir.split("-",1)
    BASE_WORK_CONFIG_PATH = f"work_dirs/{MODE}/{DATASET_NAME}-{split_original_dataset_dir}"
    temp_dataset_dir = f"data/{DATASET_NAME}/images"

    rename_dataset_directory(original_dataset_dir, temp_dataset_dir)

    CUDA_COMMAND = f"CUDA_VISIBLE_DEVICES={GPU_NUM}"

    configurations = prepare_configurations(MODE, DATASET_NAME)

    try:
        for config in configurations:
            config_name = os.path.basename(config).split(".")[0]
            work_dir = os.path.join(BASE_WORK_CONFIG_PATH, config_name)
            command = f"{CUDA_COMMAND} python tools/train.py {config} --work-dir {work_dir} --amp --auto-scale-lr"
            print(f"Executing: {command}")
        subprocess.run(command, shell=True)
    finally:
        rename_dataset_directory(temp_dataset_dir, original_dataset_dir)

if __name__ == '__main__':
    main()