import argparse
import os
import subprocess

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train models with different configurations.')
    parser.add_argument('--dataset', type=str, default='ap10k', help='Dataset name')
    parser.add_argument('--scale', type=str, nargs='+', default=['s'], help='Scale of the model, can be s(mall), b(ase), l(arge), and h(uge).')
    parser.add_argument("--optical-field-size", type=int, default=None, help="Optical field size for the entire image.")
    parser.add_argument("--sub-optical-field-size", type=int, default=None, help="Optical field size for sub-regions of the image.")
    parser.add_argument("--window-size", nargs=2, type=int, default=None, help="Window size for sub-regions of the image.")
    parser.add_argument("--inverse", action="store_true", help="Order the images by their size before splitting into sub-regions.")
    parser.add_argument("--imgsz-hadamard", type=int, default=None, help="Image size for the Hadamard transform. If not provided, it will be set to imgsz.")
    parser.add_argument("--aliasing", action="store_true", help="Use aliasing for the Hadamard transform.")

    return parser.parse_args()

def find_latest_checkpoint(directory):
    checkpoints = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pth')]
    if checkpoints:
        return max(checkpoints, key=os.path.getmtime)
    else:
        return None

def main():
    args = parse_arguments()

    DATASET_NAME = args.dataset
    SCALE = args.scale
    OPTICAL_FIELD_SIZE = args.optical_field_size
    SUB_OPTICAL_FIELD_SIZE = args.sub_optical_field_size
    WINDOW_SIZE = args.window_size
    INVERSE = args.inverse
    IMGSZ_HADAMARD = args.imgsz_hadamard
    ALIASING = args.aliasing

    scale_configs = {
        "s": "spipose-small_8xb64-210e_ap10k-256x256.py",
        "b": "spipose-base_8xb64-210e_ap10k-256x256.py",
        "l": "spipose-large_8xb64-210e_ap10k-256x256.py",
        "h": "spipose-huge_8xb64-210e_ap10k-256x256.py"
    }
    configurations = []
    for scale in SCALE:
        if scale not in scale_configs:
            print(f"Invalid scale: {scale}")
            continue
        configurations.append(scale_configs[scale])
    
    if not configurations:
        print("No valid scale provided.")
        return

    original_dataset_dir = build_output_dir(
            f"data/{DATASET_NAME}/images",
            optical_field_sizes=OPTICAL_FIELD_SIZE,
            sub_optical_field_sizes=SUB_OPTICAL_FIELD_SIZE,
            window_size=WINDOW_SIZE,
            inverse=INVERSE,
            imgsz_hadamard=IMGSZ_HADAMARD,
            aliasing=ALIASING
        )
    _, split_original_dataset_dir = original_dataset_dir.split("-", 1)
    BASE_WORK_CONFIG_PATH = f"work_dirs/spipose/{DATASET_NAME}-{split_original_dataset_dir}"

    temp_dataset_dir = f"data/{DATASET_NAME}/images"

    rename_dataset_directory(original_dataset_dir, temp_dataset_dir)
    
    try:
        for config in configurations:
            config_name = os.path.basename(config).split('.')[0]
            work_dir = os.path.join(BASE_WORK_CONFIG_PATH, config_name)
            config_path = os.path.join(BASE_WORK_CONFIG_PATH, config)
            checkpoint_path = find_latest_checkpoint(work_dir)
            if checkpoint_path is None:
               print(f"No checkpoint found for {work_dir}")
               continue
            command = f"CUDA_VISIBLE_DEVICES=0 python tools/test.py  {config_path} {checkpoint_path}"
            print(f"Executing: {command}")
            subprocess.run(command, shell=True)
    finally:
        rename_dataset_directory(temp_dataset_dir, original_dataset_dir)

if __name__ == '__main__':
    main()