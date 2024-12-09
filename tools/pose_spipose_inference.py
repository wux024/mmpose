from mmpose.apis.inferencers import MMPoseInferencer
import argparse
import os
import subprocess

def build_output_dir(
    base_dir, 
    optical_field_sizes=None, 
    sub_optical_field_sizes=None, 
    window_size=None, 
    hadamard_seed=None, 
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
    
    if hadamard_seed is not None:
        base_dir += f"-{hadamard_seed}"
    
    return base_dir

def rename_dataset_directory(original_name, temp_name):
    """Rename the dataset directory."""
    if os.path.exists(original_name):
        os.rename(original_name, temp_name)
    else:
        print(f"Dataset directory {original_name} does not exist.")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train models with different configurations.')
    parser.add_argument('--source', type=str, default='None', help='Source directory of the images or video.')
    parser.add_argument('--dataset', type=str, default='ap10k', help='Dataset name')
    parser.add_argument('--scale', type=str, nargs='+', default=['s'], help='Scale of the model, can be s(mall), b(ase), l(arge), and h(uge).')
    parser.add_argument("--optical-field-size", type=int, default=None, help="Optical field size for the entire image.")
    parser.add_argument("--sub-optical-field-size", type=int, default=None, help="Optical field size for sub-regions of the image.")
    parser.add_argument("--window-size", nargs=2, type=int, default=None, help="Window size for sub-regions of the image.")
    parser.add_argument("--inverse", action="store_true", help="Order the images by their size before splitting into sub-regions.")
    parser.add_argument("--imgsz-hadamard", type=int, default=None, help="Image size for the Hadamard transform. If not provided, it will be set to imgsz.")
    parser.add_argument("--aliasing", action="store_true", help="Use aliasing for the Hadamard transform.")
    parser.add_argument("--hadamard-seed", type=int, default=None, help="Seed for the Hadamard transform.")
    
    # inference options
    parser.add_argument('--device', type=str, default='cuda:0', help='Device used for inference.')
    parser.add_argument('--show', action='store_true', help='Show visualizations.')
    parser.add_argument('--radius', type=int, default=4, help='Keypoint radius for visualization.')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness for visualization.')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint score threshold for visualization.')
    parser.add_argument('--draw-bbox', action='store_true', help='Draw bounding box for visualization.')
    parser.add_argument('--draw-heatmap', action='store_true', help='Draw heatmap for visualization.')
    parser.add_argument('--black-background', action='store_true', help='Use black background for visualization.')
    parser.add_argument('--return-vis', action='store_true', help='Return visualizations.')
    parser.add_argument('--vis-out-dir', type=str, default='None', help='Directory to save visualizations.')
    parser.add_argument('--return-datasamples', action='store_true', help='Return data samples.')
    parser.add_argument('--pred-out-dir', type=str, default='None', help='Directory to save predictions.')
    parser.add_argument('--out-dir', type=str, default='None', help='Directory to save output files.')

    return parser.parse_args()

def find_latest_checkpoint(directory):
    checkpoints = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pth')]
    if checkpoints:
        return max(checkpoints, key=os.path.getmtime)
    else:
        return None

def main():
    args = parse_arguments()

    SOURCE_DIR = args.source
    DATASET_NAME = args.dataset
    SCALE = args.scale
    OPTICAL_FIELD_SIZE = args.optical_field_size
    SUB_OPTICAL_FIELD_SIZE = args.sub_optical_field_size
    WINDOW_SIZE = args.window_size
    INVERSE = args.inverse
    IMGSZ_HADAMARD = args.imgsz_hadamard
    ALIASING = args.aliasing
    HADAMARD_SEED = args.hadamard_seed

    DEVICE = args.device
    SHOW = args.show
    RADIUS = args.radius
    THICKNESS = args.thickness
    KPT_THR = args.kpt_thr    
    DRAW_BBOX = args.draw_bbox
    DRAW_HEATMAP = args.draw_heatmap
    BLACK_BACKGROUND = args.black_background
    RETURN_VIS = args.return_vis
    RETURN_DATASAMPLES = args.return_datasamples

    scale_configs = {
        "s": f"spipose-small_8xb64-210e_{DATASET_NAME}-256x256.py",
        "b": f"spipose-base_8xb64-210e_{DATASET_NAME}-256x256.py",
        "l": f"spipose-large_8xb64-210e_{DATASET_NAME}-256x256.py",
        "h": f"spipose-huge_8xb64-210e_{DATASET_NAME}-256x256.py"
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
            aliasing=ALIASING,
            hadamard_seed=HADAMARD_SEED
        )
    _, split_original_dataset_dir = original_dataset_dir.split("-", 1)
    BASE_WORK_CONFIG_PATH = f"work_dirs/spipose/{DATASET_NAME}-{split_original_dataset_dir}"
    BASE_CONFIG_PATH = f"configs/animal_2d_keypoint/spipose/{DATASET_NAME}"

    temp_dataset_dir = f"data/{DATASET_NAME}/images"

    if SOURCE_DIR == 'None':
        SOURCE_DIR = os.path.join(temp_dataset_dir, "test")

    rename_dataset_directory(original_dataset_dir, temp_dataset_dir)
    
    try:
        for config in configurations:
            config_name = config.split('.')[0]
            work_dir = os.path.join(BASE_WORK_CONFIG_PATH, config_name)
            config_path = os.path.join(BASE_CONFIG_PATH, config)
            checkpoint_path = find_latest_checkpoint(work_dir)
            if checkpoint_path is None:
               print(f"No checkpoint found for {work_dir}")
               continue
            inferencer = MMPoseInferencer(
                pose2d = config_path,
                pose2d_weights = checkpoint_path,
                device = DEVICE
            )
            result_generator = inferencer(SOURCE_DIR,
                show = SHOW,
                radius = RADIUS,
                thickness = THICKNESS,
                kpt_thr = KPT_THR,
                draw_bbox = DRAW_BBOX,
                draw_heatmap = DRAW_HEATMAP,
                black_background = BLACK_BACKGROUND,
                return_vis = RETURN_VIS,
                vis_out_dir = os.path.join(work_dir, "visualizations"),
                return_datasamples = RETURN_DATASAMPLES,
                pred_out_dir = os.path.join(work_dir, "predictions"))
            for _ in result_generator:
                pass

    finally:
        rename_dataset_directory(temp_dataset_dir, original_dataset_dir)

if __name__ == '__main__':
    main()

