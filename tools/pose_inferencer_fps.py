#! /usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from glob import glob
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Pose Estimation Inference')
    parser.add_argument('--dataset', type=str, default='ap10k', help='dataset name')
    parser.add_argument('--mode', type=str, default='rtmpose', help='inference model')

    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    mode = args.mode

    base_dir = os.path.join('work_dirs', mode, dataset)
    data_dir = os.path.join('data', dataset, 'images/test')
    models = os.listdir(base_dir)
    for model in models:
        pth_files = glob(os.path.join(f"{base_dir}/{model}", "*.pth"))[0]
        cmd = ['python', 'demo/inferencer_demo.py',
               '--input', f"{data_dir}",
               '--pose2d', f"{base_dir}/{model}/{model}.py",
               '--pose2d-weights', f"{base_dir}/{model}/{pth_files}",
               ]
        subprocess.call(cmd)