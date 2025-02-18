# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

import argparse
import glob
import os
import os.path as osp
import subprocess
import sys

THIS_DIR = osp.dirname(osp.abspath(__file__))
sys.path.append(THIS_DIR)

from tqdm import tqdm

from utils import get_analysis_url


ROOT_DIR = osp.dirname(THIS_DIR)


def main():
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Transpile, render, and filter all blender files in a directory')
    parser.add_argument('--blender_path', type=str, default=f'{ROOT_DIR}/infinigen/blender/blender')
    parser.add_argument('--data_root', type=str, default=f'{ROOT_DIR}/material_dataset')
    parser.add_argument('--output_folder', type=str, default=f'{ROOT_DIR}/material_dataset_info')
    args = parser.parse_args()

    blender_path = args.blender_path
    data_root = args.data_root
    output_folder = args.output_folder

    # Get all Blender source files
    all_files = glob.glob(osp.join(data_root, "*", "*", "*.blend"))

    # Process each file
    for file_path in tqdm(all_files):
        # Set target paths
        target_folder = osp.join(output_folder, osp.relpath(osp.dirname(osp.dirname(file_path)), data_root))
        os.makedirs(target_folder, exist_ok=True)

        # Check file
        analysis_path = get_analysis_url(file_path, info_dir=output_folder)
        ret = subprocess.run([
            blender_path, file_path, '-b', '-P', osp.join(THIS_DIR, 'analyze.py'),
            '--', analysis_path, '--info_dir', output_folder
        ], capture_output=True)

        # Report error
        stdout = ret.stdout.decode()
        if ('Error: Python:' in stdout
            and 'RuntimeError: Unsupported node type' not in stdout
            and 'RuntimeError: Oversized node graph' not in stdout):
            print(f"Error processing {file_path}: {stdout}")

if __name__ == '__main__':
    main()
