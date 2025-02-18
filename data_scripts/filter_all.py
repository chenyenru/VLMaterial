# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

from PIL import Image
import argparse
import glob
import os
import os.path as osp
import shutil
import sys

THIS_DIR = osp.dirname(osp.abspath(__file__))
sys.path.append(THIS_DIR)

from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np

from utils import check_file, get_analysis_url


ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))


def main():
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Transpile, render, and filter all blender files in a directory')
    parser.add_argument('--blender_path', type=str,
                        default=osp.join(ROOT_DIR, 'infinigen', 'blender', 'blender'),
                        help='Path to Blender executable')
    parser.add_argument('--data_root', type=str, default=osp.join(ROOT_DIR, 'material_dataset'),
                        help='Root directory of Blender files')
    parser.add_argument('--info_dir', type=str, default=osp.join(ROOT_DIR, 'material_dataset_info'),
                        help='Directory to analysis results')
    parser.add_argument('--output_folder', type=str, default=osp.join(ROOT_DIR, 'material_dataset_filtered'),
                        help='Output directory')
    parser.add_argument('--tokenizer', type=str, default='llava-hf/llama3-llava-next-8b-hf',
                        help='Tokenizer URL to use')
    parser.add_argument('--min_file_size', type=int, default=12000,
                        help='Minimum rendered image file size in bytes')
    parser.add_argument('--dup_pixel_threshold', type=float, default=0.02,
                        help='Pixel value difference threshold for duplicate image detection')
    parser.add_argument('--dup_image_threshold', type=float, default=0.95,
                        help='Pixel percentage threshold for duplicate image detection')
    args = parser.parse_args()

    blender_path = args.blender_path
    data_root = args.data_root
    info_dir = args.info_dir
    output_folder = args.output_folder

    # Get all Blender source files
    all_files = glob.glob(osp.join("*", "*", "*.blend"), root_dir=data_root)

    # Process each file
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    conv_version = 'v1.5-hf' if args.tokenizer.endswith('-hf') else 'v1.5'

    rendered_images, code_lengths = [], []
    success_files, failed_files = [], []

    for file_path in tqdm(all_files):
        # Set target paths
        target_folder = osp.join(output_folder, osp.dirname(file_path))
        os.makedirs(target_folder, exist_ok=True)

        # Check code length and render quality
        result, reason = check_file(
            blender_path, osp.join(data_root, file_path), target_folder, info_dir, tokenizer,
            target_size=args.min_file_size, conv_version=conv_version,
            transpile_script_path=osp.join(THIS_DIR, 'transpile.py'),
            render_script_path=osp.join(THIS_DIR, 'render.py'),
        )

        # Skip if basic tests already failed
        if not result:
            failed_files.append((file_path, reason))
            print(f"Failed: {file_path} ({reason})")
            shutil.rmtree(target_folder)
            continue

        # Get code length and rendered image
        code_path = osp.join(target_folder, 'blender_full.py')
        render_path = osp.join(target_folder, 'transpiled_render.jpg')

        with open(code_path, 'r') as f:
            code_length = len(tokenizer.encode(f.read().strip()))
        img = np.array(Image.open(render_path), dtype=np.float32) / 255

        # Check duplicate renderings
        if rendered_images:

            # Compare the rendered image with previous results
            diff_ratio = [
                (np.abs(prev_img - img) <= args.dup_pixel_threshold).all(axis=-1).mean()
                for prev_img in rendered_images
            ]
            dup_ind = np.argmax(diff_ratio)

            # If the image has duplicates, compare code lengths and remove the longer one
            if diff_ratio[dup_ind] > args.dup_image_threshold:
                remove_path = (
                    success_files[dup_ind] if code_length < code_lengths[dup_ind]
                    else file_path
                )
                reason = 'Duplicate rendering with longer code'
                failed_files.append((remove_path, reason))
                print(f"Failed: {remove_path} ({reason})")
                shutil.rmtree(osp.join(output_folder, osp.dirname(remove_path)))

                # Remove the previous recorded result if the current code is shorter
                if code_length < code_lengths[dup_ind]:
                    success_files.pop(dup_ind)
                    code_lengths.pop(dup_ind)
                    rendered_images.pop(dup_ind)

        # Keep the case
        success_files.append(file_path)
        code_lengths.append(code_length)
        rendered_images.append(img)

        # Copy the analysis result
        analysis_path = get_analysis_url(file_path, info_dir=info_dir)
        shutil.copy(analysis_path, osp.join(target_folder, 'analysis_result.json'))

    print(f"Success {len(success_files)}/{len(all_files)}")

    # Log failed files as CSV
    with open(osp.join(output_folder, 'failed_files_infinigen.csv'), 'w') as f:
        f.write('file_path,reason\n')
        for file_path, reason in failed_files:
            f.write(f'{file_path},{reason}\n')


if __name__ == '__main__':
    main()
