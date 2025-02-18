# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

from typing import Iterator
import argparse
import glob
import os
import os.path as osp
import re
import shutil

from tqdm import tqdm


ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))


def gen_id_iter(input_file_or_dir: str) -> Iterator[tuple[int, int]]:
    '''Reads the list of selected material IDs and iterates over them in the
    format (case_id, gen_id) pairs.
    '''
    # If the input is a file, read the list of selected material IDs
    if osp.isfile(input_file_or_dir):
        with open(input_file_or_dir, 'r') as f:
            yield from (
                (case_id, gen_id)
                for case_id, line in enumerate(f)
                for gen_id in map(int, line.strip().split())
            )

    # Otherwise, iterate over all the material IDs in the dataset
    else:
        case_pattern = re.compile(r'case_(\d+)')
        gen_pattern = re.compile(r'gen_(\d+)_full\.py')

        for d in os.listdir(input_file_or_dir):
            case_dir = osp.join(input_file_or_dir, d)
            if osp.isdir(case_dir) and case_pattern.fullmatch(d):
                case_id = int(case_pattern.fullmatch(d).group(1))
                yield from (
                    (case_id, int(gen_pattern.fullmatch(fn).group(1)))
                    for fn in os.listdir(case_dir) if gen_pattern.fullmatch(fn)
                )


def main():
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Copy a subset of LLM-generated materials')
    parser.add_argument('-i', '--input_file', type=str, default=None,
                        help='File containing the list of files to copy')
    parser.add_argument('-s', '--source_dir', type=str, default=osp.join(ROOT_DIR, 'material_dataset_filtered_v2'),
                        help='Source dataset directory')
    parser.add_argument('-o', '--output_dir', type=str, default=osp.join(ROOT_DIR, 'material_dataset_filtered_v3'),
                        help='Output directory')

    args = parser.parse_args()

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate the number of materials
    if args.input_file:
        with open(args.input_file, 'r') as f:
            num_materials = sum(len(l.strip().split()) for l in f)
    else:
        case_pattern = re.compile(r'case_(\d+)')
        num_materials = 0
        for d in os.listdir(args.source_dir):
            case_dir = osp.join(args.source_dir, d)
            if osp.isdir(case_dir) and case_pattern.fullmatch(d):
                num_materials += len(glob.glob('gen_*_full.py', root_dir=case_dir))

    # Create the material iterator
    mat_iter = gen_id_iter(args.input_file or args.source_dir)

    # Rename dictionary
    rename_dict = {
        'full.py': 'blender_full.py',
        'render.jpg': 'transpiled_render.jpg',
        'analysis.json': 'analysis_result.json'
    }

    # Copy the file folders
    for case_id, gen_id in tqdm(mat_iter, total=num_materials):
        src_folder = osp.join(args.source_dir, f'case_{case_id:05d}')
        dst_folder = osp.join(args.output_dir, f'case_{case_id:05d}_gen_{gen_id:02d}')

        # Create the destination folder
        os.makedirs(dst_folder, exist_ok=True)

        # Copy files
        for fn in os.listdir(src_folder):
            if fn.startswith(f'gen_{gen_id:02d}_'):
                shutil.copy(osp.join(src_folder, fn), osp.join(dst_folder, rename_dict[fn[7:]]))


if __name__ == '__main__':
    main()
