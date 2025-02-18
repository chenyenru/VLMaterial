# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

from PIL import Image
from typing import Any
import argparse
import glob
import itertools
import json
import os
import os.path as osp
import re
import sys

sys.path.append(osp.dirname(osp.abspath(__file__)))

from lpips import LPIPS
from numpy.random import default_rng, Generator
from ssim import ssim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms.functional as TF

from utils import get_analysis_url, make_grid


ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
MAX_SEED = 0x7FFFFFFF

# Input text prompt
PROMPT = (
    'Write a Python function with Blender API to create a material node graph '
    'for this image.'
)


def scan_data(
        data_root: str, ignore_variations: bool = False, llm_filter: bool | None = None
    ) -> dict[str, list[str]]:
    '''Scan all available data and return a dictionary of source data.
    '''
    # Get all Blender source files
    all_files = glob.glob(osp.join("*", "*", "blender_full.py"), root_dir=data_root)
    if llm_filter is not None:
        all_files = [
            f for f in all_files
            if osp.basename(osp.dirname(osp.dirname(f))).startswith('mat_llm') == llm_filter
        ]
    all_files.sort()

    # Compile regex pattern for matching data
    pattern = re.compile(r'(var_\d{5})_full.py')

    # Search for data in each folder
    source_data_dict = {}

    for file_path in all_files:
        mat_name = osp.dirname(file_path)
        mat_dir = osp.join(data_root, mat_name)

        # Get all code files in the material directory
        code_files = [
            f for f in os.listdir(mat_dir)
            if f == 'blender_full.py' or pattern.match(f)
        ]

        # Use code files to find corresponding rendering images
        data_files = []

        for code_file in code_files:
            if code_file == 'blender_full.py':
                data_name = 'blender'
                render_file = 'transpiled_render.jpg'
            elif not ignore_variations:
                data_name = pattern.match(code_file).group(1)
                render_file = f'{data_name}_render.jpg'
            else:
                continue
            if osp.isfile(osp.join(mat_dir, render_file)):
                data_files.append(data_name)

        # Add to source data dictionary if data files are found
        if data_files:
            source_data_dict[mat_name] = data_files

    return source_data_dict


def get_node_type_coverage(
        material_names: list[str], data_root: str, info_dir: str
    ) -> dict[str, list[str]]:
    '''Given a list of materials, get how many graphs each node type appears in.
    '''
    node_type_coverage = {}

    # Update node type coverage for each material
    for mat_name in material_names:
        # Get blend file
        mat_dir = osp.join(data_root, mat_name)

        # Get analysis URL
        url = osp.join(mat_dir, 'analysis_result.json')
        if not osp.isfile(url):
            ref_dir = osp.join(osp.join(osp.join(ROOT_DIR, 'material_dataset')), mat_name)
            blend_file = osp.join(ref_dir, next(f for f in os.listdir(ref_dir) if f.endswith('.blend')))
            url = get_analysis_url(blend_file, info_dir)

        # Load analysis result
        with open(url, 'r') as f:
            analysis_result = json.load(f)

        # Get node types
        node_types = set()

        for node in analysis_result:
            node_type = node['type']
            if node_type == 'ShaderNodeGroup':
                node_type = osp.join('node_groups', node['group_type'])
            node_types.add(node_type)

        # Update node type coverage
        for node_type in sorted(node_types):
            node_type_coverage.setdefault(node_type, []).append(mat_name)

    return node_type_coverage


def split_materials(
        material_names: list[str], split_ratios: dict[str, float], protected_set: set[str],
        rng: Generator | None = None
    ) -> dict[str, list[str]]:
    '''Split materials into data splits.
    '''
    # Split material names into protected and unprotected sets
    protected_names = [name for name in material_names if name in protected_set]
    unprotected_names = [name for name in material_names if name not in protected_set]

    # Shuffle unprotected names
    rng = default_rng() if rng is None else rng
    rng.shuffle(unprotected_names)

    # Calculate split sizes and check for validity
    split_sizes = [int(ratio * len(material_names) + 0.5) for ratio in split_ratios.values()]
    split_sizes[0] = len(material_names) - sum(split_sizes[1:])

    if split_sizes[0] < 0:
        raise ValueError('Invalid split ratios - not enough data for training set')
    if split_sizes[0] < len(protected_names):
        raise ValueError('Invalid split ratios - not enough data for splits other than training set')

    # Sample from unprotected names
    splits = {n: [] for n in split_ratios}

    for i, (name, size) in enumerate(zip(splits, split_sizes)):
        if not i:
            splits[name].extend(protected_names)
            size -= len(protected_names)
        splits[name].extend(unprotected_names[:size])
        unprotected_names = unprotected_names[size:]

    return splits


class LLMDataset(Dataset):
    '''Dataset for LLM-generated data.
    '''
    def __init__(self, llm_data_names: list[str], data_root: str):
        self.llm_data_names = llm_data_names
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.llm_data_names)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        name = self.llm_data_names[index]
        img_path = osp.join(self.data_root, name, 'transpiled_render.jpg')
        img = TF.to_tensor(Image.open(img_path).convert('RGB'))
        return name, img


def filter_llm(
        llm_data_names: list[str], ref_data_names: list[str], data_root: str, lpips_threshold: float = 0.3,
        ssim_threshold: float = 0.7, batch_size: tuple[int, int] = 6
    ) -> tuple[list[str], list[str], list[str]]:
    '''Filter LLM-generated data based on LPIPS and SSIM.
    '''
    # Get all reference images
    ref_images = []
    for name in ref_data_names:
        ref_img = Image.open(osp.join(data_root, name, 'transpiled_render.jpg')).convert('RGB')
        ref_images.append(ref_img)

    # Convert the reference images to PyTorch tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ref_tensors = torch.stack([TF.to_tensor(img) for img in ref_images]).to(device)

    # Filter LLM-generated images by SSIM and LPIPS
    output_names, filtered_names, filtered_refs = [], [], []
    llm_dataset = LLMDataset(llm_data_names, data_root)
    llm_loader = DataLoader(llm_dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    lpips = LPIPS(net='alex', version='0.1').requires_grad_(False).to(device)

    for batch, llm_tensors in tqdm(llm_loader, desc='Filtering'):
        llm_tensors = llm_tensors.to(device)

        # Calculate SSIM scores
        llm_tensors_rep = llm_tensors.repeat_interleave(len(ref_tensors), dim=0)
        ref_tensors_rep = ref_tensors.repeat(len(batch), 1, 1, 1)
        scores = ssim(llm_tensors_rep, ref_tensors_rep, data_range=1.0, size_average=False)
        scores = scores.view(len(batch), -1)

        # Get the SSIM mask
        ssim_mask = scores >= ssim_threshold

        # For each image, calculate LPIPS scores for SSIM-masked images
        for batch_id, (name, mask) in enumerate(zip(batch, ssim_mask)):
            if not mask.any():
                output_names.append(name)
                continue

            # Calculate LPIPS scores
            lpips_scores = lpips(llm_tensors[None, batch_id], ref_tensors[mask], normalize=True)

            if lpips_scores.min() > lpips_threshold:
                output_names.append(name)
            else:
                filtered_names.append(name)
                filtered_refs.append(ref_data_names[lpips_scores.argmin().item()])

    return output_names, filtered_names, filtered_refs


def create_dataset_splits(
        material_splits: dict[str, list[str]], source_data_dict: dict[str, list[str]],
        data_root: str
    ) -> dict[str, list[dict[str, Any]]]:
    '''Compile dataset splits from material splits.
    '''
    splits = {n: [] for n in material_splits}

    # Compile each dataset split
    for name, materials in material_splits.items():
        meta_list = splits[name]

        # Iterate over each data file in each material
        for mat_name in materials:
            mat_dir = osp.join(data_root, mat_name)

            for data_name in source_data_dict[mat_name]:
                # Data ID
                data_id = f"{mat_name.replace(osp.sep, '-').replace(' ', '_')}-{data_name}"

                # Read code
                code_file = osp.join(mat_dir, f'{data_name}_full.py')
                with open(code_file, 'r') as f:
                    code = f.read()

                # Get rendering path
                if data_name == 'blender':
                    render_file = osp.join(mat_name, 'transpiled_render.jpg')
                else:
                    render_file = osp.join(mat_name, f'{data_name}_render.jpg')

                # Add metadata
                meta_list.append({
                    'id': data_id,
                    'image': render_file,
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<image>\n{PROMPT}'
                        },
                        {
                            'from': 'gpt',
                            'value': f'```python\n{code}```'
                        }
                    ]
                })

    return splits


def main():
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Create LLaVA training and validation dataset')
    parser.add_argument('--data_root', type=str, default=osp.join(ROOT_DIR, 'material_dataset_filtered'),
                        help='Root directory of Blender files')
    parser.add_argument('--info_dir', type=str, default=osp.join(ROOT_DIR, 'material_dataset_info'),
                        help='Directory to analysis results')
    parser.add_argument('-o', '--output_folder', type=str, default=None, help='Output directory')

    parser.add_argument('--split_names', type=str, nargs=3, default=['train', 'val', 'test'],
                        help='Names of dataset splits')
    parser.add_argument('--split_ratios', type=float, nargs=3, default=[0.9, 0.05, 0.05],
                        help='Ratios of dataset splits')
    parser.add_argument('--read_splits', type=str, default=None, help='Path to the directory containing existing dataset splits')
    parser.add_argument('--read_prefix', type=str, default=None, help='Prefix for splits to read')
    parser.add_argument('--save_prefix', type=str, default='llava_', help='Prefix for saved files')
    parser.add_argument('--save_suffix', type=str, default='', help='Suffix for saved files')
    parser.add_argument('--ignore_variations', action='store_true', help='Ignore variations when splitting')
    parser.add_argument('--ignore_node_types', action='store_true', help='Ignore node types when splitting')

    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-a', '--add_llm', action='store_true', help='Add LLM-generated materials to training data')
    parser.add_argument('-l', '--filter_llm', action='store_true', help='Filter LLM-generated data')
    parser.add_argument('--lpips_threshold', type=float, default=0.3, help='LPIPS threshold for filtering LLM-generated data')
    parser.add_argument('--ssim_threshold', type=float, default=0.8, help='SSIM threshold for filtering LLM-generated data')
    parser.add_argument('--show_filtered', action='store_true', help='Show the filtered images')

    args = parser.parse_args()

    data_root = args.data_root
    info_dir = args.info_dir

    # Check split ratios
    if any(ratio < 0 for ratio in args.split_ratios):
        raise ValueError('Split ratios must be non-negative')
    if abs(sum(args.split_ratios) - 1) > 1e-8:
        raise ValueError('Sum of split ratios must be 1')

    # Set random seed
    rng = default_rng(args.seed)

    # Scan all available data
    source_data_dict = scan_data(
        data_root, ignore_variations=args.ignore_variations,
        llm_filter=None if args.read_splits else False
    )
    print(f'Found {len(source_data_dict)} materials')

    # Read existing dataset splits
    if args.read_splits:
        material_splits: dict[str, set[str]] = {}
        for key in args.split_names:
            split_file_name = f'{args.read_prefix or args.save_prefix}{key}{args.save_suffix}.json'
            with open(osp.join(args.read_splits, split_file_name), 'r') as f:
                material_splits[key] = {osp.dirname(d['image']) for d in json.load(f)}

        # Check the examples not in the splits
        uncategorized_names = [
            name for name in source_data_dict
            if all(name not in material_splits[k] for k in args.split_names)
        ]
        if uncategorized_names:
            print(f'Found {len(uncategorized_names)} source materials not present in the splits')

        material_splits = {k: sorted(v) for k, v in material_splits.items()}

    else:
        # Construct a protected set of materials to ensure all splits other than the training
        # set do not reference nodes not seen in the training set
        material_names = list(source_data_dict.keys())
        protected_set = set()

        if not args.ignore_node_types:
            node_type_coverage = get_node_type_coverage(material_names, data_root, info_dir)
            ps_rng = default_rng(rng.integers(0, MAX_SEED))
            for mats in node_type_coverage.values():
                protected_set.add(ps_rng.choice(mats))

        # Split materials into training and validation sets
        material_splits = split_materials(
            material_names, dict(zip(args.split_names, args.split_ratios)),
            protected_set, rng
        )
        material_splits = {k: sorted(v) for k, v in material_splits.items()}

    # Add LLM-generated data to training set
    filtered_names, filtered_refs = [], []

    if args.add_llm:
        llm_data_dict = scan_data(data_root, ignore_variations=args.ignore_variations, llm_filter=True)
        llm_data_names = sorted(llm_data_dict.keys())
        num_llm_data = len(llm_data_names)
        print(f'Found {num_llm_data} LLM-generated materials')

        # Filter LLM-generated data
        if args.filter_llm:
            ref_data_names = list(itertools.chain(*(material_splits[k] for k in args.split_names[1:])))
            llm_data_names, filtered_names, filtered_refs = filter_llm(
                llm_data_names, ref_data_names, data_root,
                lpips_threshold=args.lpips_threshold, ssim_threshold=args.ssim_threshold
            )
            print(f'Added {len(llm_data_names)}/{num_llm_data} LLM-generated data')
        else:
            print(f'Added {num_llm_data} LLM-generated data')

        material_splits['train'].extend(llm_data_names)
        source_data_dict.update(llm_data_dict)

    # Compile dataset splits
    dataset_splits = create_dataset_splits(material_splits, source_data_dict, data_root)

    # Save dataset splits
    output_folder = args.output_folder or data_root
    os.makedirs(output_folder, exist_ok=True)

    for name, data in dataset_splits.items():
        if data:
            save_path = osp.join(output_folder, f'{args.save_prefix}{name}{args.save_suffix}.json')
            with open(save_path, 'w') as f:
                json.dump(data, f)

    # Show filtered images
    if args.show_filtered and filtered_names:
        filtered_imgs = []

        # Read filtered images
        for name, ref_name in zip(filtered_names[:100], filtered_refs[:100]):
            imgs = [
                Image.open(osp.join(data_root, n, 'transpiled_render.jpg')).convert('RGB')
                for n in (name, ref_name)
            ]
            filtered_imgs.append(np.concatenate([np.asarray(img.resize((256, 256))) for img in imgs], axis=1))

        # Make a grid of the filtered images
        num_cols = 10
        grid = make_grid([filtered_imgs[i:i + num_cols] for i in range(0, len(filtered_imgs), num_cols)])
        Image.fromarray(grid).save(osp.join(output_folder, 'filtered_images.png'))

    # Print summary
    print('Dataset splits:')
    for name, data in dataset_splits.items():
        print(f'  - {name}: {len(material_splits[name])} graphs, {len(data)} samples')


if __name__ == '__main__':
    main()
