# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

from collections import Counter
from typing import Iterator, Any
import glob
import itertools
import json
import os
import os.path as osp

from tqdm import tqdm
import numpy as np

from utils import NodeSignatureReader, get_analysis_url, FLOAT_MAX


def gen_default_param_stats(param_sig: dict[str, Any], category: str) -> dict[str, Any]:
    '''Create the default parameter statistics for a given parameter.
    '''
    # Basic parameter information
    param_name, param_type = param_sig['name'], param_sig['type']
    stats = {
        'name': param_name,
        'type': param_type,
        'is_attr': category == 'attr'
    }

    # Helper function for processing individual parameters
    def stats_helper(
            name: str, dtype: str, default: Any,
            default_range: tuple[float, float] = None
        ) -> dict[str, Any]:
        ret = {
            'name': name,
            'type': dtype,
            'default': default,
            'default_range': default_range,
            'default_count': 0,
            'outlier_count': 0,
            'obs': []
        }
        if dtype not in ('float', 'int', 'str', 'vec_arr', 'color_arr'):
            del ret['default_range']
        return ret

    # Input/output parameters
    if category in ('input', 'output'):

        # Get default value
        default_val = param_sig.get('default')
        stats['default'] = default_val

        if default_val is not None:
            # Get default value range
            if param_type != 'String':
                default_range = (
                    [0.0, 1.0] if param_type.startswith('Color')
                    else [-FLOAT_MAX, FLOAT_MAX]
                )
                for i, key in enumerate(('min', 'max')):
                    if key in param_sig:
                        default_range[i] = param_sig[key]
                stats['default_range'] = default_range

            # Initialize observed values
            stats.update({'default_count': 0, 'outlier_count': 0, 'obs': []})

    # Attributes
    elif category == 'attr':
        default_val = param_sig['value']

        # Color ramps
        if param_type == 'ColorRamp':
            stats['group'] = [
                stats_helper(
                    'mode', 'str', default_val['mode'],
                    default_range=['RGB', 'HSV', 'HSL']
                ),
                stats_helper(
                    'interp', 'str', default_val['interp'],
                    default_range=['EASE', 'CARDINAL', 'LINEAR', 'B_SPLINE', 'CONSTANT']
                ),
                stats_helper(
                    'hue_interp', 'str', default_val['hue_interp'],
                    default_range=['NEAR', 'FAR', 'CW', 'CCW']
                ),
                stats_helper('elements', 'color_arr', default_val['elements'], [0.0, 1.0])
            ]

        # Curves
        elif param_type == 'CurveMapping':
            stats['group'] = [
                stats_helper('use_clip', 'bool', default_val['use_clip']),
                *(
                    stats_helper(
                        f'curves_{i}', 'vec_arr', curve,
                        default_range=[curve[0][0], curve[-1][0]]
                    ) for i, curve in enumerate(default_val['curves'])
                )
            ]

        else:
            if 'enum' in param_sig:
                default_range = param_sig['enum']
            elif 'min' in param_sig or 'max' in param_sig:
                default_range = [
                    param_sig.get(k, v)
                    for k, v in {'min': -FLOAT_MAX, 'max': FLOAT_MAX}.items()
                ]
            else:
                default_range = None
            stats.update(stats_helper(param_name, param_type, default_val, default_range))

    else:
        raise ValueError(f'Parameter from an unknown category {category}')

    return stats


def add_param_observations(
        param_stats: dict[str, list[dict[str, Any]]], node_info: dict[str, Any],
        node_sig: dict[str, Any],
    ):
    '''Collect parameter observations from a material node.
    '''
    # Get node type signature key
    node_type = node_info['type']
    if node_type != 'ShaderNodeGroup':
        sig_key = node_type
    else:
        sig_key = osp.join('node_groups', node_info['group_type'])

    # Organize parameter info into dictionary
    param_info_dict = {p['id']: p for p in node_info['params']}
    linked_ids = set(node_info['linked_input'])

    # Get or initialize node parameter statistics
    categories = ['output' if node_type in ('ShaderNodeValue', 'ShaderNodeRGB') else 'input']
    if 'attr' in node_sig:
        categories.append('attr')
    num_params = sum(len(node_sig[cat]) for cat in categories)
    node_param_stats = param_stats.setdefault(sig_key, [None] * num_params)

    # Helper function for adding a parameter observation
    def add_obs(param: dict[str, Any], value: Any):
        # Record observed value
        if value == param['default']:
            param['default_count'] += 1
        else:
            param['obs'].append(value)

        # Check for outliers
        default_range = param.get('default_range')

        if isinstance(default_range, list):
            if isinstance(default_range[0], (float, int)):
                default_min, default_max = default_range
                val_arr = np.array(value)
                param['outlier_count'] += int(
                    np.any(val_arr < default_min)
                    or np.any(val_arr > default_max)
                )
            elif isinstance(default_range[0], str):
                param['outlier_count'] += value not in default_range

    # Update parameter statistics for each parameter
    param_id = 0

    for cat in categories:
        for param_sig in node_sig[cat]:

            # Add default statistics for new parameters
            if node_param_stats[param_id] is None:
                node_param_stats[param_id] = gen_default_param_stats(param_sig, cat)
            stats = node_param_stats[param_id]

            # Skip linked input parameters or parameters without default values
            if param_id in linked_ids or (cat != 'attr' and stats.get('default') is None):
                param_id += 1
                continue

            # Register observed value
            if param_id in param_info_dict:
                param_info = param_info_dict[param_id]
                param_val = param_info['value']

                # Color ramp or curve mapping
                if 'group' in stats:
                    param_group = stats['group']
                    find_param = lambda name: next(p for p in param_group if p['name'] == name)

                    for k, v in param_val.items():
                        if k == 'curves':
                            for i, curve in enumerate(v):
                                add_obs(find_param(f'curves_{i}'), curve)
                        else:
                            add_obs(find_param(k), v)

                # Other parameters
                else:
                    add_obs(stats, param_val)

            # Register default value
            elif 'group' in stats:
                for param in stats['group']:
                    param['default_count'] += 1
            else:
                stats['default_count'] += 1

            param_id += 1


def update_param_stats(param_stats: dict[str, list[dict[str, Any]]]):
    '''Calculate the statistics for each parameter.
    '''
    # Parameter statistics iterator
    def param_stats_iter() -> Iterator[dict[str, Any]]:
        for stats in itertools.chain(*param_stats.values()):
            if 'group' in stats:
                yield from stats['group']
            else:
                yield stats

    # Calculate statistics for each parameter
    for stats in param_stats_iter():
        param_type, default_val = stats['type'], stats['default']

        # Skip parameters without default values
        if default_val is None:
            continue

        # Get all observations
        obs = [default_val] * stats['default_count']
        obs += stats.pop('obs', [])

        if not obs:
            stats['default_count'] = 1
            obs = [default_val]

        # Ranged parameters
        if (param_type in ('float', 'int', 'Color')
            or any(param_type.startswith(t) for t in ('Float', 'Vector'))):

            obs_arr = np.array(obs)
            stats.update({
                'count': len(obs),
                'range': [obs_arr.min().tolist(), obs_arr.max().tolist()],
                'mean': obs_arr.mean(axis=0).tolist(),
                'std': obs_arr.std(axis=0).tolist()
            })

        # Categorical parameters
        elif param_type in ('str', 'bool', 'String'):
            stats.update({
                'count': len(obs),
                'freq': dict(Counter(obs))
            })

        # Array parameters
        elif param_type in ('vec_arr', 'color_arr'):

            # Organize observations into a diectionary using array length as key
            obs_arr = [np.array(arr) for arr in obs]
            arr_lens = sorted(set(len(arr) for arr in obs_arr))
            obs_dict = {l: [] for l in arr_lens}
            for arr in obs:
                obs_dict[len(arr)].append(arr)

            # Calculate statistics for each array length
            stats.update({
                'count': {l: len(o) for l, o in obs_dict.items()},
                'range': [
                    min(a.min().tolist() for a in obs_arr),
                    max(a.max().tolist() for a in obs_arr)
                ],
                'mean': {l: np.mean(o, axis=0).tolist() for l, o in obs_dict.items()},
                'std': {l: np.std(o, axis=0).tolist() for l, o in obs_dict.items()},
            })

        else:
            raise ValueError(f"Parameter with unknown type '{param_type}'")


def collect_param_stats(data_root: str, info_dir: str, output_path: str):
    # Initialize parameter statistics dictionary
    param_stats: dict[str, dict[str, Any]] = {}

    # Create a node type signature reader
    node_type_dir = osp.join(info_dir, 'node_types')
    sig_reader = NodeSignatureReader(node_type_dir)

    # Process all (filtered) Blender source files
    all_files = glob.glob(osp.join("*", "*", "blender_full.py"), root_dir=data_root)

    for file_path in tqdm(all_files):

        # Get the analysis result path
        source_dir = osp.join(data_root, osp.dirname(file_path))
        analysis_path = osp.join(source_dir, 'analysis_result.json')

        if not osp.isfile(analysis_path):
            blender_file = osp.join(
                source_dir, next((f for f in os.listdir(source_dir) if f.endswith('.blend')), None)
            )
            if blender_file:
                analysis_path = get_analysis_url(osp.join(source_dir, blender_file), info_dir)
            else:
                raise FileNotFoundError(f"No blender file exists for '{file_path}'")

        # Read analysis results
        with open(analysis_path, 'r') as f:
            analysis: list[dict[str, Any]] = json.load(f)

        # Collect parameter observations for each node
        for node_info in analysis:

            # Read node type signature
            node_type = node_info['type']
            node_sig = sig_reader.read(node_type, node_info.get('group_type'))

            add_param_observations(param_stats, node_info, node_sig)

    # Calculate statistics for each parameter
    update_param_stats(param_stats)

    # Sort parameter statistics by node type
    param_stats = dict(sorted(param_stats.items(), key=lambda x: x[0]))

    # Save parameter statistics
    with open(output_path, 'w') as f:
        json.dump(param_stats, f, indent=2)
