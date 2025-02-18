# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

from typing import Any
import argparse
import json
import os
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.abspath(__file__)))

from numpy.random import Generator
import bpy.types as T
import numpy as np
import numpy.typing as npt

from curation import dfs_from_node, find_output_node
from render import setup_plane, apply_material_code
from transpile import Transpiler
from utils import get_analysis_url, FLOAT_MAX

ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))


def is_sampling_allowed(node_type: str, param_name: str) -> bool:
    '''Check if sampling is allowed for the given parameter.
    '''
    # Skip the operator parameter in math nodes
    if node_type in ('ShaderNodeMath', 'ShaderNodeVectorMath') and param_name == 'operation':
        return False

    return True


def is_confident_sampling_allowed(node_type: str, param_name: str) -> bool:
    '''Check if confident sampling is allowed for the given parameter.
    '''
    # Only apply random perturbations to the operands in math nodes
    if node_type in ('ShaderNodeMath', 'ShaderNodeVectorMath'):
        return False

    return True


def perturb_color_hsv(
        rng: Generator, color: npt.NDArray[np.floating], perturb_range: float
    ) -> npt.NDArray[np.floating]:
    '''Perturb a RGB color in HSV space.
    '''
    # Separate alpha channel
    alpha = None
    if color.shape[-1] == 4:
        color, alpha = np.split(color, [3], axis=-1)
    elif color.shape[-1] != 3:
        raise ValueError(f'Color must have 3 or 4 channels, got {color.shape[-1]}')

    # Convert color from RGB to HSV space
    color = np.clip(color, 0, 1)
    r, g, b = np.rollaxis(color, axis=-1)
    mx, mn = np.max(color, axis=-1), np.min(color, axis=-1)
    dt = mx - mn
    dt_div = np.maximum(dt, 1e-12)

    ## Compute hue
    h = np.select(
        [mx == r, mx == g, mx == b],
        [(g - b) / dt_div, 2.0 + (b - r) / dt_div, 4.0 + (r - g) / dt_div]
    )
    h = np.where(mx == mn, 0.0, h)
    h = (h / 6.0) % 1.0

    ## Compute saturation
    s = np.where(mx == 0.0, 0.0, dt / np.maximum(mx, 1e-12))

    ## Compute value
    v = mx

    # Perturb the color in HSV space
    pr = perturb_range
    h = rng.uniform(size=h.shape)
    s = rng.uniform(np.maximum(s - pr, 0), np.minimum(s + pr, 1))
    v = rng.uniform(np.maximum(v - pr, 0), np.minimum(v + pr, 1))

    # Convert color back to RGB space
    c = s * v
    x = c * (1 - np.abs((h * 6) % 2 - 1))
    m = v - c
    z = np.zeros_like(c)
    rgb = np.select(
        [h < 1 / 6, h < 2 / 6, h < 3 / 6, h < 4 / 6, h < 5 / 6],
        [(c, x, z), (x, c, z), (z, c, x), (z, x, c), (x, z, c)],
        (c, z, x)
    )
    rgb = np.clip(rgb + m, 0, 1).T

    # Combine color with alpha channel
    if alpha is not None:
        alpha = rng.uniform(np.maximum(alpha - pr, 0), np.minimum(alpha + pr, 1))
        color_new = np.concatenate([rgb, alpha], axis=-1)
    else:
        color_new = rgb

    return color_new


def sample_node_params(
        node: T.ShaderNode, node_info: dict[str, Any], node_param_stats: list[dict[str, Any]],
        sample: bool = True, rng: Generator | None = None, global_param_id: int = 0,
        source_global_ids: list[int] | None = None, confident_sample: bool = True, confident_freq: int = 20,
        confident_std_threshold: float = 10.0, constant_default_freq: int = 50,
        default_perturb_prob: float = 0.25, perturb_prob: float = 1.0,
        uniform_perturb_ratio: float = 0.25, uniform_color_perturb_ratio: float = 0.5,
        min_uniform_perturb_amount: float = 0.05
    ) -> tuple[int, list[int]]:
    '''Sample parameters for the given node. Return the number of parameters processed and sampled
    parameter (global) IDs.
    '''
    # Set up random number generator
    rng = rng or np.random.default_rng()

    # Organize parameter info into dictionary
    param_info_dict = {p['id']: p for p in node_info['params']}
    linked_ids = set(node_info['linked_input'])

    # Get node input/output slots
    if node_info['type'] in ('ShaderNodeValue', 'ShaderNodeRGB'):
        slots = node.outputs
    else:
        slots = node.inputs

    # Helper function for sampling a particular parameter
    # Returns the number of parameters processed and the number of parameters sampled
    def process_param(
            param_id: int, global_param_id: int, cur_val: Any, stats: dict[str, Any],
            sample: bool = True, is_attr: bool | None = None, src_obj: Any = None,
            trans_name_dict: dict[str, str] = {}
        ) -> tuple[int, list[int]]:
        # Read basic information
        param_name, param_type = stats['name'], stats['type']
        is_attr = is_attr if is_attr is not None else stats['is_attr']

        # Grouped parameters
        if 'group' in stats:
            if not is_attr or src_obj is None:
                raise RuntimeError('Grouped parameters must be node attributes')
            param_obj = getattr(src_obj, param_name)

            # Back-translate the group field names
            if param_type == 'ColorRamp':
                param_name_dict = {
                    'mode': 'color_mode',
                    'interp': 'interpolation',
                    'hue_interp': 'hue_interpolation'
                }
            elif param_type == 'CurveMapping':
                param_name_dict = {}
            else:
                raise ValueError(f'Unknown grouped parameter type: {param_type}')

            # Initialize return values
            ret_val = 0, []

            # Process each parameter field in the group
            for field_stats in stats['group']:
                field_name = field_stats['name']

                # Curve parameter fields
                if field_name.startswith('curves_'):
                    curve_idx = int(field_name.split('_')[-1])
                    field_val = (
                        cur_val['curves'][curve_idx] if cur_val is not None
                        else field_stats['default']
                    )
                    field_src_obj = param_obj.curves[curve_idx]

                # Regular parameter fields
                else:
                    field_val = (
                        cur_val[field_name] if cur_val is not None
                        else field_stats['default']
                    )
                    field_src_obj = param_obj

                # Sample the parameter field
                param_ret_val = process_param(
                    param_id, global_param_id + ret_val[0], field_val, field_stats,
                    sample=sample, is_attr=True, src_obj=field_src_obj,
                    trans_name_dict=param_name_dict
                )

                # Update return values
                ret_val = tuple(ret_val[i] + param_ret_val[i] for i in range(2))

            return ret_val

        # Without sampling, revert to the initial value (provided by analysis)
        val = cur_val
        if val is None:
            raise TypeError(f"Parameter '{param_name}' has no current value")

        # Initialize return values
        ret_val = 1, []

        # Sample parameter value
        if sample and is_sampling_allowed(node.bl_idname, param_name):
            default_val = stats['default']

            # Skip if the parameter is not in the source global IDs (i.e., not allowed for sampling)
            if source_global_ids is not None and global_param_id not in source_global_ids:
                return ret_val

            # Skip unused or unavailable parameters
            default_range = stats.get('default_range')
            if isinstance(default_range, list) and default_range[0] == default_range[1]:
                return ret_val

            # Skip parameters that always have the default value
            count, default_count = stats['count'], stats['default_count']
            if isinstance(count, dict):
                count = count.get(str(len(default_val)), 0)
            if count == default_count and count >= constant_default_freq:
                return ret_val

            # For parameters at default values, perturb with a certain probability
            # perturb_prob = max(min_perturb_prob, (count - default_count) / count)
            if (cur_val == default_val and default_perturb_prob < 1.0 and rng.random() > default_perturb_prob
                or cur_val != default_val and perturb_prob < 1.0 and rng.random() > perturb_prob):
                return ret_val

            # Confident sampling flag
            confident_flag = (
                confident_sample and count >= confident_freq
                and is_confident_sampling_allowed(node.bl_idname, param_name)
            )

            # Sample parameter value
            ## Ranged parameters
            if (param_type in ('float', 'int', 'vec_arr', 'color_arr', 'Color')
                or any(param_type.startswith(t) for t in ('Float', 'Vector'))):
                val = np.array(cur_val)

                # Specify clipping range
                if param_type in ('color_arr', 'Color'):
                    clip_min, clip_max = 0, 1
                else:
                    clip_min = min(default_range[0], stats['range'][0])
                    clip_max = max(default_range[1], stats['range'][1])
                    clip_min = min(max(clip_min, -FLOAT_MAX), FLOAT_MAX)
                    clip_max = min(max(clip_max, -FLOAT_MAX), FLOAT_MAX)

                # Read mean and standard deviation
                mean, std = stats['mean'], stats['std']
                if isinstance(mean, dict):
                    len_key = str(len(cur_val))
                    mean, std = np.array(mean[len_key]), np.array(std[len_key])

                # Sample from normal distribution if there are enough observations
                if confident_flag and np.max(std) <= confident_std_threshold:
                    val = rng.normal(mean, std)

                    # Clip the value to the parameter range
                    val = np.clip(val, clip_min, clip_max)

                # Sample from uniform distribution
                else:
                    upr, cpr = uniform_perturb_ratio, uniform_color_perturb_ratio

                    # Perturb the color in HSV space
                    if param_type == 'Color':
                        val = perturb_color_hsv(rng, val, cpr)

                    # Perturb the color array in HSV space
                    elif param_type == 'color_arr':
                        pos, colors = np.split(val, [1], axis=-1)
                        pos = rng.uniform(np.maximum(pos - upr, 0), np.minimum(pos + upr, 1))
                        colors = perturb_color_hsv(rng, colors, cpr)
                        val = np.concatenate([pos, colors], axis=-1)

                    # Handle the rest of the parameters
                    else:
                        perturb_range = np.maximum(np.abs(val) * upr, min_uniform_perturb_amount)
                        range_min, range_max = val - perturb_range, val + perturb_range

                        # Sample from clipped uniform distribution
                        range_min = np.clip(range_min, clip_min, clip_max)
                        range_max = np.clip(range_max, clip_min, clip_max)
                        val = rng.uniform(range_min, range_max)

                # Round to the nearest integer if necessary
                if param_type == 'int':
                    val = np.floor(val + 0.5).astype(np.int64)

            ## Discrete parameters
            elif param_type in ('str', 'bool', 'String'):

                # Sample from the frequency distribution if there are enough observations
                if confident_flag:
                    freq = stats['freq']
                    probs = np.array(list(freq.values())) / sum(freq.values())
                    val = rng.choice(list(freq.keys()), p=probs)

                    # Update the parameter value
                    if param_type == 'bool':
                        val = {'true': True, 'false': False}[val]

                # Sample uniformly from the default range
                else:
                    rand_item = rng.choice(
                        (default_range or list(stats['freq'].keys())) if param_type in ('str', 'String')
                        else [False, True]
                    )
                    rand_prob = rng.random()
                    val = rand_item if rand_prob < uniform_perturb_ratio else val

            else:
                raise ValueError(f'Unknown parameter type: {param_type}')

            # Update parameter value
            if isinstance(val, (np.ndarray, np.generic)):
                val = val.tolist()

            # Mark the parameter as sampled
            ret_val = 1, [global_param_id]

        # Write the parameter value
        ## Parameters with slots
        if not is_attr:
            slots[param_id].default_value = val

        ## Attributes - color ramp
        elif param_type == 'color_arr':
            elements: T.ColorRampElements = getattr(src_obj, param_name)
            for _ in range(len(elements) - len(val)):
                elements.remove(elements[-1])
            for i, (pos, *color) in enumerate(val):
                if i >= len(elements):
                    elements.new(pos)
                else:
                    elements[i].position = pos
                elements[i].color = color

        ## Attributes - curve mapping
        elif param_type == 'vec_arr':
            points: T.CurveMapPoints = src_obj.points
            for _ in range(len(points) - len(val)):
                points.remove(points[-1])
            for i, (x, y) in enumerate(val):
                if i >= len(points):
                    points.new(x, y)
                else:
                    points[i].location = (x, y)

        ## Other attributes
        else:
            attr_name = trans_name_dict.get(param_name, param_name)
            setattr(src_obj, attr_name, val)

        return ret_val

    # Initialize return values
    num_params, sampled_global_ids = 0, []

    # Iterate over node parameters
    for i, stats in enumerate(node_param_stats):

        # Skip connected parameters or those without default values
        default_val = stats.get('default')
        if i in linked_ids or ('group' not in stats and default_val is None):
            continue

        # Read current parameter value
        cur_val = param_info_dict.get(i, {}).get('value', default_val)

        # Sample or reset the parameter
        ret_val = process_param(
            i, global_param_id + num_params, cur_val, stats,
            sample=sample, is_attr=stats['is_attr'], src_obj=node
        )

        # Update return values
        num_params += ret_val[0]
        sampled_global_ids.extend(ret_val[1])

    return num_params, sampled_global_ids


def sample_params(
        material: T.Material, file_path: str, data_root: str, info_dir: str, output_folder: str,
        seed: int = 0, sample_name: str = 'temp', sample_options: dict[str, Any] = {}
    ) -> tuple[int, list[int]]:
    '''Sample parameters for the current material to generate appearance variations. Return the
    number of parameters processed and sampled parameter (global) IDs.
    '''
    # Get node sequence in DFS order
    node_tree = material.node_tree
    output_node = find_output_node(node_tree, recursive=False)
    if output_node is None:
        raise RuntimeError('Output node not found')
    node_seq = dfs_from_node(output_node)

    # Read analysis result
    analysis_path = osp.join(osp.dirname(file_path), 'analysis_result.json')
    if not osp.isfile(analysis_path):
        analysis_path = get_analysis_url(file_path=file_path, info_dir=info_dir)
    if not osp.isfile(analysis_path):
        raise FileNotFoundError(f'Analysis result not found: {analysis_path}')
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)

    # Check analysis result against node sequence
    if len(analysis) != len(node_seq):
        raise RuntimeError(
            f'Number of nodes in analysis result does not match node sequence'
            f'({len(analysis)} != {len(node_seq)})'
        )

    # Read parameter statistics
    param_stats_path = osp.join(data_root, 'param_stats.json')
    if not osp.isfile(param_stats_path):
        raise FileNotFoundError(f'Parameter statistics not found: {param_stats_path}')
    with open(param_stats_path, 'r') as f:
        param_stats = json.load(f)

    # Read parameter statistics per node using node type as key
    node_param_stats: list[dict[str, Any]] = []

    for node, node_info in zip(node_seq, analysis):
        node.name = node_info['name']
        sig_key = node_info['type']
        if sig_key == 'ShaderNodeGroup':
            sig_key = osp.join('node_groups', node_info['group_type'])
        node_param_stats.append(param_stats.get(sig_key, []))

    # Set up output folder
    os.makedirs(output_folder, exist_ok=True)

    # Create random number generator
    rng = np.random.default_rng(seed)

    # Create transpiler
    transpiler = Transpiler(argparse.Namespace(
        file_path=file_path, info_dir=info_dir, analysis_path=analysis_path,
        node_order='reverse_dfs', format='code'
    ))

    # Sample parameters for each node
    num_params, sampled_global_ids = 0, []

    for node, node_info, node_stats in zip(node_seq, analysis, node_param_stats):
        ret_val = sample_node_params(
            node, node_info, node_stats, rng=rng, global_param_id=num_params, **sample_options
        )
        num_params += ret_val[0]
        sampled_global_ids.extend(ret_val[1])

    # Transpile the material
    transpiler(
        node_tree, curate_material=False,
        output_file=osp.join(output_folder, f'{sample_name}_full.py')
    )

    # Reset the material
    # for node, node_info, node_stats in zip(node_seq, analysis, node_param_stats):
    #     sample_node_params(node, node_info, node_stats, sample=False)

    return num_params, sampled_global_ids


def main():
    # Command line argument parser
    PROG_DESC = 'Generate parameter variations for Blender materials'
    parser = argparse.ArgumentParser(description=PROG_DESC)
    parser.add_argument('file_path', type=str, help='Path to Blender or code file')
    parser.add_argument('-m', '--mode', type=str, default='data', choices=['data', 'opt'],
                        help='Parameter sampling mode')
    parser.add_argument('-d', '--data_root', type=str,
                        default=osp.join(ROOT_DIR, 'material_dataset_filtered'),
                        help='Root directory of Blender or code files')
    parser.add_argument('-i', '--info_dir', type=str,
                        default=osp.join(ROOT_DIR, 'material_dataset_info'),
                        help='Directory to analysis results')
    parser.add_argument('-o', '--output_folder', type=str,
                        default=osp.join(ROOT_DIR, 'material_dataset_filtered'),
                        help='Output directory')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-cf', '--confident_sample', action='store_true',
                        help='Enable confident sampling')
    parser.add_argument('--load_sampled_ids', type=str, default=None,
                        help='Load parameter IDs to sample')
    parser.add_argument('--save_sampled_ids', action='store_true',
                        help='Save sampled parameter IDs')

    # Parse command line arguments after '--'
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    args = parser.parse_args(argv)

    # Set up rendering scene
    obj = setup_plane()

    # Load Python code associated with the Blender file
    if args.file_path.endswith('.py'):
        code_path = args.file_path
    else:
        code_path = osp.join(osp.dirname(args.file_path), 'blender_full.py')
    with open(code_path, 'r') as f:
        code = f.read()

    # Apply the material code
    node_groups_dir = osp.join(args.info_dir, 'node_types', 'node_groups')
    apply_material_code(obj, code, node_groups_dir=node_groups_dir)

    # Global parameter sammpling options
    sample_options = {
        'confident_sample': args.confident_sample,
        'confident_freq': 20,
        'confident_std_threshold': 10.0,
        'constant_default_freq': 50
    }

    # Mode-specific options
    if args.mode == 'data':
        sample_options.update({
            'default_perturb_prob': 0.2,
            'perturb_prob': 1.0,
            'uniform_perturb_ratio': 0.25,
            'uniform_color_perturb_ratio': 0.5,
            'min_uniform_perturb_amount': 0.05
        })
    else:
        sample_options.update({
            'default_perturb_prob': 1.0,
            'perturb_prob': 1.0,
            'uniform_perturb_ratio': 0.2,
            'uniform_color_perturb_ratio': 0.2,
            'min_uniform_perturb_amount': 0.05
        })

        # Load parameter IDs to sample
        if args.load_sampled_ids:
            with open(args.load_sampled_ids, 'r') as f:
                source_global_ids = json.load(f)

            # Randomly sample a subset of parameter IDs
            rng = np.random.default_rng(args.seed)
            sample_options['source_global_ids'] = rng.choice(
                source_global_ids, max(int(len(source_global_ids) * 0.1 + 0.5), 2), replace=False
            ).tolist()

    # Sample parameters
    num_params, sampled_global_ids = sample_params(
        obj.active_material, args.file_path, args.data_root, args.info_dir, args.output_folder,
        seed=args.seed, sample_options=sample_options
    )

    # Print the number of parameters processed and sampled
    if args.mode == 'opt':
        print(f'Processed {num_params} parameters, sampled {len(sampled_global_ids)} parameters')

        # Save sampled parameter IDs
        if args.save_sampled_ids:
            sampled_ids_path = osp.join(args.output_folder, 'sampled_ids.json')
            with open(sampled_ids_path, 'w') as f:
                json.dump(sampled_global_ids, f)


if __name__ == '__main__':
    main()
