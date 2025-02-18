# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

from argparse import ArgumentParser
from typing import Any
import json
import mathutils
import os
import os.path as osp
import sys

ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(ROOT_DIR, 'infinigen', 'worldgen'))
sys.path.append(osp.dirname(osp.abspath(__file__)))

import bpy
import bpy.types as T

from nodes.node_transpiler import transpiler_blender as transpiler_utils
from curation import (
    get_node_attrs, get_output_node, dfs_from_node, clean_node_tree, find_output_node,
    expand_node_groups
)
from utils import translate_name, FLOAT_MAX


def get_param_value(slot: T.NodeSocket) -> Any:
    # Read parameter value
    val = getattr(slot, 'default_value', None)
    if isinstance(val, (T.bpy_prop_array, mathutils.Vector, mathutils.Euler, mathutils.Color)):
        val = list(val)
    elif val is not None and not isinstance(val, (int, float, str, bool)):
        raise NotImplementedError(f"Default value type '{type(val).__name__}' is not supported")
    return val


def get_node_group_info(node: T.ShaderNodeGroup) -> dict[str, Any]:
    # Initialize node group info dictionary
    node_group_info = {
        'name': translate_name(node.node_tree.name),
        'source_file': bpy.data.filepath,
        'source_name': node.node_tree.name
    }

    # Run DFS from the active output node
    output_node = get_output_node(node.node_tree)
    if output_node is None:
        raise RuntimeError(f"No active output node found in node group '{node.node_tree.name}'")

    node_seq = dfs_from_node(output_node)
    node_seq = [
        n for n in node_seq
        if not isinstance(n, (T.NodeGroupInput, T.NodeGroupOutput))
    ]

    # Count the number of nodes and subgroups
    node_group_info.update({
        'size': len(node_seq),
        'full_size': len(node_seq) + sum(
            get_node_group_info(n)['size'] - 1
            for n in node_seq if isinstance(n, T.ShaderNodeGroup)
        ),
        'subgroup': sorted(set([
            translate_name(n.node_tree.name) for n in node_seq
            if isinstance(n, T.ShaderNodeGroup)
        ]))
    })

    return node_group_info


def get_socket_info(slot: T.NodeSocket | T.NodeSocketInterface) -> dict[str, Any]:
    # Get socket type
    if isinstance(slot, T.NodeSocketInterface):
        dtype = slot.bl_socket_idname[len('NodeSocket'):]
        name = translate_name(slot.name)
    else:
        dtype = slot.bl_idname[len('NodeSocket'):]
        name = translate_name(slot.identifier)

    socket_info = {'name': name, 'type': dtype}

    # Input slot
    if not slot.is_output:
        # Default value
        if hasattr(slot, 'default_value'):
            socket_info['default'] = get_param_value(slot)

        # Parameter range
        if hasattr(slot, 'min_value'):
            min_value, max_value = slot.min_value, slot.max_value
            socket_info['min'] = max(min_value, -FLOAT_MAX)
            socket_info['max'] = min(max_value, FLOAT_MAX)

        # Slot with multiple inputs
        if getattr(slot, 'is_multi_input', False):
            raise RuntimeError(f"Multi-input slot '{slot.name}' is not supported")

    # Output slot
    elif (isinstance(slot, T.NodeSocket)
          and isinstance(slot.node, (T.ShaderNodeValue, T.ShaderNodeRGB))
          and hasattr(slot, 'default_value')):
        socket_info['default'] = get_param_value(slot)

    return socket_info


def get_attr_info(node: T.ShaderNode, attr_name: str) -> dict[str, Any]:
    attr_value = getattr(node, attr_name)

    # Trivial types
    if isinstance(attr_value, (int, float, str, bool)):
        dtype = type(attr_value).__name__

    # Color ramp
    elif isinstance(attr_value, T.ColorRamp):
        dtype = 'ColorRamp'
        attr_value = {
            'mode': attr_value.color_mode,
            'interp': attr_value.interpolation,
            'hue_interp': attr_value.hue_interpolation,
            'elements': [
                [el.position, *el.color]
                for el in attr_value.elements
            ]
        }

    # Curve mapping
    # Parameters like 'black_level', 'white_level', and 'clip_range' are omitted as
    # they are never manually set by the user
    elif isinstance(attr_value, T.CurveMapping):
        dtype = 'CurveMapping'
        attr_value = {
            'use_clip': attr_value.use_clip,
            'curves': [
                [list(p.location) for p in cm.points]
                for cm in attr_value.curves
            ]
        }

    # Unknown type
    else:
        raise RuntimeError(f"Attribute '{attr_name}' of node '{node.bl_idname}' "
                           f"has unsupported type '{type(attr_value).__name__}'")

    return {'name': attr_name, 'type': dtype, 'value': attr_value}


def check_unsupported_node(node: T.ShaderNode):
    # Node group
    if isinstance(node, T.ShaderNodeGroup):
        # Run DFS from the active output node
        output_node = get_output_node(node.node_tree)
        if get_output_node is None:
            raise RuntimeError(f"No active output node found in node group '{node.node_tree.name}'")

        try:
            for node in dfs_from_node(output_node):
                check_unsupported_node(node)
        except RuntimeError as e:
            msg = e.args[0]
            if msg.startswith('Unsupported node type'):
                node_type = msg.split("'")[1]
                e.args[0] = f"Node group '{node.node_tree.name}' has unsupported node type '{node_type}'"
            raise e

    # Non-shader nodes
    if (not isinstance(node, (T.ShaderNode, T.NodeGroupInput, T.NodeGroupOutput))
        or isinstance(node, (T.ShaderNodeTexImage, T.ShaderNodeTexPointDensity, T.ShaderNodeScript))):
        raise RuntimeError(f"Unsupported node type '{node.bl_idname}'")


def get_node_signature(node: T.ShaderNode, node_tree: T.ShaderNodeTree) -> dict[str, Any]:
    # Initialize node info dictionary
    signature = {'type': node.bl_idname}

    # Node group
    if isinstance(node, T.ShaderNodeGroup):
        # Node name and source file
        signature.update(get_node_group_info(node))

        # Input and output slots
        signature.update({
            'input': [get_socket_info(slot) for slot in node.node_tree.inputs],
            'output': [get_socket_info(slot) for slot in node.node_tree.outputs]
        })

    # Regular node
    else:
        # Create a reference node at default state
        ref_node = node_tree.nodes.new(node.bl_idname)

        # Input and output slots
        signature.update({
            'input': [get_socket_info(slot) for slot in ref_node.inputs],
            'output': [get_socket_info(slot) for slot in ref_node.outputs]
        })

        # Node attributes
        signature['attr'] = [
            get_attr_info(ref_node, attr_name)
            for attr_name in get_node_attrs(ref_node)
        ]

        # Remove the reference node
        node_tree.nodes.remove(ref_node)

    return signature


def match_signature(sig: dict[str, Any], ref_sig: dict[str, Any]) -> bool:
    # Check basic fields
    for key in ('type', 'name'):
        if sig.get(key, None) != ref_sig.get(key, None):
            return False

    # Check I/O slots and attributes
    check_fields = ('input', 'output')
    if sig['type'] != 'ShaderNodeGroup':
        check_fields += ('attr',)

    for key in check_fields:
        feats = [(s['name'], s['type']) for s in sig[key]]
        ref_feats = [(s['name'], s['type']) for s in ref_sig[key]]
        if feats != ref_feats:
            return False

    return True


def get_node_info(node: T.ShaderNode, sig: dict[str, Any]) -> dict[str, Any]:
    # Initialize node info dictionary
    node_info = {
        'name': translate_name(node.name),
        'type': sig['type'],
        'group_type': None,
        'linked_input': [],
        'params': []
    }

    is_node_group = isinstance(node, T.ShaderNodeGroup)
    if is_node_group:
        node_info['group_type'] = sig['name']
    else:
        del node_info['group_type']

    # Input slots
    params = node_info['params']
    num_slots = len(node.inputs)

    for i, slot in enumerate(node.inputs):
        if slot.is_linked:
            node_info['linked_input'].append(i)
        else:
            slot_sig = sig['input'][i]
            slot_name = translate_name(slot.name if is_node_group else slot.identifier)
            if slot_name != slot_sig['name']:
                raise RuntimeError(f"Slot name mismatch: '{slot_name}' != '{slot_sig['name']}'")

            # Check parameter value against default
            param_value = get_param_value(slot)
            if param_value != slot_sig.get('default', None):
                params.append({
                    'id': i,
                    'name': slot_sig['name'],
                    'value': param_value
                })

    # Output slots for value and RGB nodes
    if isinstance(node, (T.ShaderNodeValue, T.ShaderNodeRGB)):
        for i, slot in enumerate(node.outputs):
            slot_sig = sig['output'][i]
            slot_name = translate_name(slot.identifier)
            if slot_name != slot_sig['name']:
                raise RuntimeError(f"Slot name mismatch: '{slot_name}' != '{slot_sig['name']}'")

            # Check parameter value against default
            param_value = get_param_value(slot)
            if param_value != slot_sig.get('default', None):
                params.append({
                    'id': i + num_slots,
                    'name': slot_sig['name'],
                    'value': param_value
                })

        num_slots += len(node.outputs)

    # Attributes
    if not is_node_group:
        for i, attr_name in enumerate(get_node_attrs(node)):
            attr_sig = sig['attr'][i]
            if attr_name != attr_sig['name']:
                raise RuntimeError(f"Attribute name mismatch: '{attr_name}' != '{attr_sig['name']}'")

            attr_value = get_attr_info(node, attr_name)['value']
            if attr_value != attr_sig['value']:
                params.append({
                    'id': i + num_slots,
                    'name': attr_sig['name'],
                    'value': attr_value
                })

    return node_info


def analyze_object(
        obj: T.Object, save_path: str, info_dir: str, size_limit: int = 30,
        curate_material: bool = True, check_node_type: bool = False
    ):
    # Identify analysis targets
    targets = [mod for mod in obj.modifiers if mod.type == 'NODES']
    targets += [slot.material for slot in obj.material_slots if slot.material.use_nodes]
    target_funcnames = [transpiler_utils.get_func_name(target) for target in targets]

    if not targets:
        raise RuntimeError(f"No analysis target found for object {repr(obj)}")
    elif len(targets) > 1:
        raise RuntimeError(f'Found {len(targets)} analysis targets for object {repr(obj)}: {target_funcnames}')
    elif target_funcnames[0] != 'shader_material':
        raise RuntimeError(f'Unknown analysis target for object {repr(obj)}: {target_funcnames[0]}')

    # Cleanup the material
    node_tree = targets[0].node_tree
    if curate_material:
        node_tree = clean_node_tree(node_tree)

    # Find the output node
    output_node = find_output_node(node_tree, recursive=curate_material)
    if output_node is None:
        raise RuntimeError(f"No output node found in material '{target_funcnames[0]}'")

    # Expand small node groups
    if curate_material:
        expand_node_groups(node_tree, size_limit=size_limit)
    node_seq = dfs_from_node(output_node)

    # Graph size too large
    if len(node_seq) > size_limit:
        raise RuntimeError(f"Oversized node graph: {len(node_seq)} > {size_limit}")

    # Create the node type directory
    node_type_dir = osp.join(info_dir, 'node_types')
    os.makedirs(node_type_dir, exist_ok=True)

    # Collect and check node type signatures
    signatures: list[dict[str, Any]] = []

    for node in node_seq:
        # Check unsupported node types
        check_unsupported_node(node)

        # Get the node signature
        signature = get_node_signature(node, node_tree)
        is_node_group = isinstance(node, T.ShaderNodeGroup)

        # Search for the reference node type signature
        if is_node_group:
            ref_sig_path = osp.join(node_type_dir, 'node_groups', f"{signature['name']}.json")
        else:
            ref_sig_path = osp.join(node_type_dir, f"{signature['type']}.json")

        # Check node type signature
        counter, success = 0, False

        while True:
            # Apply suffix
            cur_path = (ref_sig_path.replace('.json', f'_{counter:03d}.json')
                        if counter > 0 else ref_sig_path)
            if not osp.exists(cur_path):
                break

            # Load and compare the signature
            with open(cur_path, 'r') as f:
                ref_signature = json.load(f)
            if match_signature(signature, ref_signature):
                success = True
                break
            if not is_node_group:
                raise RuntimeError(f"Node signature mismatch for '{signature['type']}'")

            counter += 1

        # Register the new signature
        if not success:
            if check_node_type:
                raise RuntimeError(f"Unknown node type signature for '{signature['type']}'")
            os.makedirs(osp.dirname(cur_path), exist_ok=True)
            with open(cur_path, 'w') as f:
                json.dump(signature, f, indent=2)

        # Update group type name to match the file name
        if counter > 0:
            signature['name'] = osp.splitext(osp.basename(cur_path))[0]

        signatures.append(signature)

    # Write the analysis to a JSON file
    node_info = [get_node_info(node, sig) for node, sig in zip(node_seq, signatures)]
    with open(save_path, 'w') as f:
        json.dump(node_info, f, indent=2)


def main():
    # Command line argument parser
    parser = ArgumentParser(description='Analyze Blender material node graph')
    parser.add_argument('save_path', type=str, help='Path to save the analysis result')
    parser.add_argument('--info_dir', type=str, default=None, help='Directory to store analysis info')
    parser.add_argument('--code_path', type=str, default=None, help='Path to source code')
    parser.add_argument('-s', '--size_limit', type=int, default=30,
                        help='Node graph size limit')
    parser.add_argument('-c', '--check_node_type', action='store_true',
                        help='Check node types and raise an error if unsupported types are found')
    parser.add_argument('--skip_curation', action='store_true', help='Skip material curation')

    # Process command line arguments
    argv = sys.argv
    argv = argv[argv.index('--') + 1:]
    args = parser.parse_args(argv)

    # Set up the scene
    bpy.ops.mesh.primitive_cube_add()
    mesh = bpy.context.active_object
    mesh.data.materials.append(bpy.data.materials[0])

    # Create a new material from the source code if provided
    if args.code_path is not None:
        from render import apply_material_code

        with open(args.code_path, 'r') as f:
            code = f.read()

        # Apply the material code
        node_groups_dir = osp.join(args.info_dir, 'node_types', 'node_groups')
        apply_material_code(mesh, code, node_groups_dir=node_groups_dir)

    # Analyze the material node graph
    analyze_object(
        mesh, args.save_path, args.info_dir, size_limit=args.size_limit,
        curate_material=not args.skip_curation, check_node_type=args.check_node_type
    )


if __name__ == '__main__':
    # Detect Blender version
    # if bpy.app.version[:2] != (3, 3):
    #     raise RuntimeError('Blender version 3.3.x is required')

    main()
