# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Iterator
import json
import os.path as osp
import sys

ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(ROOT_DIR, 'infinigen', 'worldgen'))
sys.path.append(osp.dirname(osp.abspath(__file__)))

import bpy
import bpy.types as T

from curation import find_output_node, get_output_node, clean_node_tree, expand_node_groups
from code_formatter import CodeFormatter
from utils import get_analysis_url


# Input node iterator
def input_iter(node: T.ShaderNode, reverse: bool = False) -> Iterator[T.ShaderNode]:
    # Reverse input order if needed
    make_iter = reversed if reverse else iter

    for slot in make_iter(node.inputs):
        if len(slot.links) > 1:
            raise RuntimeError(f"Node '{node.name}' has multiple connections to input '{slot.name}'")
        for link in slot.links:
            yield link.from_node


def traverse_dfs(node_tree: T.ShaderNodeTree, reverse: bool = False) -> list[T.ShaderNode]:
    # Get output node
    output_node = get_output_node(node_tree)

    # DFS traversal
    node_seq = [output_node] if reverse else []
    visited = {output_node}
    stack = [(output_node, input_iter(output_node))]

    while stack:
        for node in stack[-1][1]:
            if node not in visited:
                if reverse:
                    node_seq.append(node)
                visited.add(node)
                stack.append((node, input_iter(node)))
                break
        else:
            if not reverse:
                node_seq.append(stack[-1][0])
            stack.pop()

    return node_seq


def traverse_bfs(node_tree: T.ShaderNodeTree, reverse: bool = False) -> list[T.ShaderNode]:
    # Get output node
    output_node = get_output_node(node_tree)

    # BFS traversal
    node_seq, visited, queue = [], {output_node}, [output_node]

    while queue:
        node = queue.pop(0)
        node_seq.append(node)

        for next_node in input_iter(node, reverse=not reverse):
            if next_node not in visited:
                visited.add(next_node)
                queue.append(next_node)

    if not reverse:
        node_seq.reverse()

    return node_seq


class Transpiler:
    '''Translates Blender procedural material graph to code. Supports multiple
    output formats and node traversal orders.
    '''
    def __init__(self, args: Namespace):
        # Read analysis results
        if args.analysis_path is not None:
            analysis_path = args.analysis_path
        else:
            analysis_path = get_analysis_url(
                file_path=getattr(args, 'file_path', None),
                info_dir=args.info_dir
            )

        if not osp.isfile(analysis_path):
            raise FileNotFoundError(f'Analysis results not found: {analysis_path}')
        with open(analysis_path, 'r') as f:
            material_info = json.load(f)

        # Node type information
        node_type_dir = osp.join(args.info_dir, 'node_types')

        # Node tree traversal
        node_order = args.node_order.split('_')[-1]
        reverse_flag = args.node_order.startswith('reverse_')

        algo_dict = {'dfs': traverse_dfs, 'bfs': traverse_bfs}
        self.traverse = partial(algo_dict[node_order], reverse=reverse_flag)

        # Code formatter
        self.code_format = args.format.split('_')[-1]
        self.minimize_flag = args.format.startswith('tokens_')

        if self.code_format == 'func' and not reverse_flag:
            raise ValueError('Function format requires reverse node traversal')

        self.formatter = CodeFormatter(
            material_info, node_type_dir, minimize=self.minimize_flag)

    def __call__(
            self, node_tree: T.ShaderNodeTree, curate_material: bool = True,
            output_file: str | None = None
        ) -> str:
        # Clean up node tree
        if curate_material:
            node_tree = clean_node_tree(node_tree)

        # Find output node
        output_node = find_output_node(node_tree)
        if output_node is None:
            raise RuntimeError(f"Output node not found in node tree '{node_tree.name}'")

        # Expand small node groups
        if curate_material:
            expand_node_groups(node_tree, size_limit=30)

        # Traverse node tree
        nodes = self.traverse(node_tree)

        # Format code and write to file
        code = self.formatter(nodes)
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write(code)

        return code


def main():
    # Command line argument parser
    parser = ArgumentParser(description='Transpile Blender procedural material to code')
    parser.add_argument('output_file', metavar='FILE', type=str, help='Output file path')
    parser.add_argument('-i', '--info_dir', type=str, default=osp.join(ROOT_DIR, 'material_dataset_info'),
                        help='Directory containing node type information')
    parser.add_argument('-n', '--node_order', type=str, default='reverse_dfs',
                        choices=['dfs', 'bfs', 'reverse_dfs', 'reverse_bfs'], help='Node traversal order')
    parser.add_argument('-f', '--format', type=str, default='code',
                        choices=['code', 'func', 'tokens_code', 'tokens_func'], help='Output format')
    parser.add_argument('-a', '--analysis_path', type=str, default=None,
                        help='Path to analysis results JSON file')
    parser.add_argument('--skip-curation', action='store_true',
                        help='Skip material curation before transpilation')

    argv = sys.argv
    args = parser.parse_args(argv[argv.index('--') + 1:])

    # Set up the scene
    bpy.ops.mesh.primitive_cube_add()
    mesh = bpy.context.active_object
    mesh.data.materials.append(bpy.data.materials[0])

    # Transpile the material
    material = mesh.material_slots[0].material
    if not material.use_nodes:
        raise RuntimeError('Material must use nodes')

    Transpiler(args)(
        material.node_tree, curate_material=not args.skip_curation,
        output_file=args.output_file
    )


if __name__ == '__main__':
    main()
