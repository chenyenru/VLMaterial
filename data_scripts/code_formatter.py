# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

from abc import ABC, abstractmethod
from typing import Any
import math

import bpy.types as T

from analyze import transpiler_utils, get_param_value, get_attr_info
from utils import NodeSignatureReader, translate_name


class BaseFormatter(ABC):
    '''Base class for formatting node sequence into code.
    '''
    def __init__(self, material_info: list[dict[str, Any]], node_type_dir: str, minimize: bool = False):
        # Build node info dictionary
        self.material_info = {node['name']: node for node in material_info}

        self.node_type_dir = node_type_dir
        self.minimize = minimize

        # Node type signature reader
        self.sig_reader = NodeSignatureReader(node_type_dir)

    def _get_node_type_signature(self, node: T.ShaderNode) -> dict[str, Any]:
        '''Get node type information from node type directory.
        '''
        # Read node type name and validate against node object
        node_info = self.material_info[translate_name(node.name)]
        node_type = node_info['type']
        if node_type != node.bl_idname:
            raise ValueError(f"Node type does not match analysis result: "
                             f"{node_info['type']} != {node.bl_idname}")

        return self.sig_reader.read(node_type, node_info.get('group_type'))

    def _represent_literal(self, val: Any) -> str:
        '''Represent literal value as code.
        '''
        if isinstance(val, tuple):
            return f"({', '.join(map(self._represent_literal, val))})"
        elif isinstance(val, list):
            return f"[{', '.join(map(self._represent_literal, val))}]"
        elif isinstance(val, dict):
            repr_dict_items = ', '.join(
                f'{self._represent_literal(k)}: {self._represent_literal(v)}'
                for k, v in val.items()
            )
            return f'{{{repr_dict_items}}}'
        elif not isinstance(val, (int, float, str, bool)):
            raise ValueError(f"Unsupported literal type: {type(val).__name__}")

        # Keep float values with three most significant digits
        if isinstance(val, float) and val:
            power = 10 ** (2 - max(math.floor(math.log10(abs(val))), -1))
            val = int(val * power + (0.5 if val >= 0 else -0.5)) / power

        return repr(val)

    @abstractmethod
    def __call__(self, nodes: list[T.ShaderNode]) -> str: ...


class CodeFormatter(BaseFormatter):
    '''Format node sequence into code. Supports minimal code format which reduces
    code content to a series of tokens.
    '''
    def __init__(self, material_info: list[dict[str, Any]], node_type_dir: str, minimize: bool = False):
        super().__init__(material_info, node_type_dir, minimize=minimize)

        # Code template
        self.template = (
            "import bpy\n\n"
            "def shader_material(material: bpy.types.Material):\n"
            "    material.use_nodes = True\n"
            "    nodes = material.node_tree.nodes\n"
            "    links = material.node_tree.links\n\n"
            "    # Create nodes\n"
            "    {0}\n\n"
            "    # Create links to connect nodes\n"
            "    {1}\n\n"
            "    # Set parameters for each node\n"
            "    {2}\n"
        )
        self.indent = 4

    def _get_var_names(self, nodes: list[T.ShaderNode]) -> list[str]:
        '''Assign variable names to nodes.
        '''
        var_names = []
        for node in nodes:
            var_names.append(transpiler_utils.get_varname(node, var_names))

        return var_names

    def _represent_color_ramp(
            self, ramp: dict[str, Any], default_ramp: dict[str, Any], attr_expr: str
        ) -> list[str]:
        '''Represent color ramp as code.
        '''
        repr_code = []

        # Translate value keys to object fields
        key_dict = {
            'mode': 'color_mode',
            'interp': 'interpolation',
            'hue_interp': 'hue_interpolation'
        }

        for key, val in ramp.items():
            if val == default_ramp[key]:
                continue

            key_expr = f'{attr_expr}.{key_dict.get(key, key)}'

            # Translate color stops
            if key == 'elements':
                ref_elements = default_ramp[key]

                for i, stop in enumerate(val):
                    if i < len(ref_elements) and stop == ref_elements[i]:
                        continue

                    pos, color = stop[0], stop[1:]

                    # Add new color stop
                    if i >= len(ref_elements):
                        repr_code.extend([
                            f"{key_expr}.new({self._represent_literal(pos)})",
                            f"{key_expr}[{i}].color = {self._represent_literal(color)}"
                        ])

                    # Update existing color stop
                    else:
                        ref_stop = ref_elements[i]
                        ref_pos, ref_color = ref_stop[0], ref_stop[1:]

                        if pos != ref_pos:
                            repr_code.append(
                                f"{key_expr}[{i}].position = {self._represent_literal(pos)}"
                            )
                        if color != ref_color:
                            repr_code.append(
                                f"{key_expr}[{i}].color = {self._represent_literal(color)}"
                            )

            # Translate literal fields
            else:
                repr_code.append(f'{key_expr} = {self._represent_literal(val)}')

        return repr_code

    def _represent_curve_mapping(
            self, curves: dict[str, Any], default_curves: dict[str, Any], attr_expr: str
        ) -> list[str]:
        '''Represent curve mapping as code.
        '''
        repr_code = []

        # Translate value keys to object fields
        key_dict = {
            'black': 'black_level',
            'white': 'white_level',
            'clip_range': 'clip_'
        }

        for key, val in curves.items():
            if val == default_curves[key]:
                continue

            key_expr = f'{attr_expr}.{key_dict.get(key, key)}'

            # Translate clip range
            if key == 'clip_range':
                field_names = ['min_x', 'min_y', 'max_x', 'max_y']
                for v, dv, fn in zip(val, default_curves[key], field_names):
                    if v != dv:
                        repr_code.append(f'{key_expr}{fn} = {self._represent_literal(v)}')

            # Translate curves
            elif key == 'curves':
                for i, (curve, ref_curve) in enumerate(zip(val, default_curves[key])):
                    if curve == ref_curve:
                        continue

                    # Translate curve points
                    for j, pt in enumerate(curve):
                        if j < len(ref_curve) and pt == ref_curve[j]:
                            continue

                        # Add new point or update existing point location
                        if j >= len(ref_curve):
                            repr_code.append(
                                f"{key_expr}[{i}].points.new({self._represent_literal(pt[0])}, "
                                f"{self._represent_literal(pt[1])})"
                            )
                        elif pt != ref_curve[j]:
                            repr_code.append(
                                f"{key_expr}[{i}].points[{j}].location = "
                                f"{self._represent_literal(pt)}"
                            )

            # Translate literal fields
            else:
                repr_code.append(f'{key_expr} = {self._represent_literal(val)}')

        return repr_code

    def format_node(
            self, node: T.ShaderNode, var_dict: dict[str, str], slot_dict: dict[str, list[str]],
        ) -> tuple[str, str, str]:
        '''Generate code for node, edge, and parameter information.
        '''
        # Get node type information
        sig = self._get_node_type_signature(node)

        # Instantiate node
        var_node = var_dict[node.name]
        node_code = [f"{var_node} = nodes.new('{node.bl_idname}')"]

        # Read node group type
        if isinstance(node, T.ShaderNodeGroup):
            node_group_type = self.material_info[translate_name(node.name)]['group_type']
            node_code.append(
                f"{var_node}.node_tree = "
                f"bpy.data.node_groups['{node_group_type}']"
            )

        # Connect input edges
        edge_code = []

        for i, input_slot in enumerate(node.inputs):
            if not input_slot.is_linked:
                continue

            link = input_slot.links[0]
            from_node = link.from_node
            from_slot_idx = slot_dict[from_node.name].index(link.from_socket.identifier)
            edge_code.append(
                f"links.new({var_dict[from_node.name]}.outputs[{from_slot_idx}], "
                f"{var_node}.inputs[{i}])"
            )

        # Set default values for unused input slots
        param_code = []

        if isinstance(node, (T.ShaderNodeValue, T.ShaderNodeRGB)):
            io_type, param_slots = 'output', node.outputs
        else:
            io_type, param_slots = 'input', node.inputs

        for i, slot in enumerate(param_slots):
            if io_type == 'input' and slot.is_linked:
                continue

            # Skip slots without default values
            default_val = sig[io_type][i].get('default')
            if default_val is None:
                continue

            # Get current value
            cur_val = get_param_value(slot)

            if cur_val != default_val:
                param_code.append(
                    f"{var_node}.{io_type}s[{i}].default_value = "
                    f"{self._represent_literal(cur_val)}"
                )

        # Set node attributes
        for attr in sig.get('attr', []):
            # Get current value
            cur_val = get_attr_info(node, attr['name'])['value']

            if cur_val != attr['value']:
                # Represent color ramp
                if attr['type'] == 'ColorRamp':
                    param_code.extend(self._represent_color_ramp(
                        cur_val, attr['value'], f"{var_node}.{attr['name']}"))

                # Represent curve mapping
                elif attr['type'] == 'CurveMapping':
                    param_code.extend(self._represent_curve_mapping(
                        cur_val, attr['value'], f"{var_node}.{attr['name']}"))

                # Represent literals
                else:
                    param_code.append(
                        f"{var_node}.{attr['name']} = "
                        f"{self._represent_literal(cur_val)}"
                    )

        # Format code sections
        sep = '\n' + ' ' * self.indent
        return tuple(sep.join(s) for s in (node_code, edge_code, param_code))

    def __call__(self, nodes: list[T.ShaderNode]) -> str:
        # Assign variable names to nodes
        var_names = self._get_var_names(nodes)
        var_dict = {node.name: var_name for node, var_name in zip(nodes, var_names)}

        # Pre-cache output slot names for each node
        slot_dict = {node.name: [slot.identifier for slot in node.outputs] for node in nodes}

        # Format nodes
        node_code, edge_code, param_code = list(zip(*(
            self.format_node(node, var_dict, slot_dict) for node in nodes
        )))
        sep = '\n' + ' ' * self.indent
        code = self.template.format(*(
            sep.join(filter(None, section))
            for section in (node_code, edge_code, param_code)
        ))

        return code
