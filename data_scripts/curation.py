# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

from typing import Sequence, Iterator

import bpy.types as T


def get_node_attrs(node: T.ShaderNode) -> list[str]:
    '''Get the attribute parameters of a shader node.
    '''
    # Node groups do not have attributes
    if isinstance(node, T.ShaderNodeGroup):
        return []

    # Get node attributes that are not inherited from the base class
    orig_attrs = dir(node)
    attrs = set(orig_attrs) - set(dir(node.__class__))
    attrs = set(
        attr for attr in attrs
        if not (attr.startswith('__') or attr.startswith('bl_') or callable(getattr(node, attr)))
    )
    attrs -= {
        'id_data', 'type', 'location', 'width', 'width_hidden', 'height', 'dimensions', 'name', 'label',
        'inputs', 'outputs', 'internal_links', 'parent', 'use_custom_color', 'color', 'select',
        'show_options', 'show_preview', 'hide', 'mute', 'show_texture', 'rna_type', 'node_tree'
    }

    # Remove node-specific attributes
    attrs -= {'color_mapping', 'texture_mapping', 'object', 'image', 'image_user', 'is_active_output',
              'target', 'uv_map'}

    return sorted(attrs, key=lambda x: orig_attrs.index(x))


def get_slot_by_id(slots: Sequence[T.NodeSocket], slot_id: str) -> T.NodeSocket | None:
    '''Get a socket of a shader node by its identifier.
    '''
    return next((s for s in slots if s.identifier == slot_id), None)


def copy_node_attrs(dst: T.ShaderNode, src: T.ShaderNode):
    '''Copy attributes from one node to another.
    '''
    if dst.bl_idname != src.bl_idname:
        raise TypeError(f"Node types do not match: {dst.bl_idname} != {src.bl_idname}")

    # Copy the default values of I/O slots
    for i, src_input in enumerate(src.inputs):
        if hasattr(src_input, 'default_value'):
            dst.inputs[i].default_value = src_input.default_value
    for i, src_output in enumerate(src.outputs):
        if hasattr(src_output, 'default_value'):
            dst.outputs[i].default_value = src_output.default_value

    # Copy node attributes
    for attr in get_node_attrs(src):
        src_attr, dst_attr = getattr(src, attr), getattr(dst, attr)

        # Color ramp
        if isinstance(src_attr, T.ColorRamp):
            for k in ('color_mode', 'interpolation', 'hue_interpolation'):
                setattr(dst_attr, k, getattr(src_attr, k))

            # Add or delete color ramp elements
            src_elements: T.ColorRampElements = src_attr.elements
            dst_elements: T.ColorRampElements = dst_attr.elements
            len_diff = len(src_elements) - len(dst_elements)

            if len_diff > 0:
                for _ in range(len_diff):
                    dst_elements.new(0)
            elif len_diff < 0:
                for _ in range(-len_diff):
                    dst_elements.remove(dst_elements[-1])

            # Copy color ramp elements
            for i, src_ele in enumerate(src_elements):
                dst_ele = dst_elements[i]
                dst_ele.position = src_ele.position
                dst_ele.color = src_ele.color

        # Curve mapping
        elif isinstance(src_attr, T.CurveMapping):
            for k in (
                'black_level', 'clip_max_x', 'clip_max_y', 'clip_min_x', 'clip_min_y',
                'tone', 'use_clip', 'extend', 'white_level'
            ):
                setattr(dst_attr, k, getattr(src_attr, k))

            # Process curves
            for src_curve, dst_curve in zip(src_attr.curves, dst_attr.curves):
                src_points: T.CurveMapPoints = src_curve.points
                dst_points: T.CurveMapPoints = dst_curve.points
                len_diff = len(src_points) - len(dst_points)

                # Add or delete curve points
                if len_diff > 0:
                    for _ in range(len_diff):
                        dst_points.new(0, 0)
                elif len_diff < 0:
                    for _ in range(-len_diff):
                        dst_points.remove(dst_points[-1])

                # Copy curve points
                for i, src_point in enumerate(src_points):
                    dst_point = dst_points[i]
                    dst_point.location = src_point.location
                    dst_point.handle_type = src_point.handle_type

        # Regular types
        else:
            setattr(dst, attr, src_attr)


def get_output_node(
        node_tree: T.ShaderNodeTree
    ) -> T.ShaderNodeOutputMaterial | T.NodeGroupOutput | None:
    '''Get the material or group output node of a shader node tree.
    '''
    # Find a valid group output node
    output_node = next((
        n for n in node_tree.nodes
        if isinstance(n, T.NodeGroupOutput) and n.is_active_output
    ), None)
    if output_node is not None:
        return output_node

    # Find a valid material output node
    output_node = node_tree.get_output_node('ALL') or node_tree.get_output_node('EEVEE')

    return output_node


def dfs_from_node(
        output_node: T.ShaderNode, input_slot_ids: list[str] | None = None
    ) -> list[T.ShaderNode]:
    '''Depth-first search (DFS) traversal of a shader node tree starting from a given
    output node. Optionally receiving a mask to specify which input slots of the output
    node are used.
    '''
    # Input node iterator
    def input_iter(node: T.Node, slot_ids: list[str] | None = None) -> Iterator[T.Node]:
        # Apply input slot mask
        if slot_ids is not None:
            slot_iter = (s for s in node.inputs if s.identifier in slot_ids)
        else:
            slot_iter = node.inputs

        # Iterate over input nodes
        yield from (l.from_node for s in slot_iter for l in s.links)

    # DFS traversal
    node_seq, visited = [output_node], {output_node}
    stack = [input_iter(output_node, input_slot_ids)]

    while stack:
        for node in stack[-1]:
            if node not in visited:
                node_seq.append(node)
                visited.add(node)
                stack.append(input_iter(node))
                break
        else:
            stack.pop()

    return node_seq


def transfer_socket_value(dst: T.NodeSocket, src: T.NodeSocket):
    '''Transfer the default value of a socket to another socket. Implicit type conversion
    is handled explicitly.
    '''
    # Skip if the source socket has no default value
    if not hasattr(src, 'default_value'):
        return

    val = src.default_value
    src_type = src.bl_idname[len('NodeSocket'):]
    dst_type = dst.bl_idname[len('NodeSocket'):]

    # Same socket type - directly transfer the default value
    if src_type == dst_type:
        if hasattr(src, 'default_value'):
            dst.default_value = src.default_value
        return

    # Different socket types but the same data type - transfer the default value
    for type_name in ('Float', 'Vector'):
        if src_type.startswith(type_name) and dst_type.startswith(type_name):
            dst.default_value = src.default_value
            return

    # Other implicit conversions
    converted_value = None

    ## Float types
    if src_type.startswith('Float'):
        if dst_type.startswith('Vector'):
            converted_value = [val, val, val]
        elif dst_type == 'Color':
            val = min(max(val, 0.0), 1.0)
            converted_value = [val, val, val, 1.0]

    ## Vector types
    elif src_type.startswith('Vector'):
        if dst_type.startswith('Float'):
            converted_value = sum(val) / 3
        elif dst_type == 'Color':
            val = [min(max(v, 0.0), 1.0) for v in val]
            converted_value = val + [1.0]

    ## Color types
    elif src_type == 'Color':
        if dst_type.startswith('Float'):
            converted_value = sum(val[:3]) / 3
        elif dst_type.startswith('Vector'):
            converted_value = val[:3]

    # Unsupported conversion
    if converted_value is None:
        raise TypeError(
            f"Can not transfer default value from '{src.bl_idname}' to '{dst.bl_idname}'"
        )

    dst.default_value = converted_value


def remove_all_reroutes(node_tree: T.ShaderNodeTree) -> T.ShaderNodeTree:
    '''Remove all reroute nodes in the shader node tree.
    '''
    # List of reroute nodes to be removed
    reroutes = [n for n in node_tree.nodes if isinstance(n, T.NodeReroute)]

    for reroute in reroutes:
        reroute_name = reroute.name

        # Identify nodes and sockets linked to the reroute
        input_links = [l for l in node_tree.links if l.to_node.name == reroute_name]
        output_links = [l for l in node_tree.links if l.from_node.name == reroute_name]

        # Connect each input of the reroute to all the outputs the reroute was connected to
        for in_link in input_links:
            for out_link in output_links:
                node_tree.links.new(in_link.from_socket, out_link.to_socket)

        # Remove the reroute node
        node_tree.nodes.remove(reroute)

    return node_tree


def replace_undefined_node(node_tree: T.ShaderNodeTree, node: T.ShaderNode):
    '''Identify the signature of an undefined node and replace it with an existing one.
    '''
    links = node_tree.links

    # Node is not undefined
    if node.bl_idname != 'NodeUndefined':
        return

    # Identify the node type using input signature
    node_input_ids = [i.identifier for i in node.inputs]

    if node_input_ids == [
        'Factor_Float', 'Factor_Vector', 'A_Float', 'B_Float', 'A_Vector',
        'B_Vector', 'A_Color', 'B_Color'
    ]:
        node_type = 'ShaderNodeMixRGB'
        input_id_map = {'Factor_Float': 'Fac', 'A_Color': 'Color1', 'B_Color': 'Color2'}
        output_id_map = {'Result_Color': 'Color'}
        # attrs_map = {'use_alpha': 'use_alpha', 'blend_type': 'blend_type', 'use_clamp': 'use_clamp'}
    else:
        raise ValueError(f"Can not replace undefined node '{node.name}' with inputs {node_input_ids}")

    # Create a new node
    new_node = node_tree.nodes.new(node_type)
    new_node.location = tuple(node.location)

    # Replace input values and connections
    for slot in node.inputs:
        new_input_name = input_id_map.get(slot.identifier)
        if new_input_name is None:
            continue

        # Copy the input socket
        new_slot = new_node.inputs[new_input_name]
        if hasattr(slot, 'default_value'):
            new_slot.default_value = slot.default_value
        for link in slot.links:
            links.new(link.from_socket, new_slot)

    # Replace output connections
    for slot in node.outputs:
        new_output_name = output_id_map.get(slot.identifier)
        if new_output_name is None:
            continue

        # Copy the output socket
        new_slot = new_node.outputs[new_output_name]
        if hasattr(slot, 'default_value'):
            new_slot.default_value = slot.default_value
        for link in slot.links:
            links.new(new_slot, link.to_socket)

    # Replace attributes
    # TODO: attributes are lost, any way to get them?
    print(f"Replace undefined node with {new_node.bl_idname}")

    # Remove the old node
    node_tree.nodes.remove(node)


def ungroup_node_group(
        node_tree: T.ShaderNodeTree, node_group: T.ShaderNodeGroup,
        group_output_slot_ids: list[str] | None = None, recursive: bool = False
    ):
    '''Ungroup a shader node group within a node tree. Optionally receiving a mask to
    specify which output slots of the node group are used.
    '''
    # Exit if the input is not a node group or has an empty node tree
    if node_group.node_tree is None:
        return

    # Get the group node tree and its output node
    group_tree = node_group.node_tree
    output_node = get_output_node(group_tree)

    # Get active node sequence from material output
    if isinstance(output_node, T.ShaderNodeOutputMaterial):
        group_node_seq = dfs_from_node(output_node)

    # Get active node sequence from masked group output (connected outputs by default)
    elif isinstance(output_node, T.NodeGroupOutput):
        if group_output_slot_ids is None:
            group_output_slot_ids = [s.identifier for s in node_group.outputs if s.is_linked]
        group_node_seq = dfs_from_node(output_node, input_slot_ids=group_output_slot_ids)

    # Ungroup all nodes by default
    else:
        group_node_seq = group_tree.nodes

    # Duplicate the active nodes, excluding group input/output nodes
    group_node_names = {n.name for n in group_node_seq}
    new_nodes_dict: dict[str, T.ShaderNode] = {}

    for node in group_node_seq:
        if isinstance(node, (T.NodeGroupInput, T.NodeGroupOutput)):
            continue

        # Check unsupported nodes
        if isinstance(node, T.NodeReroute):
            raise RuntimeError(
                f"Encountered reroute node while expanding node group "
                f"'{node_group.name}'"
            )
        if node.bl_idname == 'NodeUndefined':
            raise RuntimeError(
                f"Encountered undefined node while expanding node group "
                f"'{node_group.name}'"
            )

        # Create a new node of the same type
        new_node = node_tree.nodes.new(node.bl_idname)
        new_node.location = tuple(node.location)
        if isinstance(node, T.ShaderNodeGroup):
            new_node.node_tree = node.node_tree

        # Copy attributes and default values
        copy_node_attrs(new_node, node)
        new_nodes_dict[node.name] = new_node

    # Initialize the dictionary of emission nodes for shader conversion
    emission_nodes: dict[str, T.ShaderNode] = {}

    # Create links for the new nodes
    for link in group_tree.links:
        from_node, from_slot = link.from_node, link.from_socket
        to_node, to_slot = link.to_node, link.to_socket

        # Skip links from/to irrelevant nodes
        if from_node.name not in group_node_names or to_node.name not in group_node_names:
            continue

        # Detect destination sockets (skip masked output slots)
        if isinstance(to_node, T.NodeGroupOutput):
            group_output_slot = get_slot_by_id(node_group.outputs, to_slot.identifier)
            if group_output_slot.identifier not in group_output_slot_ids:
                continue
            dst_slots = [l.to_socket for l in group_output_slot.links]
        else:
            dst_slots = [get_slot_by_id(
                new_nodes_dict[to_node.name].inputs,
                to_slot.identifier
            )]

        # Skip if there are no destination sockets
        if not dst_slots:
            continue

        # Detect source sockets
        if isinstance(from_node, T.NodeGroupInput):
            group_input_slot = get_slot_by_id(node_group.inputs, from_slot.identifier)
            src_slots = [l.from_socket for l in group_input_slot.links]
        else:
            group_input_slot = None
            src_slots = [get_slot_by_id(
                new_nodes_dict[from_node.name].outputs,
                from_slot.identifier
            )]

        # Link source and destination sockets
        if src_slots:
            for dst_slot in dst_slots:
                for src_slot in src_slots:
                    node_tree.links.new(src_slot, dst_slot)

        # Transfer the default value to the destination sockets
        elif group_input_slot is not None:
            group_input_slot_type = group_input_slot.bl_idname[len('NodeSocket'):]
            shader_convertible = any(
                group_input_slot_type.startswith(t)
                for t in ('Float', 'Vector', 'Color')
            )

            for dst_slot in dst_slots:

                # Insert an emission node for color/value to shader conversion
                if isinstance(dst_slot, T.NodeSocketShader) and shader_convertible:
                    emission_node = emission_nodes.get(group_input_slot.identifier)
                    if emission_node is None:
                        emission_node = node_tree.nodes.new('ShaderNodeEmission')
                        emission_node.location = tuple(from_node.location)
                        transfer_socket_value(emission_node.inputs['Color'], group_input_slot)
                        emission_nodes[group_input_slot.identifier] = emission_node
                    node_tree.links.new(emission_node.outputs[0], dst_slot)

                # Transfer the default value to the connected socket
                else:
                    transfer_socket_value(dst_slot, group_input_slot)

    # Transfer the default values of group output slots
    if isinstance(output_node, T.NodeGroupOutput):
        for slot_id in group_output_slot_ids:
            output_node_slot = get_slot_by_id(output_node.inputs, slot_id)
            group_output_slot = get_slot_by_id(node_group.outputs, slot_id)
            if group_output_slot.is_linked:
                for dst_slot in (l.to_socket for l in group_output_slot.links):
                    transfer_socket_value(dst_slot, output_node_slot)

    # Remove the group node
    node_tree.nodes.remove(node_group)

    # Recursively ungroup node groups
    if recursive:
        for node in new_nodes_dict.values():
            if isinstance(node, T.ShaderNodeGroup):
                ungroup_node_group(node_tree, node, recursive=True)


def clean_node_tree(node_tree: T.ShaderNodeTree) -> T.ShaderNodeTree:
    '''Clean up a shader node tree by removing unnecessary nodes and replacing deprecated
    nodes.
    '''
    # Empty node tree
    if node_tree is None:
        return

    # Remove dot nodes
    remove_all_reroutes(node_tree)

    # Replace undefined (deprecated) nodes
    for node in node_tree.nodes:
        if node.bl_idname == 'NodeUndefined':
            replace_undefined_node(node_tree, node)

    # Remove constant value nodes
    # Value nodes tend to generate significantly varying values that can be hard to
    # sample. We thus manually assign the output value to the connected socket. This
    # step conducts necessary type conversion.
    value_nodes = [n for n in node_tree.nodes if isinstance(n, T.ShaderNodeValue)]

    for node in value_nodes:
        slot, emission_node = node.outputs[0], None
        for dst_slot in (l.to_socket for l in slot.links):

            # Insert an emission node for value to shader conversion
            if isinstance(dst_slot, T.NodeSocketShader):
                if emission_node is None:
                    emission_node = node_tree.nodes.new('ShaderNodeEmission')
                    emission_node.location = tuple(node.location)
                    transfer_socket_value(emission_node.inputs['Color'], slot)
                node_tree.links.new(emission_node.outputs[0], dst_slot)

            # Transfer the default value to the connected socket
            else:
                transfer_socket_value(dst_slot, slot)

        node_tree.nodes.remove(node)

    # Clean node groups
    for node in node_tree.nodes:
        if isinstance(node, T.ShaderNodeGroup) and node.node_tree is not None:
            clean_node_tree(node.node_tree)

    return node_tree


def find_output_node(
        node_tree: T.ShaderNodeTree, recursive: bool = True
    ) -> T.ShaderNodeOutputMaterial | None:
    '''Find the material output node in a shader node tree. When searching recursively,
    ungroup node groups if necessary.
    '''
    # Find the output node
    output_node = get_output_node(node_tree)
    if isinstance(output_node, T.ShaderNodeOutputMaterial):
        return output_node
    elif not recursive:
        return None

    # Search for the output node in node groups
    output_node_group = next((
        n for n in node_tree.nodes
        if (isinstance(n, T.ShaderNodeGroup)
            and n.node_tree is not None
            and find_output_node(n.node_tree) is not None)
    ), None)
    if output_node_group is None:
        return None

    # Expand the output node group
    ungroup_node_group(node_tree, output_node_group)
    output_node = get_output_node(node_tree)

    return output_node if isinstance(output_node, T.ShaderNodeOutputMaterial) else None


def expand_node_groups(node_tree: T.ShaderNodeTree, size_limit: int = 50) -> T.ShaderNodeTree:
    '''Expand node groups in a shader node tree until meeting a size limit. The node groups
    are expanded in a greedy manner, starting from the smallest ones.
    '''
    # Run DFS from the active output node
    output_node = get_output_node(node_tree)
    if not isinstance(output_node, T.ShaderNodeOutputMaterial):
        raise RuntimeError("No material output node found in the shader node tree")

    # Check the current size of the node tree
    node_seq = dfs_from_node(output_node)
    if len(node_seq) >= size_limit:
        return node_tree

    # Helper function to calculate the expanded size of a node group
    def get_expand_size(
            node_group: T.ShaderNodeGroup,
            group_output_slot_ids: list[str]
        ) -> int:
        # Get the group node tree and its output node
        group_tree = node_group.node_tree
        if group_tree is None:
            return 0

        output_node = get_output_node(group_tree)

        # Sanity check for output node type
        if not isinstance(output_node, T.NodeGroupOutput):
            raise RuntimeError(
                f"Group '{node_group.name}' does not have a valid output node"
            )

        # Get active node sequence from masked group output
        group_node_seq = dfs_from_node(output_node, input_slot_ids=group_output_slot_ids)
        group_node_names = {n.name for n in group_node_seq}

        # Calculate the expanded size of the group
        expand_size = len([
            n for n in group_node_seq
            if not isinstance(n, (T.NodeGroupInput, T.NodeGroupOutput))
        ])

        # Account for additional emission nodes due to color/value to shader conversion
        for node in (n for n in group_node_seq if isinstance(n, T.NodeGroupInput)):
            for slot in node.outputs:
                slot_type = slot.bl_idname[len('NodeSocket'):]
                if (any(slot_type.startswith(t) for t in ('Float', 'Vector', 'Color'))
                    and any(isinstance(l.to_socket, T.NodeSocketShader) for l in slot.links)):
                    expand_size += 1

        # Add the expanded sizes of node subgroups
        for node in (n for n in group_node_seq if isinstance(n, T.ShaderNodeGroup)):
            slot_ids = [
                s.identifier for s in node.outputs
                if any(l.to_node.name in group_node_names for l in s.links)
            ]
            expand_size += get_expand_size(node, group_output_slot_ids=slot_ids) - 1

        return expand_size

    # Collect node groups, expanded sizes, and output slot masks
    node_group_records: list[tuple[T.ShaderNodeGroup, int, list[bool]]] = []
    node_names = {n.name for n in node_seq}

    for node in (n for n in node_seq if isinstance(n, T.ShaderNodeGroup)):
        slot_ids = [
            s.identifier for s in node.outputs
            if any(l.to_node.name in node_names for l in s.links)
        ]
        expand_size = get_expand_size(node, group_output_slot_ids=slot_ids)
        node_group_records.append((node, expand_size, slot_ids))

    # Iteratively expand node groups
    while node_group_records:

        # Select the smallest node group
        node_group_records.sort(key=lambda x: x[1])
        node_group, expand_size, slot_ids = node_group_records.pop(0)
        if len(node_seq) + expand_size - 1 > size_limit:
            break

        # Expand the node group
        ungroup_node_group(
            node_tree, node_group,
            group_output_slot_ids=slot_ids, recursive=True
        )

        # Update the node sequence and node group records
        node_seq = dfs_from_node(output_node)
        node_names = {n.name for n in node_seq}

        ## Remove node groups no longer in the node sequence
        node_group_records = [
            r for r in node_group_records
            if r[0].name in node_names
        ]

        ## Recalculate the expanded sizes of node groups
        for i, (n, _, s) in enumerate(node_group_records):
            slot_ids = [
                s.identifier for s in n.outputs
                if any(l.to_node.name in node_names for l in s.links)
            ]
            if slot_ids != s:
                expand_size = get_expand_size(n, group_output_slot_ids=slot_ids)
                node_group_records[i] = (n, expand_size, slot_ids)

    return node_tree
