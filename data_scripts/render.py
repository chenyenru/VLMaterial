# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

import argparse
import json
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.abspath(__file__)))

import bpy
import bpy.types as T

from analyze import clean_node_tree, get_output_node, dfs_from_node

ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))


def setup_plane() -> T.Object:
    '''Clear the scene and set up a plane for rendering.
    '''
    # Delete all existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Set render engine to Eevee
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    # bpy.context.scene.render.engine = 'CYCLES'

    # Set render device to GPU
    # cycles_pref = bpy.context.preferences.addons['cycles'].preferences
    # cycles_pref.compute_device_type = 'CUDA'

    # Set environment light
    if bpy.context.scene.world is not None:
        # Delete all existing world
        bpy.data.worlds.remove(bpy.context.scene.world, do_unlink=True)

    bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.1, 0.1, 0.1, 1)

    # Add plane
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

    # Use the fisrt material
    material = bpy.data.materials[0]
    clean_node_tree(material.node_tree)
    bpy.data.objects['Plane'].data.materials.append(material)

    # Add camera
    if 'Camera' in bpy.data.objects.keys():
        bpy.data.objects.remove(bpy.data.objects['Camera'], do_unlink=True)
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 2.78), rotation=(0, 0, 0))
    bpy.context.scene.camera = bpy.data.objects['Camera']

    # Set render resolution
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

    # Add point light
    bpy.ops.object.light_add(type='POINT', radius=0, align='WORLD', location=(0, 0, 2.78), scale=(1, 1, 1))
    bpy.data.objects['Point'].data.energy = 500

    # Set view layer use for rendering
    if "RenderLayer" in bpy.context.scene.view_layers:
        bpy.context.scene.view_layers["RenderLayer"].use = True

    return bpy.data.objects['Plane']


def apply_material_file(obj: T.Object, src_file: str) -> T.Material:
    '''Apply a material from an external Blender file to the object.
    '''
    # Delete all existing materials
    for material in obj.data.materials:
        if material is not None:
            bpy.data.materials.remove(material)

    # Load the material from the source file
    with bpy.data.libraries.load(src_file, link=False) as (data_from, data_to):
        data_to.materials = [m for m in data_from.materials if m is not None]

    # Clean up the material
    material = data_to.materials[0]
    clean_node_tree(material.node_tree)

    # Assign the material to the object
    obj.active_material = material
    return material


def apply_material_code(
        obj: T.Object, code: str, load_node_groups: bool = True,
        node_groups_dir: str | None = None, clean_materials: bool = True
    ) -> T.Material:
    '''Apply the material code to the object.
    '''
    # Clean up the materials if needed
    if clean_materials:

        # Delete all existing materials
        for material in obj.data.materials:
            if material is not None:
                bpy.data.materials.remove(material)

        # Delete all existing node groups
        for node_group in bpy.data.node_groups:
            if node_group is not None:
                bpy.data.node_groups.remove(node_group)

    # Load external node groups if needed
    if load_node_groups and node_groups_dir is not None:

        # Collect node group references in the code
        node_group_types = set()
        for line in code.splitlines():
            if 'bpy.data.node_groups' in line:
                node_group_types.add(line.split('\'')[1])

        # Load node groups
        for ng_type in node_group_types:
            with open(osp.join(node_groups_dir, f'{ng_type}.json'), 'r') as f:
                ng_info = json.load(f)
                ng_source, ng_name = ng_info['source_file'], ng_info['source_name']

                # Monkey patch in case the source Blender file is on a different machine
                if not osp.exists(ng_source):
                    ng_source = osp.join(ROOT_DIR, ng_source[ng_source.index('material_dataset'):])

            with bpy.data.libraries.load(ng_source, link=False) as (_, data_to):
                data_to.node_groups = [ng_name]

            # Clean up the node group
            node_tree = bpy.data.node_groups[ng_name]
            clean_node_tree(node_tree)

            # Rename the node group to the type name
            node_tree.name = ng_type

    # Create a new material
    material = bpy.data.materials.new(name='Material')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

    # Apply the material code
    namespace = {'bpy': bpy}
    exec(code, namespace)

    # Update the material node tree
    material_func = next((v for f, v in namespace.items() if f.startswith('shader_material')))
    material_func(material)

    # Assign the material to the object
    obj.active_material = material
    return material


def render_scene(output_path: str):
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def sort_node_tree(node_tree: T.ShaderNodeTree):
    '''Sort the nodes in the node tree and re-arrange the layout.
    '''
    # Get the output node
    output_node = get_output_node(node_tree)
    if output_node is None:
        return

    # Perform a depth-first search from the output node
    sorted_nodes = dfs_from_node(output_node)

    # Re-arrange the nodes into a grid layout
    grid_cols = min(6, len(sorted_nodes))
    grid_rows = (len(sorted_nodes) + grid_cols - 1) // grid_cols

    for i, node in enumerate(sorted_nodes):
        node.location.x = 200 * (i % grid_cols)
        node.location.y = -200 * (i // grid_cols)


def main():
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Render a material on a plane')
    parser.add_argument('-f', '--file', type=str, default=None, help='Material Blender file path')
    parser.add_argument('-c', '--code', type=str, default=None, help='Material code path')
    parser.add_argument('-i', '--info_dir', type=str, default='material_dataset_info',
                        help='Directory to analysis results')
    parser.add_argument('-s', '--save-path', type=str, default=None, help='Path to save the Blender file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output file path')
    parser.add_argument('--image-format', type=str, default='jpg', choices=['jpg', 'png'],
                        help='Output image format')
    parser.add_argument('--sort-nodes', action='store_true', help='Sort nodes in the node tree')

    argv = sys.argv
    argv = argv[argv.index('--') + 1:]
    args = parser.parse_args(argv)

    # Set up the scene
    obj = setup_plane()

    # Change the output image format
    if args.image_format == 'png':
        bpy.context.scene.render.image_settings.file_format = 'PNG'

    # Apply the material file if given
    if args.file is not None:
        mat = apply_material_file(obj, args.file)

    # Apply the material code if given
    elif args.code is not None:
        with open(args.code, 'r') as f:
            code = f.read()
        node_groups_dir = osp.join(args.info_dir, 'node_types', 'node_groups')
        mat = apply_material_code(obj, code, node_groups_dir=node_groups_dir)

    else:
        mat = bpy.data.materials[0]

    # Render the material on a plane
    render_scene(args.output)

    # Save the Blender file
    if args.save_path:

        # Remove all objects
        # bpy.ops.object.select_all(action='SELECT')
        # bpy.ops.object.delete()

        # Remove all materials but the one used
        for material in bpy.data.materials:
            if material is not mat:
                bpy.data.materials.remove(material)

        # Sort the nodes in the node tree
        if args.sort_nodes:
            sort_node_tree(mat.node_tree)

        bpy.ops.wm.save_as_mainfile(filepath=args.save_path)


if __name__ == "__main__":
    main()
