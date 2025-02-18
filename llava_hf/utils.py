# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

import os.path as osp
import subprocess


# System prompts for the chat dialog
SYSTEM_PROMPT_GPT = (
    "You are familiar with creating procedural materials using Blender's Python API. "
    "You will be given an image that describes a material appearance. Your task is to "
    "write a Python function `shader_material` that creates a Blender procedural material "
    "to match the appearance of the image when rendered on a flat surface. Write your code "
    "following the guidelines below.\n\n"
    "Code template:\n"
    "```python\nimport bpy\n\n"
    "def shader_material(material: bpy.types.Material):\n"
    "    material.use_nodes = True\n"
    "    nodes = material.node_tree.nodes\n"
    "    links = material.node_tree.links\n\n"
    "    # Create nodes\n"
    "    # YOUR CODE HERE\n\n"
    "    # Create links to connect nodes\n"
    "    # YOUR CODE HERE\n\n"
    "    # Set parameters for each node\n"
    "    # YOUR CODE HERE\n```\n\n"
    "Rules:\n"
    "1. Create no more than 30 nodes.\n"
    "2. Make sure your code can be correctly executed in Blender 3.3. Refer to the Blender "
    "Python API documentation for valid node types and parameters.\n"
    "3. Simply reply with code. Exclude any additional text or explanations.\n"
)

SYSTEM_PROMPT_LLAVA = (
    "Write a Python function `shader_material` that creates a Blender procedural material "
    "to match the appearance of the image when rendered on a flat surface. Use the code "
    "template below and exclude any explanation. Make sure your code executes correctly in "
    "Blender 3.3.\n\n"
    "Code template:\n"
    "```python\nimport bpy\n\n"
    "def shader_material(material: bpy.types.Material):\n"
    "    material.use_nodes = True\n"
    "    nodes = material.node_tree.nodes\n"
    "    links = material.node_tree.links\n\n"
    "    # Create nodes\n"
    "    # YOUR CODE HERE\n\n"
    "    # Create links to connect nodes\n"
    "    # YOUR CODE HERE\n\n"
    "    # Set parameters for each node\n"
    "    # YOUR CODE HERE\n```\n\n"
)


def log_info(log_path: str, info: str, print_stdout: bool = False):
    '''Log an info message to file and optionally print it to stdout.
    '''
    with open(log_path, 'a') as f:
        f.write(f'{info}\n')
    if print_stdout:
        print(info)


def check_stdout(
        stdout: str, file_path: str, test_id: int, log_path: str, print_stdout: bool = False
    ) -> bool:
    '''Detect errors from Blender stdout.
    '''
    err_str = ''

    # Check for Python errors
    if 'Error: Python:' in stdout:
        stdout = stdout[stdout.index('Error: Python:'):stdout.index('Blender quit')]
        err_str = f"Error when processing test case {test_id}:\n{stdout}\n"

    # Check for file existence
    elif not osp.isfile(file_path):
        file_type = (
            'Code' if file_path.endswith('.py')
            else 'Rendered image' if file_path.endswith('.jpg')
            else 'Analysis result' if file_path.endswith('.json')
            else f"File '{file_path}'"
        )
        err_str = f"Error when processing test case {test_id}:\n{file_type} not found\n{stdout}\n"

    # Log error
    if err_str:
        log_info(log_path, err_str, print_stdout=print_stdout)
        return True

    return False


def check_display_id(display_id: int):
    '''Check if the display ID is valid.
    '''
    # Run glxinfo to check the display ID
    env = {'DISPLAY': f':{display_id}'}
    ret = subprocess.run(['glxinfo'], capture_output=True, text=True, env=env)

    if not ret.stdout.startswith('name of display:') or 'NVIDIA Corporation' not in ret.stdout:
        raise ValueError(
            f"Failed to validate display ID ':{display_id}'. "
            f"Screen output:\n{ret.stdout}"
        )
