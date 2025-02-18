# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

from PIL import Image
from typing import Any
import itertools
import json
import os.path as osp
import re
import subprocess

import numpy as np
import numpy.typing as npt


# Maximum float value
FLOAT_MAX = 1e4

# Filtering parameters
BW_THRESHOLD = 0.9
BW_PIXEL_THRESHOLD = 0.01
UNIFORM_THRESHOLD = 0.05
MAX_CODE_LEN = 2048

# Input text prompt
PROMPT = (
    'Write a Python function with Blender API to create a material node graph '
    'for this image.'
)


class NodeSignatureReader:
    '''Read node type signatures with caching.
    '''
    def __init__(self, node_type_dir: str):
        self.src_dir = node_type_dir
        self.cache: dict[str, dict[str, Any]] = {}

        # Read supplementary default parameter ranges
        self.supp_param_ranges: dict[str, dict[str, tuple[float, float]]] = {}
        file_path = osp.join(osp.dirname(osp.abspath(__file__)), 'default_ranges.json')
        try:
            with open(file_path, 'r') as f:
                self.supp_param_ranges.update(json.load(f))
        except FileNotFoundError:
            pass

    def read(self, node_type: str, group_type: str | None = None) -> dict[str, Any]:
        # Read node type information from cache
        if node_type != 'ShaderNodeGroup':
            sig_key = node_type
        elif group_type is not None:
            sig_key = osp.join('node_groups', group_type)
        else:
            raise ValueError(f"Group type must be specified for 'ShaderNodeGroup' nodes")

        if sig_key in self.cache:
            return self.cache[sig_key]

        # Read node type signature
        file_path = osp.join(self.src_dir, f'{sig_key}.json')
        if not osp.isfile(file_path):
            raise FileNotFoundError(
                f"Node type signature not found for '{sig_key}' at '{file_path}'"
            )

        with open(file_path, 'r') as f:
            sig = json.load(f)

        # Add supplementary default parameter ranges
        supp_ranges = self.supp_param_ranges.get(sig_key, {})
        io_type = 'output' if node_type in ('ShaderNodeValue', 'ShaderNodeRGB') else 'input'

        for param in itertools.chain(sig[io_type], sig.get('attr', [])):
            supp_range = supp_ranges.get(param['name'])
            if isinstance(supp_range, list):
                if isinstance(supp_range[0], (int, float)):
                    param['min'], param['max'] = supp_range
                elif isinstance(supp_range[0], str):
                    param['enum'] = supp_range
            elif supp_range == 'unused':
                param['min'], param['max'] = 0.0, 0.0

        # Cache the node type signature
        self.cache[sig_key] = sig

        return sig


def translate_name(name: str) -> str:
    '''Translate a node group name to a valid Python variable name.
    '''
    name = re.sub(r'[^\w]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')

    # Further distinguish node group names
    if name == 'NodeGroup' or name.startswith('NodeGroup_'):
        import bpy
        file_name = osp.splitext(osp.basename(bpy.data.filepath))[0]
        file_name = '_'.join([s for s in file_name.split('_') if s and len(s) <= 20])
        name = name.replace('NodeGroup', file_name, 1)

    return name


def get_analysis_url(file_path: str | None = None, info_dir: str | None = None) -> str:
    '''Get the URL of the analysis result for the current material.
    '''
    # Get dataset name and material name
    if file_path is None:
        import bpy
        file_path = bpy.data.filepath
    dataset_name = osp.basename(osp.dirname(osp.dirname(file_path)))

    if dataset_name == 'infinigen' or dataset_name.startswith('mat_llm'):
        material_name = osp.basename(osp.dirname(file_path))
    else:
        material_name = osp.splitext(osp.basename(file_path))[0]

    # Get analysis result URL
    url = osp.join(dataset_name, f'{material_name}.json')
    if info_dir:
        url = osp.join(info_dir, url)

    return url


def get_code_length(code: str, tokenizer: Any, conv_version: str = 'v1.5') -> int:
    '''Get the length of the transpiled code in tokens.
    '''
    # Github version of LLaVA v1.5 7B (Vicuna)
    if conv_version == 'v1.5':
        prompt = (
            f"A chat between a curious user and an artificial intelligence assistant. "
            f"The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: <image>\n{PROMPT} ASSISTANT: ```python\n{code.strip()}```</s>"
        )

    # Hugging Face versions of LLaVA
    elif conv_version == 'v1.5-hf':
        conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': PROMPT},
                    {'type': 'image'}
                ]
            },
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': f'```python\n{code.strip()}'}
                ]
            }
        ]
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)

    # Unknown version
    else:
        raise ValueError(f"Unknown conversation version '{conv_version}'")

    return len(tokenizer.encode(prompt))


def check_render(render_path: str, target_size: int | None = None) -> tuple[bool, str]:
    '''Check the rendered image for filtering.
    '''
    # Check rendered image
    img = np.array(Image.open(render_path), dtype=np.float32) / 255
    if img.shape[-1] == 4:
        img = img[..., :3]

    ## Black and white
    if max((img.max(axis=-1) < BW_PIXEL_THRESHOLD).mean(),
           (img.min(axis=-1) > 1 - BW_PIXEL_THRESHOLD).mean()) > BW_THRESHOLD:
        return False, 'Rendered image is black or white'

    ## Uniform
    if (img.max(axis=(0, 1)) - img.min(axis=(0, 1)) < UNIFORM_THRESHOLD).all():
        return False, 'Rendered image is uniform'

    # Check image file size
    if target_size is not None and osp.getsize(render_path) < target_size:
        return False, 'Rendered image file size too small'

    return True, ''


def check_file(
        blender_path: str, file_path: str, target_folder: str, info_dir: str,
        tokenizer: Any, target_size: int | None = None, conv_version: str = 'v1.5',
        transpile_script_path: str | None = None,
        render_script_path: str | None = None
    ) -> tuple[bool, str]:
    # Check if the file has been analyzed
    analysis_path = get_analysis_url(file_path, info_dir=info_dir)
    if not osp.isfile(analysis_path):
        return False, 'Analysis result not found'

    # Transpile source material
    code_path = osp.join(target_folder, 'blender_full.py')
    subprocess.run([
        blender_path, file_path, '-b', '-P', transpile_script_path or 'transpile.py',
        '--', code_path, '-i', info_dir
    ])
    if not osp.exists(code_path):
        return False, 'Transpiler failure'

    # Check code length
    with open(code_path, 'r') as f:
        code = f.read().strip()
    if 'ShaderNodeTexImage' in code:
        return False, 'Code contains image texture'
    if get_code_length(code, tokenizer, conv_version) > MAX_CODE_LEN:
        return False, 'Code too long'

    # Render source material
    render_path = osp.join(target_folder, 'load_render.jpg')
    subprocess.run([
        blender_path, '-b', '-P', render_script_path or 'render.py',
        '--', '-f', file_path, '-o', render_path
    ])
    if not osp.exists(render_path):
        return False, 'Render failure'

    # Render transpiled material
    transpiled_path = osp.join(target_folder, 'transpiled_render.jpg')
    subprocess.run([
        blender_path, '-b', '-P', render_script_path or 'render.py',
        '--', '-c', code_path, '-i', info_dir, '-o', transpiled_path
    ])
    if not osp.exists(transpiled_path):
        return False, 'Code render failure'

    # Check rendered image
    return check_render(transpiled_path, target_size=target_size)


def make_grid(
        grid_imgs: list[list[npt.NDArray[np.uint8]]], pad: int = 4, pad_value: int = 255,
        margin: int | None = None
    ) -> npt.NDArray[np.uint8]:
    '''Make a grid image from a nested list of images.
    '''
    # Check image sizes
    H, W = grid_imgs[0][0].shape[:2]
    for row_imgs in grid_imgs:
        for img in row_imgs:
            if img.shape[:2] != (H, W):
                raise ValueError(f'Image sizes do not match ({img.shape[:2]} vs. {H, W})')

    # Calculate margin
    margin = margin if margin is not None else pad

    # Make grid image
    num_rows = len(grid_imgs)
    num_cols = max(len(row_imgs) for row_imgs in grid_imgs)
    grid_img = np.full(
        (num_rows * (H + pad) - pad + margin * 2, num_cols * (W + pad) - pad + margin * 2, 3),
        pad_value, dtype=np.uint8
    )

    for i, row_imgs in enumerate(grid_imgs):
        for j, img in enumerate(row_imgs):
            h_pos, w_pos = i * (H + pad) + margin, j * (W + pad) + margin
            grid_img[h_pos:h_pos + H, w_pos:w_pos + W] = img

    return grid_img


def log_info(log_path: str, info: str, print_stdout: bool = False):
    '''Log an info message to file and optionally print it to stdout.
    '''
    with open(log_path, 'a') as f:
        f.write(info)
    if print_stdout:
        print(info)


def check_stdout(
        stdout: str, file_path: str, test_id: int, log_path: str, print_stdout: bool = True
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
