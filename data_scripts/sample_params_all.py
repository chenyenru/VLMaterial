# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

import os.path as osp
import sys

THIS_DIR = osp.dirname(osp.abspath(__file__))
sys.path.append(THIS_DIR)

from argparse import Namespace
from multiprocessing import Process, Queue
from PIL import Image
import argparse
import glob
import os
import re
import subprocess

from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from numpy.random import Generator, default_rng
import numpy as np
import numpy.typing as npt

from param_stats import collect_param_stats
from utils import MAX_CODE_LEN, log_info, check_stdout, check_display_id, get_code_length


ROOT_DIR = osp.dirname(THIS_DIR)
MAX_SEED = 0x7FFFFFFF


def verify_code(
        code_path: str, tokenizer: PreTrainedTokenizer, test_id: int, log_path: str,
        rng: Generator | None = None
    ) -> str | None:
    '''Check and reduce the code length in tokens.
    '''
    # Read the code and check length
    with open(code_path, 'r') as f:
        code = f.read()
    if len(tokenizer.encode(code)) <= MAX_CODE_LEN:
        return code

    # Split the code into lines
    lines = code.split('\n')

    # Regular expressions for code reduction
    re_slot = re.compile(r'\w+\.(inputs|outputs)\[\d+\]\.default_value = ')
    re_color_stop = re.compile(r'\w+\.color_ramp\.elements\[\d+\]\.(position|color) = ')
    re_curve_point = re.compile(r'\w+\.curves\[\d+\]\.points\[\d+\]\.location = ')
    re_new_color_stop = re.compile(r'\w+\.color_ramp\.elements\.new(.*)')
    re_new_curve_point = re.compile(r'\w+\.curves\[\d+\]\.points\.new(.*)')

    # Detect lines of code about parameter value specifications, which can be removed
    remove_lines = []
    i, at_params = 0, False

    while i < len(lines):
        line = lines[i]

        # Locate the start of parameter value specifications
        if '# Set parameters for each node' in line:
            at_params = True
        if not at_params or not line:
            i += 1
            continue

        # New color ramp element or curve point
        if re_new_color_stop.match(line) or re_new_curve_point.match(line):
            remove_lines.append([line, lines[i + 1]])
            i += 2
            continue

        # Node slot default value
        if re_slot.match(line):
            remove_lines.append([line])

        # Node attribute
        elif not (re_color_stop.match(line) or re_curve_point.match(line)):
            remove_lines.append([line])

        i += 1

    # Iteratively remove lines of code
    rng = rng or default_rng()
    remove_lines = [remove_lines[i] for i in rng.permutation(len(remove_lines))]
    success = False

    for remove_line in remove_lines:
        for l in remove_line:
            lines.remove(l)
        code = '\n'.join(lines)
        if get_code_length(code, tokenizer, conv_version='v1.5-hf') <= MAX_CODE_LEN:
            success = True
            break

    # Log error
    if not success:
        err_str = f"Error when processing test case {test_id}:\nCode length too long\n"
        log_info(log_path, err_str)

    return code if success else None


def verify_render(
        render_path: str, prev_images: list[npt.NDArray[np.float32]], test_id: int, log_path: str,
        min_file_size: int = 12000, dup_pixel_threshold: float = 0.02, dup_image_threshold: float = 0.95
    ) -> npt.NDArray[np.float32] | None:
    '''Check the rendered image for degenerate or duplicate output.
    '''
    img, err_str = None, ''

    # Check image file size
    if osp.getsize(render_path) < min_file_size:
        err_str = f"Error when processing test case {test_id}:\nRendered image file size too small\n"

    # Check for duplicate images
    else:
        img = np.array(Image.open(render_path).convert('RGB')).astype(np.float32) / 255

        if prev_images:
            same = (np.abs(np.array(prev_images) - img) <= dup_pixel_threshold).all(axis=-1)
            if same.mean(axis=(1, 2)).max() > dup_image_threshold:
                err_str = f"Error when processing test case {test_id}:\nDuplicate image detected\n"

    # Log error
    if err_str:
        log_info(log_path, err_str)
        return None

    return img


def gen_variations(
        args: Namespace, file_path: str, data_root: str, info_dir: str, output_folder: str,
        tokenizer: PreTrainedTokenizer, worker_id: int, seed: int = 0
    ) -> str:
    '''Generate parameter variations for a Blender material file.
    '''
    # Create RNGs for this material
    seed_rng = np.random.default_rng(seed)
    shuf_rng = np.random.default_rng(seed + 1)

    # Extract material name
    if 'blenderkit' in file_path:
        mat_name = osp.basename(file_path)
        mat_name = mat_name[:mat_name.rindex('_')]
    else:
        mat_name = osp.basename(osp.dirname(file_path))

    # Create target folder
    target_folder = osp.join(output_folder, osp.dirname(file_path))
    os.makedirs(target_folder, exist_ok=True)

    # Initialize sampling state
    sample_id, start_test_id = 0, 0
    rendered_images = []

    # Read from existing files if not overwriting previous results
    stdout_path = osp.join(target_folder, 'sample_params_stdout.log')
    code_file_pattern = re.compile(r'var_(\d+)_full\.py')
    render_file_pattern = re.compile(r'var_(\d+)_render\.jpg')

    if not args.overwrite:

        # Read generated code
        code_files = sorted([f for f in os.listdir(target_folder) if code_file_pattern.fullmatch(f)])

        # Read rendered images
        for f in code_files:
            render_path = osp.join(target_folder, f.replace('_full.py', '_render.jpg'))
            if osp.isfile(render_path):
                try:
                    render_img = np.asarray(Image.open(render_path).convert('RGB')).astype(np.float32) / 255
                    rendered_images.append(render_img)
                except Exception:
                    print(f"[ERROR] Failed to read image: {render_path}. Skipping.")

        # Read failed samples from existing stdout log
        num_failed = 0

        if osp.isfile(stdout_path):
            error_header = 'Error when processing test case '
            with open(stdout_path, 'r') as f:
                error_lines = [l for l in f if l.startswith(error_header)]

            num_failed = len(error_lines)
            start_test_id = max((int(l[len(error_header):l.find(':')]) for l in error_lines), default=-1) + 1

        # Update sampling state
        sample_id = len(rendered_images)
        start_test_id = max(start_test_id, sample_id + num_failed)

        # Advance the seeding RNG state
        if start_test_id > 0:
            seed_rng.integers(0, MAX_SEED, size=start_test_id)

    # Clear existing data
    else:
        for f in os.listdir(target_folder):
            if any(p.fullmatch(f) for p in (code_file_pattern, render_file_pattern)):
                os.remove(osp.join(target_folder, f))

        open(stdout_path, 'w').close()

    # Set the display name for OpenGL rendering
    device_ids = args.device_id or list(range(args.num_workers))
    display_name = f':{args.display_id}.{device_ids[worker_id % len(device_ids)]}'

    # Start the main loop
    code_path = osp.join(target_folder, 'temp_full.py')
    render_path = osp.join(target_folder, 'temp_render.jpg')
    info_msg = ''

    for test_id in range(start_test_id, args.max_samples):

        # Check if the number of samples has been reached
        if sample_id >= args.num_samples:
            info_msg = f'Finished at {test_id} samples'
            break

        # Check if the number of samples is too small
        elif test_id == args.num_samples * 2 and sample_id < args.num_samples ** 2 / args.max_samples:
            info_msg = f'Too few samples after {test_id} trials'
            break

        # Remove existing temporary files
        for temp_file in (code_path, render_path):
            if osp.exists(temp_file):
                os.remove(temp_file)

        # Generate a new parameter sample
        ret = subprocess.run([
            args.blender_path, '-b', '-P', osp.join(THIS_DIR, 'sample_params.py'),
            '--', osp.join(data_root, file_path), '-d', data_root, '-i', info_dir, '-o', target_folder,
            '-s', str(seed_rng.integers(0, MAX_SEED))
        ], capture_output=True, text=True, env={'DISPLAY': display_name})

        # Check code generation result
        if check_stdout(ret.stdout, code_path, test_id, stdout_path, print_stdout=False):
            continue

        # Reduce code length if necessary
        code = verify_code(code_path, tokenizer, test_id, stdout_path, rng=shuf_rng)
        if not code:
            continue
        with open(code_path, 'w') as f:
            f.write(code)

        # Render the code to image
        ret = subprocess.run([
            args.blender_path, '-b', '-P', osp.join(THIS_DIR, 'render.py'), '--', '-c', code_path,
            '-i', info_dir, '-o', render_path,
        ], capture_output=True, text=True, env={'DISPLAY': display_name})

        # Check render result
        if check_stdout(ret.stdout, render_path, test_id, stdout_path, print_stdout=False):
            continue

        # Validate the rendered image
        orig_render_path = osp.join(data_root, osp.dirname(file_path), 'transpiled_render.jpg')
        img = verify_render(
            render_path, rendered_images, test_id, stdout_path,
            max(args.min_file_size, int(osp.getsize(orig_render_path) * args.min_file_size_ratio)),
            args.dup_pixel_threshold, args.dup_image_threshold
        )
        if img is None:
            continue

        # Rename the temporary files using the sample ID
        os.rename(code_path, osp.join(target_folder, f'var_{sample_id:05d}_full.py'))
        os.rename(render_path, osp.join(target_folder, f'var_{sample_id:05d}_render.jpg'))

        # Update the sampling state
        sample_id += 1
        rendered_images.append(img)

    else:
        info_msg = f'Generated {sample_id} samples'

    # Remove temporary files
    for temp_file in (code_path, render_path):
        if osp.exists(temp_file):
            os.remove(temp_file)

    return info_msg


def worker_func(
        task_queue: "Queue[tuple[str | None, int]]", result_queue: "Queue[tuple[str, str]]", args: Namespace,
        worker_id: int
    ):
    '''Worker function for generating parameter variations.
    '''
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    while True:
        file_path, material_seed = task_queue.get()

        # Catch the termination signal
        if file_path is None:
            break

        # Generate parameter variations
        info_msg = gen_variations(
            args, file_path, args.data_root, args.info_dir, args.output_folder,
            tokenizer, worker_id, seed=material_seed
        )
        result_queue.put((file_path, info_msg))


def main():
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Generate parameter variations for Blender materials')

    ## I/O parameters
    parser.add_argument('--blender_path', type=str,
                        default=osp.join(ROOT_DIR, 'infinigen', 'blender', 'blender'),
                        help='Path to Blender executable')
    parser.add_argument('--data_root', type=str, default=osp.join(ROOT_DIR, 'material_dataset_filtered'),
                        help='Root directory of Blender files')
    parser.add_argument('--info_dir', type=str, default=osp.join(ROOT_DIR, 'material_dataset_info'),
                        help='Directory to analysis results')
    parser.add_argument('-o', '--output_folder', type=str, default=osp.join(ROOT_DIR, 'material_dataset_filtered'),
                        help='Output directory')

    ## Parameter sampling settings
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-n', '--num_samples', type=int, default=10, help='Number of samples per material')
    parser.add_argument('-m', '--max_samples', type=int, default=50, help='Maximum number of trials')
    parser.add_argument('-u', '--update_stats', action='store_true', help='Update parameter statistics')
    parser.add_argument('-w', '--overwrite', action='store_true', help='Overwrite previous results')

    ## Sample verification settings
    parser.add_argument('--tokenizer', type=str, default='llava-hf/llama3-llava-next-8b-hf',
                        help='Tokenizer URL to use')
    parser.add_argument('--min_file_size', type=int, default=12000,
                        help='Minimum rendered image file size in bytes')
    parser.add_argument('--min_file_size_ratio', type=float, default=0.4,
                        help='Minimum rendered image file size ratio relative to original file size')
    parser.add_argument('--dup_pixel_threshold', type=float, default=0.05,
                        help='Pixel value difference threshold for duplicate image detection')
    parser.add_argument('--dup_image_threshold', type=float, default=0.9,
                        help='Pixel percentage threshold for duplicate image detection')
    parser.add_argument('--log_path', type=str, default='sample_params.log', help='Log file path')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--display_id', type=int, default=0, help='Display ID for rendering')
    parser.add_argument('--device_id', type=int, nargs='+', default=None,
                        help='Device ID for rendering')

    args = parser.parse_args()

    data_root = args.data_root
    info_dir = args.info_dir
    output_folder = args.output_folder

    # Check display ID
    check_display_id(args.display_id)

    # Set random seed
    seed_rng = np.random.default_rng(args.seed)

    # Collect parameter statistics
    stats_file = osp.join(output_folder, 'param_stats.json')
    if not osp.isfile(stats_file) or args.update_stats:
        collect_param_stats(data_root, info_dir, stats_file)

    # Gather source files
    all_files = sorted(glob.glob(osp.join("*", "*", "blender_full.py"), root_dir=data_root))

    # Create multiprocessing queues and workers
    task_queue: "Queue[tuple[str | None, int]]" = Queue()
    result_queue: "Queue[tuple[str, str]]" = Queue()
    workers = [
        Process(target=worker_func, args=(task_queue, result_queue, args, i))
        for i in range(args.num_processes)
    ]

    # Add initial tasks to the queue
    task_iter = ((file_path, seed_rng.integers(0, MAX_SEED)) for file_path in all_files)

    for _ in range(args.num_processes * 2):
        next_task = next(task_iter, (None, -1))
        if next_task[0] is not None:
            task_queue.put(next_task)

    # Start workers
    for worker in workers:
        worker.start()

    # Collect results from workers and insert new tasks
    pbar = tqdm(total=len(all_files), desc='Generating samples')
    results = []

    for _ in range(len(all_files)):
        file_path, info_msg = result_queue.get()
        results.append((file_path, info_msg))

        # Update progress bar
        pbar.write(f'{file_path}: {info_msg}')
        pbar.update()

        # Insert a new task
        next_task = next(task_iter, (None, -1))
        if next_task[0] is not None:
            task_queue.put(next_task)

    pbar.close()

    # Terminate workers
    for _ in range(args.num_processes):
        task_queue.put((None, -1))
    for worker in workers:
        worker.join()

    # Write results to log file
    with open(osp.join(output_folder, args.log_path), 'w') as f:
        f.write('Case,Info\n')
        for file_path, info_msg in sorted(results):
            f.write(f'{file_path},{info_msg}\n')


if __name__ == '__main__':
    main()
