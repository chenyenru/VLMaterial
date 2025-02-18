# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

import os
import os.path as osp
import sys

# Align CUDA device order with X server
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from argparse import ArgumentParser, Namespace
from functools import partial
from PIL import Image
import re
import shutil
import subprocess

from lpips import LPIPS
from torch.multiprocessing import Process, Queue
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from utils import check_display_id
from vgg import VGGTextureDescriptor as VGGTD


# Directories
ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
SCRIPTS_DIR = osp.join(ROOT_DIR, 'data_scripts')

# Maximum seed value
MAX_SEED = 0x7FFFFFFF


def read_image(file_path: str, target_size: tuple[int, int] = (512, 512)) -> Image.Image:
    '''Read an image file.
    '''
    img = Image.open(file_path).convert('RGB')
    return img.resize(target_size) if img.size != target_size else img


def calc_lpips(
        lpips: LPIPS, input_img: torch.Tensor, pred_img: torch.Tensor
    ) -> list[float]:
    '''Calculate the LPIPS distance between the input and prediction images.
    '''
    return lpips(input_img, pred_img, normalize=True).tolist()


def calc_style(
        vgg_td: VGGTD, input_img: torch.Tensor, pred_img: torch.Tensor,
        input_td: torch.Tensor | None = None, ds_weight: float = 0.1
    ) -> list[float]:
    '''Calculate the style loss between the input and prediction images.
    '''
    # Calculate the texture descriptor loss
    input_td = input_td if input_td is not None else vgg_td(input_img)
    pred_td = vgg_td(pred_img)
    style_loss = F.l1_loss(input_td.expand_as(pred_td), pred_td, reduction='none').mean(dim=1)

    # Calculate the downsampled image loss
    input_ds = F.avg_pool2d(input_img, kernel_size=32, stride=32).flatten(1)
    pred_ds = F.avg_pool2d(pred_img, kernel_size=32, stride=32).flatten(1)
    ds_loss = F.l1_loss(input_ds.expand_as(pred_ds), pred_ds, reduction='none').mean(dim=1)

    style_loss += ds_loss * ds_weight
    return style_loss.tolist()


def run_command(cmd: list[str], env: dict[str, str] = None) -> str:
    '''Run a command, check for errors, and return the stdout.
    '''
    # Execute the command
    ret = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    stdout = ret.stdout

    # Check for Python errors
    err_header = 'Error: Python: '
    if err_header in stdout:
        stdout = stdout[stdout.index(err_header) + len(err_header):stdout.index('Blender quit')]
        i = next(
            i for i, line in enumerate(stdout.splitlines())
            if i and not line.startswith((' ', '\t'))
        )
        stdout = '\n'.join(stdout.splitlines()[:i + 1]).strip()
        raise RuntimeError(f'Blender Python error with the following message:\n{stdout}')

    return ret.stdout


def param_search_worker(
        task_queue: "Queue[tuple[str | None, int]]", result_queue: "Queue[str | None]",
        args: Namespace, worker_id: int, seed: int = 42
    ):
    '''Worker function for local parameter search.
    '''
    # Set the CUDA device
    device_id = args.device_id[worker_id % len(args.device_id)]
    device = f'cuda:{device_id}'

    # LPIPS metric
    if args.metric == 'lpips':
        lpips = LPIPS(net='vgg').requires_grad_(False).to(device)
        calc_metric = partial(calc_lpips, lpips)

    # Style metric
    elif args.metric == 'style':
        vgg_td = VGGTD().to(device)
        calc_metric = partial(calc_style, vgg_td)

    else:
        raise ValueError(f'Unknown metric: {args.metric}')

    # Create the seeding RNG
    seed_rng = np.random.default_rng(seed)
    prob_rng = np.random.default_rng(seed_rng.integers(MAX_SEED))

    # Environment variables for rendering
    env = {'DISPLAY': f':{args.display_id}.{worker_id}'}

    # Worker loop
    while True:
        # Get the next task and check for termination
        example_name, traj_id = task_queue.get()
        if example_name is None:
            break

        # Set the output folder
        output_folder = osp.join(args.output_folder, example_name)

        # Skip if the optimization result already exists
        if osp.isfile(osp.join(output_folder, 'best_render.jpg')) and not args.overwrite:
            result_queue.put(f'[{example_name} - traj {traj_id}] Skip existing results')
            continue

        # For the first trajectory, initialize the output folder
        if not traj_id:
            os.makedirs(output_folder, exist_ok=True)

            target_code_path = osp.join(output_folder, 'init_full.py')
            target_render_path = osp.join(output_folder, 'init_render.jpg')
            target_analysis_path = osp.join(output_folder, 'init_analysis_result.json')
            example_dir = osp.join(args.eval_dir, example_name)

            # Read predictions from the example directory
            pred_pattern = re.compile(r'pred_(\d+)_render.jpg')
            pred_img_paths = sorted([
                osp.join(example_dir, f)
                for f in os.listdir(example_dir) if pred_pattern.fullmatch(f)
            ])
            pred_imgs = torch.stack([
                TF.to_tensor(read_image(f)).to(device, non_blocking=True)
                for f in pred_img_paths
            ])

            # Read the input image
            input_img_path = osp.join(example_dir, 'input.jpg')
            input_img = TF.to_tensor(read_image(input_img_path)).to(device)[None]

            # Calculate metrics and get the best prediction
            pred_metrics = calc_metric(input_img, pred_imgs)
            best_ind = np.argmin(pred_metrics)
            best_pred, init_metric = pred_img_paths[best_ind], pred_metrics[best_ind]

            # Copy the input and best prediction to the output folder
            shutil.copy(input_img_path, osp.join(output_folder, 'input.jpg'))
            shutil.copy(best_pred, target_render_path)
            shutil.copy(
                osp.join(example_dir, osp.basename(best_pred).replace('_render.jpg', '_full.py')),
                target_code_path
            )

            # Run initial analysis
            run_command([
                args.blender_path, '-b', '-P', osp.join(SCRIPTS_DIR, 'analyze.py'),
                '--', osp.join(output_folder, 'analysis_result.json'), '--info_dir', args.info_dir,
                '--code_path', target_code_path, '-c', '--skip_curation'
            ])

            # Get the list of optimizable parameters
            ret = run_command([
                args.blender_path, '-b', '-P', osp.join(SCRIPTS_DIR, 'sample_params.py'),
                '--', target_code_path, '-m', 'opt', '-d', args.data_root, '-i', args.info_dir,
                '-o', output_folder, '--save-sampled-ids'
            ])

            # Print the number of optimizable parameters
            param_line = next((l for l in ret.splitlines() if l.startswith('Processed ')), '')
            if param_line:
                print(f'[{example_name} - traj {traj_id}] {param_line.strip()}')

            # Copy the initial analysis result
            shutil.copy(osp.join(output_folder, 'analysis_result.json'), target_analysis_path)

        # Calculate the initial score
        else:
            # Read the input and prediction images
            input_img_path = osp.join(output_folder, 'input.jpg')
            input_img = TF.to_tensor(read_image(input_img_path)).to(device)[None]
            pred_img_path = osp.join(output_folder, 'init_render.jpg')
            pred_img = TF.to_tensor(read_image(pred_img_path)).to(device)[None]

            init_metric = calc_metric(input_img, pred_img)[0]

        if args.verbose:
            print(f'[{example_name} - traj {traj_id}] Initial metric = {init_metric:.4g}')

        # Specify temporary file paths
        temp_code_path = osp.join(output_folder, 'temp_full.py')
        temp_render_path = osp.join(output_folder, 'temp_render.jpg')
        temp_analysis_path = osp.join(output_folder, 'temp_analysis_result.json')
        temp_files = temp_code_path, temp_render_path, temp_analysis_path

        # Prepare the current state
        code_path = osp.join(output_folder, 'cur_full.py')
        render_path = osp.join(output_folder, 'cur_render.jpg')
        analysis_path = osp.join(output_folder, 'analysis_result.json')
        cur_files = code_path, render_path, analysis_path

        best_metric = cur_metric = init_metric
        shutil.copy(osp.join(output_folder, 'init_full.py'), code_path)
        shutil.copy(osp.join(output_folder, 'init_render.jpg'), render_path)
        shutil.copy(osp.join(output_folder, 'init_analysis_result.json'), analysis_path)

        # Run the optimization
        for i in range(args.max_iter):
            # Run parameter search
            run_command([
                args.blender_path, '-b', '-P', osp.join(SCRIPTS_DIR, 'sample_params.py'),
                '--', code_path, '-m', 'opt', '-d', args.data_root, '-i', args.info_dir,
                '-o', output_folder, '-s', str(seed_rng.integers(MAX_SEED)),
                '--load-sampled-ids', osp.join(output_folder, 'sampled_ids.json')
            ])

            # Render the generated code
            run_command([
                args.blender_path, '-b', '-P', osp.join(SCRIPTS_DIR, 'render.py'),
                '--', '-c', temp_code_path, '-i', args.info_dir, '-o', temp_render_path
            ], env=env)

            # Read the rendered image and calculate the metric
            temp_img = TF.to_tensor(Image.open(temp_render_path)).to(device)[None]
            temp_metric = calc_metric(input_img, temp_img)[0]

            # Accept the new solution
            if temp_metric < cur_metric or prob_rng.random() < args.accept_prob:
                # Run analysis
                run_command([
                    args.blender_path, '-b', '-P', osp.join(SCRIPTS_DIR, 'analyze.py'),
                    '--', temp_analysis_path, '--info_dir', args.info_dir,
                    '--code_path', temp_code_path, '-c', '--skip_curation'
                ])

                # Update the current state
                cur_metric = temp_metric
                shutil.copy(temp_code_path, code_path)
                shutil.copy(temp_render_path, render_path)
                shutil.copy(temp_analysis_path, analysis_path)

            # Save intermediate results
            if not (i + 1) % args.save_interval:
                if args.verbose:
                    print(
                        f'[{example_name} - traj {traj_id}] Iteration {i + 1}: '
                        f'Current metric = {cur_metric:.4g}  Best metric = {best_metric:.4g}'
                    )

                # Copy the current state to intermediate files
                shutil.copy(code_path, osp.join(output_folder, f'traj{traj_id}_iter{i + 1:03d}_full.py'))
                shutil.copy(render_path, osp.join(output_folder, f'traj{traj_id}_iter{i + 1:03d}_render.jpg'))

            # Update the best solution
            if cur_metric < best_metric:
                best_metric = cur_metric
                shutil.copy(code_path, osp.join(output_folder, 'best_full.py'))
                shutil.copy(render_path, osp.join(output_folder, 'best_render.jpg'))

        # Finalize the optimization
        if args.verbose:
            print(
                f'[{example_name} - traj {traj_id}] Final metric = {cur_metric:.4g}  '
                f'Best metric = {best_metric:.4g}'
            )
        result_queue.put(
            f'[{example_name} - traj {traj_id}] Initial metric = {init_metric:.4g}  '
            f'Best metric = {best_metric:.4g}'
        )

        # Remove temporary files
        for f in temp_files + cur_files:
            if osp.isfile(f):
                os.remove(f)


def main():
    # Command line argument parser
    parser = ArgumentParser(description='Local parameter search')

    ## I/O paths
    parser.add_argument('eval_dir', type=str, help='Path to the evaluation directory')
    parser.add_argument('--blender_path', type=str,
                        default=osp.join(ROOT_DIR, 'infinigen', 'blender', 'blender'),
                        help='Path to Blender executable')
    parser.add_argument('--data_root', type=str,
                        default=osp.join(ROOT_DIR, 'material_dataset_filtered'),
                        help='Path to the material dataset root directory')
    parser.add_argument('--info_dir', type=str,
                        default=osp.join(ROOT_DIR, 'material_dataset_info'),
                        help='Path to the material dataset info directory')
    parser.add_argument('-o', '--output_folder', type=str, default='results_opt',
                        help='Output folder')

    ## Optimization settings
    parser.add_argument('--metric', type=str, default='style',
                        help='Optimization metric')
    parser.add_argument('-n', '--num_examples', type=int, default=0,
                        help='Number of examples to optimize (0 for all examples)')
    parser.add_argument('-m', '--max_iter', type=int, default=100,
                        help='Maximum number of iterations')
    parser.add_argument('-t', '--num_trajs', type=int, default=1,
                        help='Number of optimization trajectories')
    parser.add_argument('--accept_prob', type=float, default=0.05,
                        help='Acceptance probability for worse solutions')

    ## Other settings
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Interval for saving intermediate results')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='Number of parallel processes')
    parser.add_argument('--display_id', type=int, default=0, help='Display ID')
    parser.add_argument('--device_id', type=int, nargs='+', default=[0], help='Device IDs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')

    args = parser.parse_args()

    # Check the display ID
    check_display_id(args.display_id)

    # Scan the evaluation directory for optimization examples
    example_patten = re.compile(r'\d+(-[\w\.\(\)]+)*')
    example_names = sorted([
        d for d in os.listdir(args.eval_dir)
        if osp.isdir(osp.join(args.eval_dir, d)) and example_patten.fullmatch(d)
    ])

    # Remove examples without valid predictions
    example_names_filtered = []
    pred_pattern = re.compile(r'pred_\d+_full.py')

    for example_name in example_names:
        example_dir = osp.join(args.eval_dir, example_name)
        pred_files = [
            f for f in os.listdir(example_dir)
            if osp.isfile(osp.join(example_dir, f)) and pred_pattern.fullmatch(f)
        ]
        if pred_files:
            example_names_filtered.append(example_name)

    if args.num_examples > 0:
        example_names_filtered = example_names_filtered[:args.num_examples]
    if not example_names_filtered:
        print('No optimization examples found.')
        return

    # Put the examples into a queue
    task_queue: "Queue[tuple[str | None, int]]" = Queue()

    for example_name in example_names_filtered:
        for traj_id in range(args.num_trajs):
            task_queue.put((example_name, traj_id))

    # Add termination signals
    for _ in range(args.num_processes):
        task_queue.put((None, 0))

    # Create the result queue
    result_queue: "Queue[str | None]" = Queue()

    # Start the optimization processes
    seed_rng = np.random.default_rng(args.seed)

    processes = [
        Process(
            target=param_search_worker,
            args=(task_queue, result_queue, args, i, seed_rng.integers(MAX_SEED))
        )
        for i in range(args.num_processes)
    ]
    for p in processes:
        p.start()

    # Collect results with a progress bar
    with tqdm(total=len(example_names_filtered) * args.num_trajs, desc='Param search') as pbar:
        while pbar.n < pbar.total:
            result = result_queue.get()
            pbar.write(result)
            pbar.update()

    # Wait for all processes to finish
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
