# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

from argparse import ArgumentParser, Namespace
from asyncio import Queue as AsyncQueue
from dataclasses import dataclass
from multiprocessing import Process, Queue
from PIL import Image
import asyncio
import glob
import os
import os.path as osp
import shutil
import subprocess
import time

from numpy.random import default_rng
from numpy.typing import NDArray
from openai import AsyncOpenAI, RateLimitError
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np

from utils import MAX_CODE_LEN, log_info, check_stdout, get_code_length, check_display_id


THIS_DIR = osp.dirname(osp.abspath(__file__))
ROOT_DIR = osp.dirname(THIS_DIR)
MAX_SEED = 0x7FFFFFFF

# System prompts for the chat dialog
SYSTEM_PROMPT = (
    "You are a helpful assistant familiar with Blender and python. Given two python functions "
    "that create procedural materials in Blender, use elements from these two functions and "
    "write a new function (`shader_material`) that creates a new material. Use only node types "
    "and parameter fields that already exist in the provided functions."
)

SYSTEM_PROMPT_V1 = (
    "You are familiar with creating procedural materials using Blender's Python API. "
    "You will be given a Python code block delimited by triple backticks, which contains "
    "two functions that define the node trees of two procedural materials in Blender. "
    "The functions are named `shader_material_1` and `shader_material_2`, respectively. "
    "Your task is to write a Python function `shader_material` that creates a new material "
    "by combining elements from the two provided functions. Your code should conform to "
    "the guidelines below.\n\n"
    "Hints:\n"
    "1. Write the code in the same format as the examples, using no more than 30 nodes.\n"
    "2. Only use node types and parameter fields referenced in the two functions.\n"
    "3. Ensure that your code is syntactically correct and can be executed in Blender 3.3.\n"
    "4. Avoid generating materials that look too trivial or similar to the provided examples.\n\n"
    "Code template:\n"
    "```python\nimport bpy\n\n"
    "def shader_material(material: bpy.types.Material):\n"
    "    # Your code here\n```"
)

SYSTEM_PROMPT_V2 = (
    "You are familiar with creating procedural materials using Blender's Python API. "
    "You will be given a Python code block delimited by triple backticks, which contains "
    "two functions that define the node trees of two procedural materials in Blender. "
    "The functions are named `shader_material_1` and `shader_material_2`, respectively. "
    "Your task is to write a Python function `shader_material` that creates a new material "
    "based on the provided functions. Write your code following the guidelines below.\n\n"
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
    "1. Create no more than 30 nodes. Only use node types and parameter fields referenced in "
    "the provided functions. Change parameter values as needed.\n"
    "2. Make sure your code can be correctly executed in Blender 3.3. Refer to the Blender "
    "Python API documentation for valid node types and parameters.\n"
    "3. Try to generate materials with complex and semantically meaningful appearances. Avoid "
    "generating materials that lack structure or look too similar to the provided examples.\n"
    "4. Follow the format and coding style of the example functions. Do not add new code "
    "blocks or comments.\n"
    "5. Simply reply with code. Exclude any additional text or explanations.\n"
)

# Global OpenAI client (use your own API credentials)
client = AsyncOpenAI(
    api_key="[API_KEY]",
    organization="[ORGANIZATION]"
)


@dataclass
class TaskInfo:
    '''Information about a generation task.
    '''
    case_id: int                # Case ID
    sample_id: int              # Sample ID
    return_id: int              # Registered sample ID
    files: tuple[str, str]      # Source files
    output_folder: str          # Output folder for generated materials


class TaskManager:
    '''Manage asynchronous program generation tasks.
    '''
    def __init__(self, dataset: list[str], args: Namespace):
        self.dataset = dataset
        self.args = args
        self.reset()

        # Resume from previous run
        if args.resume:
            self.resume(args.output_folder)

    def reset(self):
        '''Initialize or reset the task manager's state.
        '''
        N = self.args.num_cases

        # Initialize the task status arrays
        self.total_samples = [0] * N        # Total samples generated
        self.passed_samples = [0] * N       # Samples passed the quality check
        self.in_progress = [False] * N      # Generation/rendering in progress

        # Initialize the LLM prompt cache
        self.prompt_cache: dict[int, list[dict[str, str]]] = {}

        # Pre-sample candidates for each case
        rng = default_rng(self.args.seed)
        self.cand_inds = [
            rng.choice(len(self.dataset), size=2, replace=False).tolist()
            for _ in range(N)
        ]

        # Token usage counters
        self.token_usage: list[int] = [0, 0]

    def resume(self, output_folder: str):
        '''Resume the task manager from a previous run.
        '''
        # Load the task status of each case
        for case_id in range(self.args.num_cases):
            case_folder = osp.join(output_folder, f'case_{case_id:05d}')

            # Skip cases whose output folders haven't been created
            if not osp.isdir(case_folder):
                self.total_samples[case_id] = 0
                self.passed_samples[case_id] = 0
                continue

            # Find the total and passed samples
            stdout_file = osp.join(case_folder, 'gen_programs_stdout.log')
            passed = len(glob.glob(osp.join(case_folder, 'gen_*_render.jpg')))
            total = passed

            ## Read the log file to find failed samples
            if osp.isfile(stdout_file):
                with open(stdout_file, 'r') as f:                    
                    total += len([
                        l for l in f
                        if l.startswith('Error when processing test case')
                    ])

            # Update the task status
            self.total_samples[case_id] = total
            self.passed_samples[case_id] = passed

        # Print the progress of the resumed task manager
        print('Resuming task manager from previous run...')
        self.show_progress()

    def get_next_task(self, index: int | None = None) -> TaskInfo | None:
        '''Get the next available task to process. If no case index is provided,
        return one with the smallest case ID that has available tasks. Return None
        if no tasks are available at the moment.
        '''
        args = self.args
        num_samples, max_samples = args.num_samples, args.max_samples
        output_folder = args.output_folder

        # Find the next case with available tasks
        if index is None:
            next_index_iter = (
                i for i, (total, passed, in_prog) in enumerate(
                    zip(self.total_samples, self.passed_samples, self.in_progress)
                ) if total < max_samples and passed < num_samples and not in_prog
            )
            index = next(next_index_iter, -1)
            if index < 0:
                return None

        # Verify the selected case has available tasks
        total = self.total_samples[index]
        passed = self.passed_samples[index]
        in_prog = self.in_progress[index]

        if total >= max_samples or passed >= num_samples or in_prog:
            return None

        # Get the next task for the selected case
        return TaskInfo(
            case_id=index,
            sample_id=total,
            return_id=passed,
            files=tuple(self.dataset[i] for i in self.cand_inds[index]),
            output_folder=osp.join(output_folder, f'case_{index:05d}')
        )

    def start_task(self, index: int):
        '''Mark a task as in progress.
        '''
        self.in_progress[index] = True

    def complete_task(self, index: int, success: bool):
        '''Mark a task as completed.
        '''
        self.in_progress[index] = False
        self.total_samples[index] += 1
        self.passed_samples[index] += success

    def get_prompt(self, index: int) -> str | None:
        '''Get the cached LLM prompt for a case.
        '''
        return self.prompt_cache.get(index)

    def cache_prompt(self, index: int, prompt: str):
        '''Cache the LLM prompt for a case.
        '''
        self.prompt_cache[index] = prompt

    def num_done(self) -> int:
        '''Return the number of completed cases.
        '''
        args = self.args
        return sum(
            (total >= args.max_samples or passed >= args.num_samples) and not in_prog
            for total, passed, in_prog in zip(
                self.total_samples, self.passed_samples, self.in_progress
            )
        )

    def all_done(self) -> bool:
        '''Check if all cases are completed.
        '''
        return self.num_done() >= self.args.num_cases

    def report_token_usage(self, input_count: int, output_count: int):
        '''Report the token usage for a generated program.
        '''
        self.token_usage[0] += input_count
        self.token_usage[1] += output_count

    def _f(self, x: int) -> str:
        '''Format an integer using K, M, G units.
        '''
        if x < 10 ** 3:
            return str(x)
        elif x < 10 ** 6:
            return f"{x / 1e3:.2f}K"
        elif x < 10 ** 9:
            return f"{x / 1e6:.2f}M"
        return f"{x / 1e9:.2f}G"

    def show_progress(self, pbar: tqdm | None = None, max_cases: int = 10):
        '''Print the current progress of the task manager.
        '''
        # Collect half-finished cases
        progress: list[tuple[int, int, int]] = []

        for i, (total, passed) in enumerate(zip(self.total_samples, self.passed_samples)):
            if 0 < total < self.args.max_samples and passed < self.args.num_samples:
                progress.append((i, total, passed))

        if not progress:
            return

        # Sort by passed samples descendingly and then by case ID ascendingly
        progress.sort(key=lambda x: (-x[2], x[0]))
        _f, it, ot = self._f, *self.token_usage
        msg = (
            f"Samples:  {sum(self.passed_samples)}/{sum(self.total_samples)}    "
            f"Tokens:  {_f(it)} (i)  {_f(ot)} (o)    "
            f"Est. Price:  $ {it * 1.5e-7 + ot * 6e-7:.4f}    "
            f"Cases:  {'  '.join(f'{i} ({p}/{t})' for i, t, p in progress[:max_cases])}"
        )

        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg)

    def log_stats(self, log_path: str):
        '''Log the final statistics of the task manager.
        '''
        with open(log_path, 'w') as f:
            f.write(f"File1,File2,Total,Passed\n")
            for i, (total, passed) in enumerate(zip(self.total_samples, self.passed_samples)):
                f.write(
                    f"{self.dataset[self.cand_inds[i][0]]},"
                    f"{self.dataset[self.cand_inds[i][1]]},"
                    f"{total},{passed}\n"
                )
            f.write(f"Total,,{sum(self.total_samples)},{sum(self.passed_samples)}\n")


def make_prompt(func1: str, func2: str) -> list[dict[str, str]]:
    '''Construct a ChatGPT prompt dialog.
    '''
    # Remove the import statement and leading/trailing whitespaces
    func1 = func1.replace('import bpy', '').strip()
    func2 = func2.replace('import bpy', '').strip()

    dialog = [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {"role": "user", "content": f"```python\nimport bpy\n\n{func1}\n\n{func2}\n```"},
    ]

    return dialog


async def gen_worker(
        tm: TaskManager, task_queue: AsyncQueue[TaskInfo | None],
        render_queue: "Queue[tuple[TaskInfo | None, list[str]]]",
        worker_id: int, args: Namespace
    ):
    '''Keep retrieving tasks from the queue and generate material programs asynchronously.
    The generated programs are then sent to the render queue for rendering.
    '''
    while True:
        # Get the next task from the queue
        task_info = await task_queue.get()

        # No more tasks to process, signal other async workers to stop
        if task_info is None:
            await task_queue.put(None)
            break

        # Unpack the task information
        case_id = task_info.case_id
        sample_id = task_info.sample_id
        output_folder = task_info.output_folder

        # Initialize the case
        if not sample_id:
            os.makedirs(output_folder, exist_ok=True)

            # Copy source code and render files
            for gt_id, f in enumerate(task_info.files):
                code_path = osp.join(args.data_root, osp.dirname(f), 'blender_full.py')
                render_path = osp.join(args.data_root, osp.dirname(f), 'transpiled_render.jpg')
                shutil.copy(code_path, osp.join(output_folder, f'gt_{gt_id:02d}_full.py'))
                shutil.copy(render_path, osp.join(output_folder, f'gt_{gt_id:02d}_render.jpg'))

            # Create stdout log file
            stdout_path = osp.join(output_folder, 'gen_programs_stdout.log')
            open(stdout_path, 'w').close()

        # Retrieve the prompt dialog
        prompt = tm.get_prompt(case_id)

        # Create the prompt if not available
        if not prompt:

            # Clear temporary files in case of continuing from a previous run
            for fn in os.listdir(output_folder):
                if fn.startswith('temp_'):
                    os.remove(osp.join(output_folder, fn))

            # Load the source code for the case
            codes = []
            for gt_id in range(len(task_info.files)):
                with open(osp.join(output_folder, f'gt_{gt_id:02d}_full.py'), 'r') as code_file:
                    code = code_file.read().strip()
                    code = code.replace('def shader_material(', f'def shader_material_{gt_id + 1}(')
                    codes.append(code)

            # Make and cache the prompt dialog
            prompt = make_prompt(*codes)
            tm.cache_prompt(case_id, prompt)

        # Determine the batch size
        batch_size = min(args.batch_size, args.max_samples - sample_id)

        # Call the OpenAI API to generate a new program
        wait_time = args.num_workers_gen

        while True:
            try:
                result = await client.chat.completions.create(
                    model=args.model_id,
                    messages=prompt,
                    max_tokens=4096,
                    n=batch_size
                )

            # Handle quota limit
            except RateLimitError:
                if not worker_id:
                    print(f'Quota limit exceeded. Wait for {wait_time} seconds before retrying.')
                await asyncio.sleep(wait_time)
                wait_time *= 2
            else:
                break

        # Collect the responses
        responses = [c.message.content for c in result.choices if c.message.content]

        # Record the token usage
        usage = result.usage
        if usage is not None:
            tm.report_token_usage(usage.prompt_tokens, usage.completion_tokens)

        # Put the task into the render queue
        render_queue.put((task_info, responses))


def verify_response(response: str, test_id: int, log_path: str) -> str | None:
    '''Check LLM's response and extract the generated code.
    '''
    err_str, code = '', None

    # Check response format
    if '```python' not in response:
        err_str = f"Error when processing test case {test_id}: No code block found\n"
        log_info(log_path, err_str)
        return None

    # Extract the code block
    response = response[response.index('```python'):]
    ind_delim = response.index('```', 3) if '```' in response[3:] else None
    code = response[response.index('\n') + 1:ind_delim].strip()

    # Check function definition
    if 'def shader_material(' not in code:
        err_str = f"Error when processing test case {test_id}: Function not defined\n"

    # Log error message
    if err_str:
        log_info(log_path, err_str)
        return None

    return code


def verify_render(
        render_path: str, prev_images: list[NDArray[np.float32]], test_id: int, log_path: str,
        min_file_size: int = 12000, dup_pixel_threshold: float = 0.02, dup_image_threshold: float = 0.95
    ) -> NDArray[np.float32] | None:
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


def verify_program(
        response: str, output_folder: str, sample_id: int, tokenizer: PreTrainedTokenizer,
        worker_id: int, args: Namespace
    ) -> bool:
    '''Combined verification of the generated program and rendered image. Returns
    the rendered image if the program passes all checks, otherwise none.
    '''
    # Extract the generated code from the response
    stdout_path = osp.join(output_folder, 'gen_programs_stdout.log')
    code = verify_response(response.strip(), sample_id, stdout_path)
    if not code:
        return False

    # Save the generated code
    code_path = osp.join(output_folder, 'temp_code.py')
    with open(code_path, 'w') as code_file:
        code_file.write(code)

    # Set the display name for OpenGL rendering
    device_ids = args.device_id or list(range(args.num_workers_verify))
    display_name = f':{args.display_id}.{device_ids[worker_id % len(device_ids)]}'

    # Render the code to image asynchonously
    material_path = osp.abspath(osp.join(output_folder, 'temp_material.blend'))
    render_path = osp.join(output_folder, 'temp_render.jpg')
    ret = subprocess.run([
        args.blender_path, '-b', '-P', osp.join(THIS_DIR, 'render.py'), '--', '-c', code_path,
        '-i', args.info_dir, '-s', material_path, '-o', render_path
    ], capture_output=True, text=True, env={'DISPLAY': display_name})

    if check_stdout(ret.stdout, render_path, sample_id, stdout_path, print_stdout=False):
        return False

    # Validate the rendered image against previous images
    prev_image_files = [
        *glob.glob(osp.join(output_folder, 'gen_*_render.jpg')),
        *glob.glob(osp.join(output_folder, 'gt_*_render.jpg'))
    ]
    if len(prev_image_files) < 2:
        raise RuntimeError(f'Ground-truth images not found for test case {sample_id}')

    prev_images = [np.asarray(Image.open(f).convert('RGB')) for f in prev_image_files]
    img = verify_render(
        render_path, prev_images, sample_id, stdout_path,
        args.min_file_size, args.dup_pixel_threshold, args.dup_image_threshold
    )
    if img is None:
        return False

    # Analyze the saved material to check unsupported nodes
    analysis_path = osp.join(output_folder, 'temp_analysis.json')
    ret = subprocess.run([
        args.blender_path, material_path, '-b', '-P', osp.join(THIS_DIR, 'analyze.py'),
        '--', analysis_path, '--info_dir', args.info_dir, '-c'
    ], capture_output=True, text=True)

    if check_stdout(ret.stdout, analysis_path, sample_id, stdout_path, print_stdout=False):
        return False

    # Transpile the saved material
    transpile_path = osp.join(output_folder, 'temp_full.py')
    ret = subprocess.run([
        args.blender_path, material_path, '-b', '-P', osp.join(THIS_DIR, 'transpile.py'),
        '--', transpile_path, '-i', args.info_dir, '-a', analysis_path
    ], capture_output=True, text=True)

    if check_stdout(ret.stdout, transpile_path, sample_id, stdout_path, print_stdout=False):
        return False

    # Check the code length
    with open(transpile_path, 'r') as f:
        transpiled_code = f.read().strip()

    conv_version = 'v1.5-hf' if args.model_id.endswith('-hf') else 'v1.5'
    if get_code_length(transpiled_code, tokenizer, conv_version) > MAX_CODE_LEN:
        log_info(stdout_path, f"Error when processing test case {sample_id}:\nCode too long\n")
        return False

    # The program is successfully verified
    return True


def verify_worker(
        render_queue: "Queue[tuple[TaskInfo | None, list[str]]]",
        result_queue: "Queue[tuple[TaskInfo | None, list[bool]]]",
        worker_id: int, args: Namespace
    ):
    '''Keep verifying generated programs asynchronously. The results are posted to the
    task manager and new tasks are added to the task queue for processing.
    '''
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    while True:
        # Get the next task from the queue
        task_info, responses = render_queue.get()

        # End of the task queue
        if task_info is None:
            render_queue.put((None, []))
            break

        # Unpack the task information
        output_folder = task_info.output_folder
        sample_id = task_info.sample_id
        return_id = task_info.return_id

        # Verify each response
        results: list[bool] = []

        for response in responses:
            success = verify_program(
                response, output_folder, sample_id, tokenizer, worker_id, args
            )

            # Save the result if successful, otherwise remove temporary files
            if success:
                rename_list = [
                    ('temp_render.jpg', f'gen_{return_id:02d}_render.jpg'),
                    ('temp_full.py', f'gen_{return_id:02d}_full.py'),
                    ('temp_analysis.json', f'gen_{return_id:02d}_analysis.json')
                ]
                for src, dst in rename_list:
                    os.rename(osp.join(output_folder, src), osp.join(output_folder, dst))

            # Remove temporary files
            for fn in os.listdir(output_folder):
                if fn.startswith('temp_'):
                    os.remove(osp.join(output_folder, fn))

            # Record the result
            sample_id += 1
            return_id += success
            results.append(success)

        # Put the task into the result queue
        result_queue.put((task_info, results))


async def result_worker(
        tm: TaskManager, result_queue: "Queue[tuple[TaskInfo | None, list[bool]]]",
        task_queue: AsyncQueue[TaskInfo | None],
        render_queue: "Queue[tuple[TaskInfo | None, list[str]]]",
        poll_interval: float = 0.1, progress_interval: float = 10.0
    ):
    '''Collect the results from the verification workers and update the task manager.
    '''
    # Create a progress bar
    pbar = tqdm(
        desc='Generating materials',
        total=tm.args.num_cases,
        initial=tm.num_done()
    )
    timestamp = time.time()

    while True:
        # Print progress information
        if time.time() - timestamp > progress_interval:
            tm.show_progress(pbar)
            timestamp = time.time()

        # Get the next result from the queue if available
        if not result_queue.empty():
            task_info, results = result_queue.get()

            # Update the task manager
            case_id = task_info.case_id
            for success in results:
                tm.complete_task(case_id, success)

            # If the case has finished, update the progress bar and generate a new task
            if tm.get_next_task(case_id) is None:
                pbar.update(1)

            # Generate a new task
            next_task = tm.get_next_task()
            if next_task is not None:
                tm.start_task(next_task.case_id)
                await task_queue.put(next_task)

            # Check if all tasks are completed
            elif tm.all_done():
                tm.show_progress(pbar)
                pbar.close()
                break

        # Wait for a short time before checking the queue again
        else:
            await asyncio.sleep(poll_interval)

    # Signal other workers to stop
    await task_queue.put(None)
    render_queue.put((None, []))


async def main():
    # Command line argument parser
    parser = ArgumentParser(description='Generate procedural materials using LLM')

    ## I/O parameters
    parser.add_argument('-b', '--blender_path', type=str,
                        default=osp.join(ROOT_DIR, 'infinigen', 'blender', 'blender'),
                        help='Path to Blender executable')
    parser.add_argument('-d', '--data_root', type=str,
                        default=osp.join(ROOT_DIR, 'material_dataset_filtered_best'),
                        help='Root directory of dataset')
    parser.add_argument('-i', '--info_dir', type=str, default=osp.join(ROOT_DIR, 'material_dataset_info'),
                        help='Directory to analysis results')
    parser.add_argument('-o', '--output_folder', type=str, default='gen_programs',
                        help='Output directory')

    ## Generation settings
    parser.add_argument('-n', '--num_cases', type=int, default=600,
                        help='Number of cases to select for mutation')
    parser.add_argument('-m', '--num_samples', type=int, default=4,
                        help='Number of samples per case to generate')
    parser.add_argument('-x', '--max_samples', type=int, default=20,
                        help='Maximum number of samples per case')
    parser.add_argument('--model_id', type=str, default='gpt-4o-mini', help='Model ID')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for each request')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-r', '--resume', action='store_true', help='Resume from previous run')

    ## Verification settings
    parser.add_argument('--min_file_size', type=int, default=12000,
                        help='Minimum render image file size for generated materials')
    parser.add_argument('--dup_pixel_threshold', type=float, default=0.02,
                        help='Pixel value difference threshold for duplicate image detection')
    parser.add_argument('--dup_image_threshold', type=float, default=0.95,
                        help='Pixel percentage threshold for duplicate image detection')
    parser.add_argument('--tokenizer', type=str, default='llava-hf/llama3-llava-next-8b-hf',
                        help='Tokenizer URL to use')
    parser.add_argument('--log_path', type=str, default='gen_programs.log', help='Log file path')
    parser.add_argument('--display_id', type=int, default=0, help='Display ID for rendering')
    parser.add_argument('--device_id', type=int, nargs='+', default=None,
                        help='Device ID for rendering')

    ## Asynchronous settings
    parser.add_argument('--num_workers_gen', type=int, default=1,
                        help='Number of generation workers')
    parser.add_argument('--num_workers_verify', type=int, default=1,
                        help='Number of verification workers')

    args = parser.parse_args()

    # Check display ID
    check_display_id(args.display_id)

    # Scan the dataset for all Blender files
    all_files = glob.glob(osp.join('*', '*', 'blender_full.py'), root_dir=args.data_root)
    all_files = sorted([
        f for f in all_files
        if osp.isfile(osp.join(args.data_root, osp.dirname(f), 'transpiled_render.jpg'))
    ])
    print(f"Found {len(all_files)} data items in the dataset")

    # Initialize the task manager
    tm = TaskManager(all_files, args)

    # Create async and multiprocessing queues
    task_queue = AsyncQueue()
    render_queue, result_queue = Queue(), Queue()

    # Create asynchronous workers
    async_tasks = [
        asyncio.create_task(gen_worker(tm, task_queue, render_queue, i, args))
        for i in range(args.num_workers_gen)
    ]
    async_tasks.append(
        asyncio.create_task(result_worker(tm, result_queue, task_queue, render_queue))
    )

    # Create multiprocessing workers
    procs = [
        Process(target=verify_worker, args=(render_queue, result_queue, i, args))
        for i in range(args.num_workers_verify)
    ]
    for p in procs:
        p.start()

    # Assign the first set of tasks
    for _ in range(args.num_workers_gen * 2):
        task_info = tm.get_next_task()
        if task_info is None:
            break
        tm.start_task(task_info.case_id)
        await task_queue.put(task_info)

    # Wait for all tasks to complete
    await asyncio.gather(*async_tasks)

    # Wait for all processes to complete
    for p in procs:
        p.join()

    # Log the statistics from the task manager
    os.makedirs(args.output_folder, exist_ok=True)
    tm.log_stats(osp.join(args.output_folder, args.log_path))


if __name__ == '__main__':
    asyncio.run(main())
