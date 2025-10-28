# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

import os
import os.path as osp
import sys

# Align CUDA device order with X server
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

sys.path.append(osp.dirname(osp.abspath(__file__)))

import os
import subprocess
from dataclasses import dataclass, field

import torch
from dataset import DataArguments, MaterialDataset
from model import load_pretrained_model
from PIL import Image
from torch.multiprocessing import Process, Queue, set_start_method
from tqdm import tqdm
from transformers import AutoProcessor, HfArgumentParser
from utils import SYSTEM_PROMPT_LLAVA, check_stdout, log_info

ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
SCRIPTS_DIR = osp.join(ROOT_DIR, 'data_scripts')


@dataclass
class Arguments:
    '''Arguments for LLM inference.
    '''
    # I/O
    test_data_path: str = ""
    output_dir: str = "./output"

    # Model configuration
    model_path: str | None = None
    model_base: str = 'llava-hf/llama3-llava-next-8b-hf'
    display_id: int = 0
    device_id: list[int] = field(default_factory=list)

    # Dataset-related info
    blender_path: str = osp.join(ROOT_DIR, 'infinigen', 'blender', 'blender')
    image_folder: str = osp.join(ROOT_DIR, 'material_dataset_filtered')
    info_dir: str = osp.join(ROOT_DIR, 'material_dataset_info')
    data_format: str = 'v1.5'

    # Inference settings
    mode: str = 'all'                   # Inference mode (generation, verification, both, or single)
    num_processes: int = 1              # Number of processes for parallel inference
    num_samples: int = 4                # Target number of samples to generate
    max_samples: int = 20               # Maximum number of trials
    max_length: int = 2048              # Maximum token sequence length
    batch_size: int | None = None       # Batch size for inference
    temperature: float = 1.0
    top_k: int = 10
    top_p: float = 0.9
    min_file_size: int = 12000          # Minimum file size for rendered material images
    
    # Single prompt mode settings
    single_image_path: str = ""         # Path to single image for single prompt mode
    single_text_prompt: str = ""        # Text prompt for single prompt mode


def response_to_code(response: str) -> str:
    '''Extract code block delimited by triple backticks from the response.
    '''
    code = response.strip()
    if "```" in code:
        code = code[code.index("```") + 3:]
    if code.startswith("python"):
        code = code[6:]
    if "```" in code:
        code = code[:code.index("```")]
    return code


def verify_program(
        response: str, args: Arguments, target_id: int, sample_id: int, example_dir: str, stdout_path: str,
        min_file_size: int = 12000, display_id: int | None = None, device_id: int | None = None
    ) -> bool:
    '''Verify the generated response and save the program if successful.
    '''
    # Define temporary file paths
    code_path = osp.join(example_dir, 'temp_full.py')
    render_path = osp.join(example_dir, 'temp_render.jpg')

    # Clean up temporary files if any
    for f in os.listdir(example_dir):
        if f.startswith('temp_'):
            os.remove(osp.join(example_dir, f))

    # Extract the program from the response
    with open(code_path, "w") as f:
        f.write(response_to_code(response))

    # Render the code to an image
    display_name = f':{display_id if display_id is not None else 0}'
    display_name += f'.{device_id}' if device_id is not None else ''
    render_kwargs = {
        'capture_output': True,
        'text': True,
        'env': {'DISPLAY': display_name},
        'timeout': 120
    }

    try:
        # print("args.blender_path: ", args.blender_path)
        # print(f"{args.blender_path} -b -P {osp.join(SCRIPTS_DIR, 'render.py')} '--' '-c' {code_path} -i {args.info_dir} -o {render_path}")
        ret = subprocess.run([
            args.blender_path, '-b', '-P', osp.join(SCRIPTS_DIR, 'render.py'),
            '--', '-c', code_path, '-i', args.info_dir, '-o', render_path,
        ], **render_kwargs)

    except subprocess.TimeoutExpired:
        log_info(stdout_path, f"Error when processing test case {sample_id}:\nRender timed out")
        return False

    # Check render result
    if check_stdout(ret.stdout, render_path, sample_id, stdout_path):
        # print("Error result: ", ret.stdout)
        return False

    # Check the rendered image file size
    if osp.getsize(render_path) < min_file_size:
        log_info(stdout_path, f"Error when processing test case {sample_id}:\nRendered image too small")
        return False

    # Rename the generated files
    pred_name = f'pred_{target_id:05d}'
    os.rename(code_path, osp.join(example_dir, f'{pred_name}_full.py'))
    os.rename(render_path, osp.join(example_dir, f'{pred_name}_render.jpg'))

    return True


def run_inference(
        args: Arguments, rank: int, task_queue: Queue, result_queue: Queue
    ):
    '''Runs the inference process.
    '''
    # Create a local dataset copy
    processor = AutoProcessor.from_pretrained(args.model_base)
    data_args = DataArguments(
        image_folder=args.image_folder,
        data_format=args.data_format
    )
    dataset = MaterialDataset(
        args.test_data_path, processor, data_args, inference=True,
        zero_shot=args.model_path is None, system_prompt=SYSTEM_PROMPT_LLAVA
    )

    # PyTorch device
    devices = [f'cuda:{i}' for i in args.device_id]
    device = torch.device(devices[rank] if len(devices) > 1 else devices[0])

    # Load the pretrained model
    if args.mode in ('all', 'gen'):
        model = load_pretrained_model(args.model_path, args.model_base, device)
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    else:
        model = None

    # Main inference loop
    while True:
        # Get the next task index; break if the stop signal is received
        idx = task_queue.get()
        if idx < 0:
            break

        # Read the source data dictionary
        source = dataset.get_source(idx)

        # Create the output folder for this case
        example_dir = osp.join(args.output_dir, f"{idx:03d}" + "-" + source['id'])
        os.makedirs(example_dir, exist_ok=True)

        # Generation-mode preparation
        if args.mode in ('all', 'gen'):
            # Save input image and ground-truth code
            input_image = Image.open(osp.join(args.image_folder, source['image'])).convert('RGB')
            input_image.save(osp.join(example_dir, "input.jpg"))

            if len(source['conversation']) > 1:
                gt_code = source['conversation'][1]['content'][0]['text'].strip().strip('```')
                gt_code = gt_code[gt_code.index('\n') + 1:] if gt_code.startswith('python') else gt_code
                with open(osp.join(example_dir, "gt_full.py"), "w") as f:
                    f.write(gt_code)

            # Move data to device and cast to the correct data type
            inputs = {k: v.to(device, non_blocking=True) for k, v in dataset[idx].items()}
            inputs = {
                k: v.to(model.dtype) if torch.is_floating_point(v) else v
                for k, v in inputs.items()
            }

        # Verification-moe preparation
        if args.mode in ('all', 'render'):
            # Create the stdout log file
            stdout_path = osp.join(example_dir, 'gen_programs_stdout.log')
            open(stdout_path, 'w').close()

        # Initialize the progress bar
        pbar = tqdm(
            total=args.max_samples if args.mode == 'gen' else args.num_samples,
            desc=osp.basename(example_dir),
            position=0
        )

        # Keep generating samples until reaching the user-specified number
        num_sampled, num_passed = 0, 0

        while num_sampled < args.max_samples and num_passed < args.num_samples:
            # Calculate batch size
            batch_size = args.batch_size if args.batch_size is not None else args.num_samples
            batch_size = min(batch_size, args.max_samples - num_sampled)

            # Generate a batch of responses (included in mode 'all' or 'gen')
            if args.mode in ('all', 'gen'):
                # Create batched input
                batch_inputs = {
                    k: (
                        v.expand(batch_size, *v.shape[1:]) if v.shape[0] == 1
                        else v.repeat(batch_size, *([1] * (v.ndim - 1)))
                    ) 
                    for k, v in inputs.items()
                }

                # Generate samples
                with torch.inference_mode():
                    outputs = model.generate(
                        **batch_inputs,
                        do_sample=args.temperature > 0,
                        max_new_tokens=args.max_length,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        use_cache=True,
                        return_dict_in_generate=True
                    )

                # Check if the input tokens are correctly returned
                input_ids, output_ids = batch_inputs['input_ids'], outputs.sequences
                input_length = input_ids.shape[1]
                if not torch.equal(input_ids, output_ids[:, :input_length]):
                    raise ValueError("Input tokens are not correctly returned.")

                # Decode the generated samples
                responses = processor.batch_decode(output_ids[:, input_length:], skip_special_tokens=True)

                # Save responses
                for i, response in enumerate(responses):
                    with open(osp.join(example_dir, f'sample_{num_sampled + i:05d}_response.txt'), 'w') as f:
                        f.write(response)

                    # Update the progress bar in generation-only mode
                    if args.mode == 'gen':
                        pbar.update(1)

                pbar.refresh()

            # Read responses from saved files if the mode is verification only
            else:
                responses = []
                for i in range(batch_size):
                    with open(osp.join(example_dir, f'sample_{num_sampled + i:05d}_response.txt')) as f:
                        responses.append(f.read())

            # Verify the responses
            if args.mode in ('all', 'render'):
                # Process each response
                for response in responses:
                    print("Verifying Programs")
                    if verify_program(
                        response, args, num_passed, num_sampled, example_dir, stdout_path,
                        min_file_size=args.min_file_size, display_id=args.display_id,
                        device_id=device.index
                    ):
                        pbar.update(1)
                        num_passed += 1
                    else:
                       print("Program verification failed!") 
                    num_sampled += 1

                    # Finish early if the desired number of samples is reached
                    if num_passed >= args.num_samples:
                        break

            # Otherwise, update the number of samples so that the iteration can continue
            else:
                num_sampled += batch_size

        # Delete the temporary files if any
        for f in os.listdir(example_dir):
            if f.startswith('temp_'):
                os.remove(osp.join(example_dir, f))

        # Update the result queue
        result_queue.put(idx)

def render_single_code(
    code: str | None = None,
    code_path: str | None = None,
    *,
    args: Arguments,
    example_dir: str,
    stdout_path: str | None = None,
    min_file_size: int = 12000,
    display_id: int | None = None,
    device_id: int | None = None,
    timeout: int = 120,
    save_path: str | None = None,
) -> str | None:
    """Render a single Python program using the project's Blender render script.

    Either `code` (a string containing the program) or `code_path` (path to an existing .py file)
    must be provided. The function writes temporary files into `example_dir`, runs Blender to
    render an image, checks the output, and on success returns the path to the rendered image.
    Returns None on failure.
    """
    # Prepare example dir
    os.makedirs(example_dir, exist_ok=True)

    # Paths for temp files
    temp_code_path = code_path or osp.join(example_dir, "temp_single.py")
    render_path = osp.join(example_dir, "temp_single_render.jpg")

    # Clean up any previous temp files
    for f in os.listdir(example_dir):
        if f.startswith("temp_"):
            try:
                os.remove(osp.join(example_dir, f))
            except OSError:
                pass

    # Write code string to file if provided
    if code is not None:
        with open(temp_code_path, "w") as f:
            f.write(response_to_code(code))
    else:
        if not osp.exists(temp_code_path):
            raise FileNotFoundError(f"code_path does not exist: {temp_code_path}")

    # Ensure stdout_path exists
    if stdout_path is None:
        stdout_path = osp.join(example_dir, "render_single_stdout.log")
        open(stdout_path, "w").close()

    # Build display name and subprocess kwargs
    display_name = f":{display_id if display_id is not None else 0}"
    display_name += f".{device_id}" if device_id is not None else ""
    # Preserve the caller environment and set DISPLAY so Blender sees the
    # same PYTHONPATH and other vars (avoids ModuleNotFoundError for packages
    # installed in the user's environment).
    env = os.environ.copy()
    env["DISPLAY"] = display_name
    render_kwargs = {
        "capture_output": True,
        "text": True,
        "env": env,
        "timeout": timeout,
    }

    try:
        cmd = [
            args.blender_path,
            "-b",
            "-P",
            osp.join(SCRIPTS_DIR, "render.py"),
            "--",
            "-c",
            temp_code_path,
            "-i",
            args.info_dir,
            "-o",
            render_path,
        ]
        if save_path is not None:
            cmd += ["--save-path", save_path]
        ret = subprocess.run(cmd, **render_kwargs)
    except subprocess.TimeoutExpired:
        log_info(stdout_path, "Error when rendering single program:\nRender timed out")
        return None

    # Check render success and file size
    if check_stdout(ret.stdout, render_path, 0, stdout_path):
        return None

    if not osp.exists(render_path) or osp.getsize(render_path) < min_file_size:
        log_info(
            stdout_path,
            "Error when rendering single program:\nRendered image too small or missing",
        )
        return None

    # Move outputs to a stable name
    out_render = osp.join(example_dir, "render_single.jpg")
    try:
        os.rename(render_path, out_render)
    except OSError:
        # If rename fails, try copy
        import shutil

        shutil.copyfile(render_path, out_render)

    # If we wrote a temp file at temp_code_path (and code was provided), move it too
    if code is not None:
        out_code = osp.join(example_dir, "render_single.py")
        try:
            os.rename(temp_code_path, out_code)
        except OSError:
            import shutil

            shutil.copyfile(temp_code_path, out_code)

    return out_render


def run_inference_single_prompt(args: Arguments, image_path: str, text_prompt: str) -> str:
    """Run inference on a single image with a custom text prompt.
    
    Args:
        args: Arguments containing model configuration
        image_path: Path to the input image
        text_prompt: Custom text prompt for the model
        
    Returns:
        Generated response text
    """
    # Create processor
    processor = AutoProcessor.from_pretrained(args.model_base)
    
    # Set up device - use CPU if CUDA is not available
    if torch.cuda.is_available() and args.device_id:
        devices = [f'cuda:{i}' for i in args.device_id]
        device = torch.device(devices[0])
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    
    # Load model
    model = load_pretrained_model(args.model_path, args.model_base, device)
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    
    # Create conversation format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_prompt}
            ]
        }
    ]
    
    # Apply chat template
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(
        images=image,
        text=prompt, 
        return_tensors="pt",
        add_special_tokens=not prompt.startswith(processor.tokenizer.bos_token)
    )
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate response
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=args.temperature > 0,
            max_new_tokens=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=True,
            return_dict_in_generate=True
        )
    
    # Decode response
    input_length = inputs['input_ids'].shape[1]
    response = processor.decode(outputs.sequences[0][input_length:], skip_special_tokens=True)
    
    return response


class SinglePromptEngine:
    """Preload processor and model once and run single-prompt inference efficiently.

    Usage:
        engine = SinglePromptEngine(args)
        response = engine.generate(image_path, text_prompt)
    """

    def __init__(self, args: Arguments):
        self.args = args
        # Processor
        self.processor = AutoProcessor.from_pretrained(args.model_base)

        # Device selection (prefer specified CUDA device, fallback to CUDA:0, then CPU)
        if torch.cuda.is_available() and args.device_id:
            devices = [f'cuda:{i}' for i in args.device_id]
            self.device = torch.device(devices[0])
        elif torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            print("CUDA not available, using CPU")
            self.device = torch.device('cpu')

        # Model
        self.model = load_pretrained_model(args.model_path, args.model_base, self.device)
        # Ensure pad token is set for generation
        self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.eval()

    @torch.inference_mode()
    def generate(self, image_path: str, text_prompt: str) -> str:
        # Load and process image
        image = Image.open(image_path).convert('RGB')

        # Conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image"},
                ]
            }
        ]

        # Apply chat template
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(
            images=image, text=prompt,
            add_special_tokens=not prompt.startswith(self.processor.tokenizer.bos_token),
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate
        outputs = self.model.generate(
            **inputs,
            do_sample=self.args.temperature > 0,
            max_new_tokens=self.args.max_length,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            use_cache=True,
            return_dict_in_generate=True
        )

        # Decode
        input_length = inputs['input_ids'].shape[1]
        print(outputs.sequences.shape)
        response = self.processor.decode(outputs.sequences[0][input_length:], skip_special_tokens=True)
        return response


def main():
    # Parse command-line arguments
    hf_parser = HfArgumentParser(Arguments)
    args = hf_parser.parse_args_into_dataclasses()[0]

    # Check mode parameter
    if args.mode not in ('all', 'gen', 'render'):
        raise ValueError(f'Unknown inference mode: {args.mode}')

    # Create the dataset
    processor = AutoProcessor.from_pretrained(args.model_base)
    data_args = DataArguments(
        image_folder=args.image_folder,
        data_format=args.data_format
    )
    dataset = MaterialDataset(
        args.test_data_path, processor, data_args, inference=True,
        zero_shot=args.model_path is None, system_prompt=SYSTEM_PROMPT_LLAVA
    )

    # Create the multi-processing queues
    task_queue = Queue()
    result_queue = Queue()

    # Insert tasks and stop signals
    for i in range(len(dataset)):
        task_queue.put(i)
    for _ in range(args.num_processes):
        task_queue.put(-1)

    # Spawn processes
    processes = []
    for i in range(args.num_processes):
        p = Process(target=run_inference, args=(args, i, task_queue, result_queue))
        p.start()
        processes.append(p)

    # Use a progress bar to monitor the total progress
    pbar = tqdm(total=len(dataset), desc='Inference', unit='images', position=0)
    while pbar.n < len(dataset):
        result_queue.get()
        pbar.update()

    # Wait for all processes to finish
    for p in processes:
        p.join()


if __name__ == '__main__':
    set_start_method('spawn')
    main()
