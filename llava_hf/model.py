# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

from dataclasses import dataclass
import os.path as osp

from transformers import (
    PreTrainedModel, AutoModelForVision2Seq, AutoConfig, TrainingArguments
)
from peft import get_peft_model, LoraConfig, PeftModel
import torch


@dataclass
class ModelArguments:
    """LLaVA-Next model and LoRA adapter arguments.
    """
    model_name_or_path: str | None = None       # Path or URL to the LLaVA-Next model
    lora_enable: bool = True                    # Whether to enable the LoRA adapter

    # LoRA adapter hyperparameters
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05


def find_all_linear_names(model: PreTrainedModel) -> str:
    """Finds all the linear layer names in the model.
    """
    cls = torch.nn.Linear
    model_class = type(model).__name__

    # Gather linear module names for LoRA
    # Skip linear layers in the projector, the vision encoder, and the LM head
    lora_module_names = set()

    keywords = ['lm_head']
    keywords.extend(
        ['visual'] if model_class.startswith('Qwen2')
        else ['multi_modal_projector', 'vision_tower'] if model_class.startswith('Llava')
        else ['multi_modal_projector', 'vision_model']
    )

    for name, module in model.named_modules():
        if isinstance(module, cls) and all(key not in name for key in keywords):
            lora_module_names.add(name.split('.')[-1])

    # Use the regex pattern to constrain all matches inside the language model
    language_model_name = 'model' if model_class.startswith('Qwen2') else 'language_model'
    return f"{language_model_name}.*.({'|'.join(list(lora_module_names))})"


def build_model(
        model_args: ModelArguments, training_args: TrainingArguments
    ) -> PreTrainedModel | PeftModel:
    """Builds the LLaVA-Next model with the LoRA adapter.
    """
    # Load the pre-trained model
    model_base = model_args.model_name_or_path
    model = AutoModelForVision2Seq.from_pretrained(
        model_base,
        attn_implementation='sdpa' if 'Llama-3.2' in model_base else 'flash_attention_2',
        torch_dtype='auto'
    )
    model_class = type(model).__name__

    # Patch the model config to specify the hidden size for DeepSpeed
    if not hasattr(model.config, 'hidden_size'):
        if model_class.startswith(('Llava', 'Mllama')):
            model.config.hidden_size = model.language_model.config.hidden_size
        else:
            raise RuntimeError("The model config does not specify 'hidden_size'.")

    # Suppress the warning message about incompatibility between default model config
    # and gradient checkpointing
    lm = model if model_class.startswith('Qwen2') else model.language_model
    lm.config.use_cache = False

    # Enable gradient checkpointing for the language model
    if training_args.gradient_checkpointing:
        lm = model.model if model_class.startswith('Qwen2') else model.language_model
        lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )

    # Load the LoRA adapter
    if model_args.lora_enable:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            target_modules=find_all_linear_names(model),
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            task_type='CASUAL_LM'
        )
        model = get_peft_model(model, lora_config)

    # Cast the model to BF16 or FP16
    model = model.to(
        torch.bfloat16 if training_args.bf16
        else torch.float16 if training_args.fp16
        else torch.float32
    )

    # Set the vision projector to be trainable in addition to LoRA adapters
    vision_projector = (
        model.visual.merger.mlp if model_class.startswith('Qwen2')
        else model.multi_modal_projector
    )
    vision_projector.requires_grad_(True)

    return model


def load_pretrained_model(
        model_path: str | None, model_base: str, device: str | torch.device = 'cpu'
    ) -> PreTrainedModel:
    '''Load a pretrained model from user-specified path.
    '''
    # Load the pre-trained model
    model = AutoModelForVision2Seq.from_pretrained(
        model_base,
        config=AutoConfig.from_pretrained(model_path or model_base),
        low_cpu_mem_usage=True,
        attn_implementation='sdpa' if 'Llama-3.2' in model_base else 'flash_attention_2',
        torch_dtype='auto',
        device_map=device
    )
    model_class = type(model).__name__

    # Enable 'use_cache' for inference
    lm = model if model_class.startswith('Qwen2') else model.language_model
    lm.config.use_cache = True

    # Read from fine-tuned checkpoints
    if model_path:
        # Read non-LoRA parameters
        non_lora_state_dict = torch.load(osp.join(model_path, 'base_model.bin'), weights_only=True)
        module_prefix = 'base_model.model.'     # Prefix for Peft-wrapped models
        non_lora_state_dict = {
            (k[len(module_prefix):] if k.startswith(module_prefix) else k): v
            for k, v in non_lora_state_dict.items()
        }
        model.load_state_dict(non_lora_state_dict, strict=False)

        # Load and merge LoRA parameters into the model
        if osp.exists(osp.join(model_path, 'adapter_model.bin')):
            model = PeftModel.from_pretrained(model, model_path)
            model: PreTrainedModel = model.merge_and_unload()

    return model
