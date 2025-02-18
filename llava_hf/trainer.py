# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

import os.path as osp

from transformers import Trainer
from transformers.trainer import has_length, LengthGroupedSampler
import torch
import torch.nn as nn

from dataset import MaterialDataset


def get_peft_state_dict(model: nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Returns the state dict of trainable parameters only.
    """
    from deepspeed import zero

    # Filter out the non-trainable parameters
    trainable_params = {k: v for k, v in model.named_parameters() if v.requires_grad}

    # Gather the parameters from ZeRO-3 if needed
    state_dict: dict[str, torch.Tensor] = {}

    for n, p in trainable_params.items():
        if hasattr(p, 'ds_id'):
            with zero.GatheredParameters([p]):
                state_dict[n] = p.data.detach().cpu().clone()
        else:
            state_dict[n] = p.detach().cpu().clone()

    # Split the state dict into adapter and other parameters
    lora_state_dict = {k: v for k, v in state_dict.items() if 'lora_' in k}
    other_state_dict = {k: v for k, v in state_dict.items() if 'lora_' not in k}

    return lora_state_dict, other_state_dict


class LlavaTrainer(Trainer):
    """Adapts the original HF Trainer to the custom dataset.
    """
    def _get_train_sampler(self) -> torch.utils.data.Sampler | None:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler
        if self.args.group_by_length and isinstance(self.train_dataset, MaterialDataset):
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=self.train_dataset.lengths,
                model_input_name=model_input_name
            )

        return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        """Overrides the original checkpointing method to save only trainable parameters.
        """
        # Determine the checkpoint folder
        PREFIX_CHECKPOINT_DIR = 'checkpoint'
        should_save = self.args.should_save

        if self.args.save_strategy == 'steps':
            checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'
        else:
            epoch = int(self.state.epoch + 1e-8)
            checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-epoch{epoch}'

            # Save the model every `save_epochs` epochs
            should_save = should_save and (epoch % self.args.save_epochs == 0)

        # Get the output directory
        run_dir = self._get_output_dir(trial=trial)
        output_dir = osp.join(run_dir, checkpoint_folder)

        # Get the state dict of trainable parameters (supporting ZeRO-3)
        lora_state_dict, other_state_dict = get_peft_state_dict(self.model)

        # Save the model only on the first rank
        if should_save:
            self.model.config.save_pretrained(output_dir)
            self.model.save_pretrained(output_dir, state_dict=lora_state_dict, safe_serialization=False)
            torch.save(other_state_dict, osp.join(output_dir, 'base_model.bin'))
