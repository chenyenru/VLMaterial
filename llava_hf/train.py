# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

from dataclasses import dataclass, field
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.abspath(__file__)))

from transformers import TrainingArguments as HfTrainingArguments, HfArgumentParser, AutoProcessor

from dataset import MaterialDataset, DataArguments
from model import ModelArguments, build_model
from peft import PeftModel
from trainer import LlavaTrainer


# Helper function to print only on the first rank
local_rank = None

def rank0_print(*args, **kwargs):
    """Prints only on the first rank.
    """
    if not local_rank:
        print(*args, **kwargs)


@dataclass
class TrainingArguments(HfTrainingArguments):
    """Custom training arguments.
    """
    save_epochs: int = 1                                        # Save a checkpoint every n epochs
    label_names: list[str] = field(default_factory=lambda: ['labels'])      # Show evaluation loss


def main():
    # Parse command-line arguments
    hf_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses()

    # Set the local rank for printing
    global local_rank
    local_rank = training_args.local_rank

    # Create the training and validation datasets
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    train_dataset = MaterialDataset(data_args.train_data_path, processor, data_args)
    val_dataset = MaterialDataset(data_args.val_data_path, processor, data_args)

    # Build the model
    model = build_model(model_args, training_args)

    # Print the number of parameters
    if isinstance(model, PeftModel) and not local_rank:
        model.print_trainable_parameters()

    # Gradient checkpointing has been properly set up
    # Disable `gradient_checkpointing` to prevent further interference from the HF Trainer
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing = False
        rank0_print(
            f'Gradient checkpointing has been enabled for the following modules:',
            [n for n, m in model.named_modules() if getattr(m, 'gradient_checkpointing', False)]
        )

    # Create the Trainer
    trainer = LlavaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_dataset.collate_fn,
        tokenizer=processor.tokenizer
    )

    # Train the model
    trainer.train()


if __name__ == '__main__':
    main()
