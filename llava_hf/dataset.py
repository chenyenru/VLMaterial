# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

from dataclasses import dataclass
from PIL import Image
from typing import Any
import copy
import json
import os.path as osp

from transformers import ProcessorMixin, Qwen2VLProcessor
import torch


@dataclass
class DataArguments:
    """Dataset-related arguments."""
    train_data_path: str | None = None      # Path to the training data
    val_data_path: str | None = None        # Path to the validation data
    image_folder: str | None = None         # Path to the image folder
    data_format: str = 'v1.5'               # Data format (LLaVA) version
    max_length: int = 2048                  # Maximum token sequence length
    verbose: bool = False                   # Print verbose information


class MaterialDataset(torch.utils.data.Dataset):
    """Blender procedural material dataset for the LLaVA-Next model.
    """
    def __init__(
            self, data_path: str, processor: ProcessorMixin,
            data_args: DataArguments, inference: bool = False, zero_shot: bool = False,
            system_prompt: str = ''
        ):
        super().__init__()
        self.processor = processor
        self.data_args = data_args

        # Inference mode flag
        self.inference = inference

        # Load the JOSN file
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        # Convert the data to be compatible with the processor
        if data_args.data_format not in ('v1.5', 'hf'):
            raise ValueError(f"Data format '{data_args.data_format}' is not supported.")

        if data_args.data_format == 'v1.5':
            self.data = self._convert_v1_5(self.data)

        # Set the system prompt for zero-shot prediction
        if inference and zero_shot:
            self.data = self._convert_zero_shot(self.data, system_prompt)

        # Set the padding token and ignore index
        self.pad_token = self.processor.tokenizer.pad_token_id
        self.ignore_index = -100

        # Image features collate mode
        self.image_collate_mode = (
            'cat' if isinstance(self.processor, Qwen2VLProcessor) else 'stack'
        )

    def _convert_v1_5(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Converts the data from LLaVA v1.5 to the LLaVA-Next format.
        """
        new_data = []

        # Convert each item
        for item in data:
            new_item = {
                'id': item['id'],
                'image': item['image'],
            }

            # Translate the conversation
            new_conv = []
            for message in item['conversations']:
                role = message['from']
                if role == 'human':
                    new_message = {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': message['value'].replace('<image>', '').strip()},
                            {'type': 'image'}
                        ]
                    }
                elif role == 'gpt':
                    new_message = {
                        'role': 'assistant',
                        'content': [
                            {'type': 'text', 'text': message['value'].strip()},
                        ]
                    }
                else:
                    raise ValueError(f"Unknown role '{role}'")
                new_conv.append(new_message)

            new_item['conversation'] = new_conv
            new_data.append(new_item)

        return new_data

    def _convert_zero_shot(
            self, data: list[dict[str, Any]], system_prompt: str
        ) -> list[dict[str, Any]]:
        """Converts the data for zero-shot prediction.
        """
        new_data = []
        applied_cases = 0

        # Convert each item
        for item in data:
            new_item = copy.deepcopy(item)

            # Change the first user message to the system prompt
            user_message = next((m for m in new_item['conversation'] if m['role'] == 'user'), None)
            if user_message:
                user_text = next((t for t in user_message['content'] if t['type'] == 'text'), None)
                if user_text:
                    user_text['text'] = system_prompt
                    applied_cases += 1

            new_data.append(new_item)

        # Check if the system prompt was applied to all cases
        if applied_cases != len(data):
            print(f"Warning: system prompt was applied to {applied_cases} out of {len(data)} cases")

        return new_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source = self.data[index]

        # Read the source image
        image_path = osp.join(self.data_args.image_folder, source['image'])
        image = Image.open(image_path)

        # Only use the generation prompt for inference
        user_conv = [m for m in source['conversation'] if m['role'] == 'user']
        user_prompt = self.processor.apply_chat_template(user_conv, add_generation_prompt=True)
        inputs = self.processor(
            images=image, text=user_prompt,
            add_special_tokens=not user_prompt.startswith(self.processor.tokenizer.bos_token),
            return_tensors='pt'
        )

        if self.inference:
            return inputs

        # Extract and tokenize the assistant response
        prompt = self.processor.apply_chat_template(source['conversation'])
        responses = self.processor.tokenizer(
            prompt[len(user_prompt):], add_special_tokens=False, return_tensors='pt'
        )
        user_prompt_len = inputs['input_ids'].shape[1]

        # Add the assistant response to the inputs (excluding the extra bos token)
        bos_token = self.processor.tokenizer.bos_token_id
        start_pos = int(bos_token is not None and responses['input_ids'][0, 0] == bos_token)

        for key, val in inputs.items():
            if key in ('input_ids', 'attention_mask', 'cross_attention_mask'):
                if key in responses:
                    response = responses[key][:, start_pos:]
                else:
                    response = val[:, -1:].expand(
                        val.shape[0], responses['input_ids'].shape[1] - start_pos, *val.shape[2:]
                    )
                inputs[key] = torch.cat((val, response), dim=1)

        # Remove the batch dimension of the input tensors, note that some models concatenate
        # image features along the first dimension rather than stacking them (e.g. Qwen2 VL)
        for key, val in inputs.items():
            if (
                key in ('input_ids', 'attention_mask', 'cross_attention_mask')
                or self.image_collate_mode == 'stack'
            ):
                inputs[key] = val.squeeze(0)

        # Copy the input_ids to labels for causal language modeling. Mask all tokens up to
        # the generation prompt
        labels = inputs['input_ids'].clone()
        labels[:user_prompt_len] = self.ignore_index
        labels[labels == self.pad_token] = self.ignore_index
        inputs['labels'] = labels

        return inputs

    @property
    def lengths(self) -> list[int]:
        lengths = []

        # Calculate the number of "words" in each prompt as a rough indication
        # for tokenized length
        for d in self.data:
            d_len = 0
            for msg in d['conversation']:
                for text in (m for m in msg['content'] if m['type'] == 'text'):
                    d_len += len(text['text'].split())
            lengths.append(d_len)

        return lengths

    def get_source(self, index: int) -> dict[str, Any]:
        return self.data[index]

    def collate_fn(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Custom collate function for the dataset.
        """
        # Check the input features
        if not len(features):
            raise ValueError("No features provided")

        # This collator function is only used for training
        if self.inference:
            raise NotImplementedError("Collator function is not available for inference")

        # Initialize the batch dictionary
        batch: dict[str, torch.Tensor] = {}

        for key in features[0]:
            # Pad and stack the tokenized sequences and attention masks
            if key in ('input_ids', 'labels', 'attention_mask', 'cross_attention_mask'):
                padding_value = (
                    self.pad_token if key == 'input_ids'
                    else self.ignore_index if key == 'labels'
                    else 0
                )
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [f[key] for f in features],
                    batch_first=True,
                    padding_value=padding_value
                )

            # Stack or concatenate other image-related features
            else:
                batch[key] = (
                    torch.stack([f[key] for f in features]) if self.image_collate_mode == 'stack'
                    else torch.cat([f[key] for f in features], dim=0)
                )

        # Check the padded length against the maximum length (excluding the extra bos token)
        if batch['input_ids'].size(1) - 1 > self.data_args.max_length:
            print(
                f"Warning: batched input sequence length exceeds the maximum length "
                f"({batch['input_ids'].size(1) - 1} > {self.data_args.max_length})"
            )
        elif self.data_args.verbose:
            print(
                f"Batch input sequence lengths: {(batch['input_ids'] != self.pad_token).sum(1).tolist()}, "
                f"label length: {(batch['labels'] != self.ignore_index).sum(1).tolist()}"
            )

        # Check the attention mask (backwards compatibility)
        if not torch.equal(batch['attention_mask'], batch['input_ids'] != self.pad_token):
            raise ValueError("Attention mask mismatch")

        return batch
