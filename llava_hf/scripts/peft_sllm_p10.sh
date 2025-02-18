#!/bin/bash

EXP_NAME="llava-llama3-8b-sllm-p10"
ROOT_DIR="/path/to/workspace/VLMaterial"
WORK_DIR="${ROOT_DIR}/llava_hf"
DATA_DIR="${ROOT_DIR}/material_dataset_filtered"
SPLIT_DIR="${DATA_DIR}/dataset_splits"
DEVICE_IDS="0,1,2,3,4,5,6,7"

deepspeed --include localhost:${DEVICE_IDS} --master_port=29501 ${WORK_DIR}/train.py \
    --model_name_or_path llava-hf/llama3-llava-next-8b-hf \
    --max_length 5120 \
    --lora_r 8 \
    --lora_alpha 32 \
    --deepspeed ${WORK_DIR}/scripts/zero3.json \
    --train_data_path ${SPLIT_DIR}/llava_noaug_train.json \
    --val_data_path ${SPLIT_DIR}/llava_noaug_val.json \
    --image_folder ${DATA_DIR} \
    --group_by_length True \
    --bf16 True \
    --tf32 True \
    --output_dir ${WORK_DIR}/checkpoints/${EXP_NAME} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard
