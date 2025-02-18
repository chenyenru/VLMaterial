#!/bin/bash

EXP_NAME="llava-llama3-8b-sllm-p10"
ROOT_DIR="/path/to/workspace/VLMaterial"
WORK_DIR="${ROOT_DIR}/llava_hf"
DATA_DIR="${ROOT_DIR}/material_dataset_filtered"
SPLIT_DIR="${DATA_DIR}/dataset_splits"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

python ${WORK_DIR}/inference.py \
    --model_path ${WORK_DIR}/checkpoints/${EXP_NAME}/checkpoint-epoch5 \
    --model_base llava-hf/llama3-llava-next-8b-hf \
    --test_data_path ${SPLIT_DIR}/llava_noaug_test.json \
    --image_folder ${DATA_DIR} \
    --output_dir ${WORK_DIR}/results/${EXP_NAME}/eval-epoch5 \
    --num_processes 8 \
    --display_id 0 \
    --device_id 0 1 2 3 4 5 6 7 \
    --mode render
