#!/bin/bash

EXP_NAME="llava-llama3-8b-sllm-p10"
ROOT_DIR="/data/VLMaterial"
WORK_DIR="${ROOT_DIR}/llava_hf"
DATA_DIR="${ROOT_DIR}/material_dataset_filtered"
SPLIT_DIR="${DATA_DIR}/dataset_splits"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

python ${WORK_DIR}/inference.py \
    --model_path ${WORK_DIR}/checkpoints_pretrained/${EXP_NAME}/checkpoint-epoch5 \
    --model_base llava-hf/llama3-llava-next-8b-hf \
    --test_data_path ${SPLIT_DIR}/llava_noaug_test.json \
    --image_folder ${DATA_DIR} \
    --output_dir ${WORK_DIR}/results/${EXP_NAME}/testing-inference-function \
    --num_processes 4 \
    --display_id 1 \
    --device_id 4 5 6 7 \
    --temperature 0.6 \
    --top_k 50 \
    --top_p 0.9 \
    --mode gen
