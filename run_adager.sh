#!/bin/bash

DATASET="sst2"
SPLIT="val"
N_INPUTS=10
MODEL="gpt2"

BATCH_SIZES=(1 2 4 8 16 32 64 128)

for BS in "${BATCH_SIZES[@]}"; do
    echo "========================================"
    echo "Running batch_size=${BS}"
    echo "========================================"

    python attack_autoregressive_dager.py \
        --dataset ${DATASET} \
        --task seq_class \
        --split ${SPLIT} \
        --batch_size ${BS} \
        --n_inputs ${N_INPUTS} \
        --model_path ${MODEL} \
        --use_hf_split \
        --l1_filter maxB \
        --l2_filter overlap

    echo ""
done