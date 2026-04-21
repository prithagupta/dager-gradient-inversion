#!/bin/bash
array=( "$@" )
len=${#array[@]}
last_args=( "${array[@]:2:$len}" )

python attack_hybrid.py --dataset "$1" --split val --n_inputs 5 --batch_size "$2" \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 --device auto \
  --task seq_class --cache_dir $HF_HOME/gia_exp_cache "${last_args[@]}"