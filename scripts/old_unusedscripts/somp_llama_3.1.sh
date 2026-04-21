#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <dataset> <batch_size> [extra attack_somp.py args...]" >&2
  exit 1
fi

array=( "$@" )
last_args=( "${array[@]:2}" )

python attack_somp.py --dataset "$1" --split val --n_inputs 10 --batch_size "$2" \
  --l1_filter all --l2_filter non-overlap --model_path meta-llama/Meta-Llama-3.1-8B --device auto --task seq_class --cache_dir $HF_HOME/gia_exp_cache --pad left --attn_implementation eager "${last_args[@]}"
