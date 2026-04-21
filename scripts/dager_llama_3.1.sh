#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <dataset> <batch_size> [extra attack.py args...]" >&2
  exit 1
fi

array=( "$@" )
last_args=( "${array[@]:2}" )

python attack.py --dataset "$1" --split val --n_inputs 100 --batch_size "$2" --l1_filter all --l2_filter non-overlap \
  --model_path meta-llama/Meta-Llama-3.1-8B --device auto --task seq_class --cache_dir $HF_HOME/gia_exp_cache \
  --rank_tol 1e-8 --l1_span_thresh 1e-4 --l2_span_thresh 5e-5 --pad left --precision half --parallel 10 --n_incorrect 8 \
  --attn_implementation eager "${last_args[@]}"
