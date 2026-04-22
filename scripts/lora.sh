#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_attack_args.sh"

array=( "$@" )
last_args=( "${array[@]:2}" )

set_default_max_ids_arg "$2" "${last_args[@]}"
set_default_rng_seed_arg 42 "${ATTACK_EXTRA_ARGS[@]}"

python attack.py --dataset "$1" --split val --n_inputs 100 --batch_size "$2" --l1_filter all --l2_filter non-overlap --model_path meta-llama/Meta-Llama-3.1-8B --device auto --task seq_class --l1_span_thresh 0.05 --l2_span_thresh 0.05 --cache_dir "$HF_HOME/gia_exp_cache" --train_method lora --lora_r 256 --rank_tol 5e-9 --finetuned_path ./models/lora_8530.pt --pad left "${ATTACK_EXTRA_ARGS[@]}"
