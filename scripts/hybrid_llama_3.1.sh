#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_attack_args.sh"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <dataset> <batch_size> [extra attack_hybrid.py args...]" >&2
  exit 1
fi

array=( "$@" )
last_args=( "${array[@]:2}" )

set_default_max_ids_arg "$2" "${last_args[@]}"
set_default_rng_seed_arg 42 "${ATTACK_EXTRA_ARGS[@]}"
set_default_idager_hybrid_args no_lm "${ATTACK_EXTRA_ARGS[@]}"
set_default_arg "--device_grad" "cuda" "${ATTACK_EXTRA_ARGS[@]}"
set_default_arg "--parallel" "4" "${ATTACK_EXTRA_ARGS[@]}"
set_default_arg "--n_inputs" "50" "${ATTACK_EXTRA_ARGS[@]}"
set_default_hf_split_arg "$1" "${ATTACK_EXTRA_ARGS[@]}"

python attack_hybrid.py --dataset "$1" --split val --batch_size "$2" \
  --l1_filter all --l2_filter non-overlap --model_path meta-llama/Meta-Llama-3.1-8B --device auto --task seq_class \
  --cache_dir "$HF_HOME/gia_exp_cache" --rank_tol 1e-8 --l1_span_thresh 1e-4 --l2_span_thresh 5e-5 --pad left \
  --precision full --n_incorrect 8 --attn_implementation sdpa "${ATTACK_EXTRA_ARGS[@]}"
