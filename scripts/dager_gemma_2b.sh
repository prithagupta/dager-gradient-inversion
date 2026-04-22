#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_attack_args.sh"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <dataset> <batch_size> [extra attack.py args...]" >&2
  exit 1
fi

array=( "$@" )
last_args=( "${array[@]:2}" )

set_default_max_ids_arg "$2" "${last_args[@]}"
set_default_rank_tol_arg "$2" "${ATTACK_EXTRA_ARGS[@]}"
set_default_rng_seed_arg 42 "${ATTACK_EXTRA_ARGS[@]}"

python attack.py --dataset "$1" --split val --n_inputs 100 --batch_size "$2" \
  --l1_filter all --l2_filter non-overlap --model_path google/gemma-2b --device auto \
  --task seq_class --cache_dir "$HF_HOME/gia_exp_cache" --pad left "${ATTACK_EXTRA_ARGS[@]}"
