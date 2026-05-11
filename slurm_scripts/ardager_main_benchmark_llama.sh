#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common_attack_args.sh"
source "$REPO_ROOT/slurm_scripts/common_benchmark_args.sh"
cd "$REPO_ROOT"

extra_args=( "$@" )
seeds=( ${AR_DAGER_SEEDS:-40 41 42} )
datasets=( ${AR_DAGER_DATASETS:-sst2 cola rotten_tomatoes} )
batches=( ${AR_DAGER_BATCHES:-8 16 32 64} )
model="${AR_DAGER_LLAMA_MODEL:-meta-llama/Meta-Llama-3.1-8B}"
use_canary="${USE_SYNTHETIC_CANARY:-0}"

if [[ "$use_canary" != "0" && "$use_canary" != "1" ]]; then
  echo "[ERROR] USE_SYNTHETIC_CANARY must be 0 or 1, got: $use_canary" >&2
  exit 1
fi

echo "[CONFIG] script=autoregressivedager_main_benchmark_llama.sh"
echo "[CONFIG] datasets=${datasets[*]}"
echo "[CONFIG] batches=${batches[*]}"
echo "[CONFIG] model=${model}"
echo "[CONFIG] seeds=${seeds[*]}"
echo "[CONFIG] use_synthetic_canary=${use_canary}"
echo "[CONFIG] canary_marker_prefix=${CANARY_MARKER_PREFIX:-qxjkvcanary}"
echo "[CONFIG] extra_args=$(printf '%q ' "${extra_args[@]}")"

run_wrapper() {
  local dataset="$1"
  local batch="$2"
  local seed="$3"
  local run_args=()

  if [ "${#extra_args[@]}" -gt 0 ]; then
    run_args=( "${extra_args[@]}" )
  fi

  set_default_arg --rng_seed "$seed" "${run_args[@]}"
  set_default_rank_tol_arg "$batch" "${ATTACK_EXTRA_ARGS[@]}"
  set_default_max_ids_arg "$batch" "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg --n_inputs 50 "${ATTACK_EXTRA_ARGS[@]}"
  append_safe_eval_dataset_args "$dataset" "$batch" 50 "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg --pad left "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg --device_grad cuda "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg --parallel 4 "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg --l1_span_thresh 1e-4 "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg --l2_span_thresh 5e-5 "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg --precision full "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg --n_incorrect 8 "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg --attn_implementation sdpa "${ATTACK_EXTRA_ARGS[@]}"

  if [ "$use_canary" = "1" ]; then
    set_default_flag_arg --preprocess_unique_canary_markers "${ATTACK_EXTRA_ARGS[@]}"
    set_default_arg --canary_marker_prefix "${CANARY_MARKER_PREFIX:-qxjkvcanary}" "${ATTACK_EXTRA_ARGS[@]}"
  fi

  run_args=( "${ATTACK_EXTRA_ARGS[@]}" )

  echo "Resolved attack args: $(printf '%q ' "${run_args[@]}")"
  echo ""
  echo "=================================================="
  echo "Running autoregressivedager | model=${model} | dataset=${dataset} | batch_size=${batch} | rng_seed=${seed} | synthetic_canary=${use_canary}"
  echo "Command: python attack_autoregressive_dager.py --dataset ${dataset} --split val --batch_size ${batch} --model_path ${model} --cache_dir ${DAGER_CACHE_DIR} --device auto --task seq_class --l1_filter all --l2_filter non-overlap ${run_args[*]}"
  echo "=================================================="

  python attack_autoregressive_dager.py \
    --dataset "$dataset" \
    --split val \
    --batch_size "$batch" \
    --model_path "$model" \
    --cache_dir "$DAGER_CACHE_DIR" \
    --device auto \
    --task seq_class \
    --l1_filter all \
    --l2_filter non-overlap \
    "${run_args[@]}"
}

for batch in "${batches[@]}"; do
  for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
      run_wrapper "$dataset" "$batch" "$seed"
    done
  done
done