#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common_attack_args.sh"
source "$REPO_ROOT/slurm_scripts/common_benchmark_args.sh"
cd "$REPO_ROOT"

extra_args=( "$@" )

datasets=( ${AR_DAGER_DATASETS:-sst2 cola} )
models=( ${AR_DAGER_MODELS:-gpt2 openai-community/gpt2-large} )
batches=( ${AR_DAGER_BATCHES:-1 2 4 8 16 32 64 128} )

# Slurm-compatible default:
# - USE_SYNTHETIC_CANARY=0 runs no-canary only
# - USE_SYNTHETIC_CANARY=1 runs canary only
# - AR_DAGER_CANARY_MODES="0 1" overrides this and sweeps both
canary_modes=( ${AR_DAGER_CANARY_MODES:-${USE_SYNTHETIC_CANARY:-0}} )

echo "[CONFIG] script=autoregressivedager_batch_ablation.sh"
echo "[CONFIG] datasets=${datasets[*]}"
echo "[CONFIG] batches=${batches[*]}"
echo "[CONFIG] models=${models[*]}"
echo "[CONFIG] canary_modes=${canary_modes[*]}"
echo "[CONFIG] seed_rule=wrapper/default seed"
echo "[CONFIG] canary_marker_prefix=${CANARY_MARKER_PREFIX:-qxjkvcanary}"
echo "[CONFIG] extra_args=$(printf '%q ' "${extra_args[@]}")"

run_wrapper() {
  local model="$1"
  local dataset="$2"
  local batch="$3"
  local use_canary="$4"
  local run_args=()

  if [ "${#extra_args[@]}" -gt 0 ]; then
    run_args=( "${extra_args[@]}" )
  fi

  set_default_rank_tol_arg "$batch" "${run_args[@]}"
  set_default_max_ids_arg "$batch" "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg --n_inputs 50 "${ATTACK_EXTRA_ARGS[@]}"
  append_safe_eval_dataset_args "$dataset" "$batch" 50 "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg --device_grad cpu "${ATTACK_EXTRA_ARGS[@]}"

  if [ "$use_canary" = "1" ]; then
    set_default_flag_arg --preprocess_unique_canary_markers "${ATTACK_EXTRA_ARGS[@]}"
    set_default_arg --canary_marker_prefix "${CANARY_MARKER_PREFIX:-qxjkvcanary}" "${ATTACK_EXTRA_ARGS[@]}"
  fi

  run_args=( "${ATTACK_EXTRA_ARGS[@]}" )

  echo "Resolved attack args: $(printf '%q ' "${run_args[@]}")"
  echo ""
  echo "=================================================="
  echo "Running autoregressivedager | model=${model} | dataset=${dataset} | batch_size=${batch} | synthetic_canary=${use_canary}"
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
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      for use_canary in "${canary_modes[@]}"; do
        run_wrapper "$model" "$dataset" "$batch" "$use_canary"
      done
    done
  done
done