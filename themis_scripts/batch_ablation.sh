#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common_themis_env.sh"
source "$REPO_ROOT/scripts/common_attack_args.sh"
source "$SCRIPT_DIR/common_benchmark_args.sh"
cd "$REPO_ROOT"

extra_args=( "$@" )

MODELS_VALUE="${MODELS:-gpt2 gpt2-large}"
BATCHES_VALUE="${BATCHES:-128}" #1 2 4 8 16 32 64
METHODS_VALUE="${METHODS:-dager hybrid}"
DATASETS_VALUE="${DATASETS:-rotten_tomatoes}" #sst2 cola
MODELS_VALUE="${MODELS_VALUE//,/ }"
BATCHES_VALUE="${BATCHES_VALUE//,/ }"
METHODS_VALUE="${METHODS_VALUE//,/ }"
DATASETS_VALUE="${DATASETS_VALUE//,/ }"
IFS=' ' read -r -a models <<< "$MODELS_VALUE"
IFS=' ' read -r -a batches <<< "$BATCHES_VALUE"
IFS=' ' read -r -a methods <<< "$METHODS_VALUE"
IFS=' ' read -r -a datasets <<< "$DATASETS_VALUE"

echo "[CONFIG] script=batch_ablation.sh"
echo "[CONFIG] datasets=${datasets[*]}"
echo "[CONFIG] batches=${batches[*]}"
echo "[CONFIG] models=${models[*]}"
echo "[CONFIG] methods=${methods[*]}"
echo "[CONFIG] seed_rule=wrapper/default seed"
echo "[CONFIG] extra_args=$(printf '%q ' "${extra_args[@]}")"

run_wrapper() {
  local method="$1"
  local model="$2"
  local dataset="$3"
  local batch="$4"
  local run_args=()

  if [ "${#extra_args[@]}" -gt 0 ]; then
    run_args=( "${extra_args[@]}" )
  fi

  local script="${REPO_ROOT}/scripts/${method}_${model}.sh"
  if [ "$batch" -gt 64 ]; then
    if ! has_cli_arg "--max_ids" "${run_args[@]}"; then
      run_args+=( --max_ids 96 )
    fi
    if ! has_cli_arg "--rank_tol" "${run_args[@]}"; then
      run_args+=( --rank_tol 1e-8 )
    fi
    if ! has_cli_arg "--l1_span_thresh" "${run_args[@]}"; then
      run_args+=( --l1_span_thresh 5e-5 )
    fi
    if ! has_cli_arg "--l2_span_thresh" "${run_args[@]}"; then
      run_args+=( --l2_span_thresh 2e-3 )
    fi
    if ! has_cli_arg "--distinct_thresh" "${run_args[@]}"; then
      run_args+=( --distinct_thresh 0.6 )
    fi
  fi
  if [ "${#run_args[@]}" -gt 0 ]; then
    set_default_rank_tol_arg "$batch" "${run_args[@]}"
  else
    set_default_rank_tol_arg "$batch"
  fi
  run_args=( "${ATTACK_EXTRA_ARGS[@]}" )
  if [ "${#run_args[@]}" -gt 0 ]; then
    set_default_max_ids_arg "$batch" "${run_args[@]}"
  else
    set_default_max_ids_arg "$batch"
  fi
  run_args=( "${ATTACK_EXTRA_ARGS[@]}" )
  append_safe_eval_dataset_args "$dataset" "$batch" 50 "${run_args[@]}"
  set_default_arg --device_grad cuda "${run_args[@]}"
  set_default_arg --cache_dir "$DAGER_CACHE_DIR" "${ATTACK_EXTRA_ARGS[@]}"
  run_args=( "${ATTACK_EXTRA_ARGS[@]}" )
  echo "Resolved attack args: $(printf '%q ' "${run_args[@]}")"
  echo ""
  echo "=================================================="
  echo "Running ${method} | model=${model} | dataset=${dataset} | batch_size=${batch} | rng_seed=wrapper-default"
  echo "Command: ${script} ${dataset} ${batch} ${run_args[*]}"
  echo "=================================================="
  bash "$script" "$dataset" "$batch" "${run_args[@]}"
}

for batch in "${batches[@]}"; do
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      for method in "${methods[@]}"; do
        run_wrapper "$method" "$model" "$dataset" "$batch"
      done
    done
  done
done
