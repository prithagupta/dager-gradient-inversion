#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common_attack_args.sh"
source "$REPO_ROOT/slurm_scripts/common_benchmark_args.sh"
cd "$REPO_ROOT"

extra_args=( "$@" )
seeds=( 40 41 42 )
datasets=( "sst2" "cola" "rotten_tomatoes")
batches=( 16 32 64 )
models=( "gpt2" "gpt2-large" )
methods=( "dager" "hybrid" )

echo "[CONFIG] script=main_benchmark_canary.sh"
echo "[CONFIG] datasets=${datasets[*]}"
echo "[CONFIG] batches=${batches[*]}"
echo "[CONFIG] models=${models[*]}"
echo "[CONFIG] methods=${methods[*]}"
echo "[CONFIG] seeds=${seeds[*]}"
echo "[CONFIG] canary_marker_prefix=${CANARY_MARKER_PREFIX:-qxjkvcanary}"
echo "[CONFIG] extra_args=$(printf '%q ' "${extra_args[@]}")"

run_wrapper() {
  local method="$1"
  local model="$2"
  local dataset="$3"
  local batch="$4"
  local seed="$5"
  local run_args=()

  if [ "${#extra_args[@]}" -gt 0 ]; then
    run_args=( "${extra_args[@]}" )
  fi

  local script="${REPO_ROOT}/scripts/${method}_${model}.sh"
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
  run_args+=( --rng_seed "$seed" )
  append_safe_eval_dataset_args "$dataset" "$batch" 50 "${run_args[@]}"
  set_default_arg --device_grad cpu "${run_args[@]}"
  run_args=( "${ATTACK_EXTRA_ARGS[@]}" )
  set_default_flag_arg --preprocess_unique_canary_markers "${run_args[@]}"
  run_args=( "${ATTACK_EXTRA_ARGS[@]}" )
  set_default_arg --canary_marker_prefix "${CANARY_MARKER_PREFIX:-qxjkvcanary}" "${run_args[@]}"
  run_args=( "${ATTACK_EXTRA_ARGS[@]}" )
  echo "Resolved attack args: $(printf '%q ' "${run_args[@]}")"

  echo ""
  echo "=================================================="
  echo "Running ${method} | model=${model} | dataset=${dataset} | batch_size=${batch} | rng_seed=${seed} | synthetic_canary=1"
  echo "Command: ${script} ${dataset} ${batch} ${run_args[*]}"
  echo "=================================================="
  bash "$script" "$dataset" "$batch" "${run_args[@]}"
}
for batch in "${batches[@]}"; do
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      for seed in "${seeds[@]}"; do
        for method in "${methods[@]}"; do
          run_wrapper "$method" "$model" "$dataset" "$batch" "$seed"
        done
      done
    done
  done
done
