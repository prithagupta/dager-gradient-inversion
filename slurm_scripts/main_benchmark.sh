#!/bin/bash


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common_attack_args.sh"
cd "$REPO_ROOT"

extra_args=( "$@" )
seeds=( 40 41 42 )

datasets=(  "sst2" "cola" "rotten_tomatoes")
models=("gpt2" "gpt2-large")
methods=("dager" "hybrid")

has_split_arg() {
  local arg
  if [ "${#extra_args[@]}" -eq 0 ]; then
    return 1
  fi
  for arg in "${extra_args[@]}"; do
    if [ "$arg" = "--split" ] || [[ "$arg" == --split=* ]]; then
      return 0
    fi
  done
  return 1
}

has_use_hf_split_arg() {
  local arg
  if [ "${#extra_args[@]}" -eq 0 ]; then
    return 1
  fi
  for arg in "${extra_args[@]}"; do
    if [ "$arg" = "--use_hf_split" ]; then
      return 0
    fi
  done
  return 1
}

default_batch_size() {
  echo "8"
}

run_wrapper() {
  local method="$1"
  local model="$2"
  local dataset="$3"
  local seed="$4"
  local batch
  local run_args=()
  batch="$(default_batch_size "$dataset" "$model")"

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
  if [ "$dataset" = "sst2" ] || [ "$dataset" = "cola" ]; then
    if ! has_use_hf_split_arg; then
      run_args+=( --use_hf_split )
    fi
  fi
  echo ""
  echo "=================================================="
  echo "Running ${method} | model=${model} | dataset=${dataset} | batch_size=${batch} | rng_seed=${seed}"
  echo "Command: ${script} ${dataset} ${batch} ${run_args[*]}"
  echo "=================================================="
  bash "$script" "$dataset" "$batch" "${run_args[@]}"
}


for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
      for method in "${methods[@]}"; do
        run_wrapper "$method" "$model" "$dataset" "$seed"
      done
    done
  done
done
