#!/bin/bash


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

extra_args=( "$@" )

datasets=(  "sst2" "cola" "rotten_tomatoes")
models=(
  "gpt2"
  "gpt2-large"
)
methods=( "dager" "hybrid" )

has_rank_tol_arg() {
  local arg
  if [ "${#extra_args[@]}" -eq 0 ]; then
    return 1
  fi
  for arg in "${extra_args[@]}"; do
    if [ "$arg" = "--rank_tol" ] || [[ "$arg" == --rank_tol=* ]]; then
      return 0
    fi
  done
  return 1
}

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

default_rank_tol_for_batch() {
  local batch="$1"
  if [ "$batch" -le 2 ]; then
    echo "1e-7"
  elif [ "$batch" -le 16 ]; then
    echo "1e-8"
  else
    echo "1e-9"
  fi
}

default_batch_size() {
  echo "8"
}

run_wrapper() {
  local method="$1"
  local model="$2"
  local dataset="$3"
  local batch
  local run_args=()
  batch="$(default_batch_size "$dataset" "$model")"

  if [ "${#extra_args[@]}" -gt 0 ]; then
    run_args=( "${extra_args[@]}" )
  fi

  local script="${REPO_ROOT}/scripts/${method}_${model}.sh"
  if ! has_rank_tol_arg; then
    run_args+=( --rank_tol "$(default_rank_tol_for_batch "$batch")" )
  fi
  if [ "$dataset" = "sst2" ] || [ "$dataset" = "cola" ]; then
    if ! has_use_hf_split_arg; then
      run_args+=( --use_hf_split )
    fi
  fi
  echo ""
  echo "=================================================="
  echo "Running ${method} | model=${model} | dataset=${dataset} | batch_size=${batch}"
  echo "Command: ${script} ${dataset} ${batch} ${run_args[*]}"
  echo "=================================================="
  bash "$script" "$dataset" "$batch" "${run_args[@]}"
}

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
      run_wrapper "$method" "$model" "$dataset"
    done
  done
done
