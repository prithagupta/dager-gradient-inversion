#!/bin/bash

set -euo pipefail

extra_args=( "$@" )

models=( "gpt2" "gpt2-large" )
batches=( 1 2 4 8 16 32 64 128 )
methods=( "dager" "hybrid" )
dataset="sst2"

has_rank_tol_arg() {
  local arg
  for arg in "${extra_args[@]}"; do
    if [ "$arg" = "--rank_tol" ] || [[ "$arg" == --rank_tol=* ]]; then
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

run_wrapper() {
  local method="$1"
  local model="$2"
  local batch="$3"
  local run_args=( "${extra_args[@]}" )

  local script="/Users/prithagupta/projects/dager-gradient-inversion/scripts/${method}_${model}.sh"
  if ! has_rank_tol_arg; then
    run_args+=( --rank_tol "$(default_rank_tol_for_batch "$batch")" )
  fi
  echo ""
  echo "=================================================="
  echo "Running ${method} | model=${model} | dataset=${dataset} | batch_size=${batch}"
  echo "Command: ${script} ${dataset} ${batch} ${run_args[*]}"
  echo "=================================================="
  bash "$script" "$dataset" "$batch" "${run_args[@]}"
}

for model in "${models[@]}"; do
  for batch in "${batches[@]}"; do
    for method in "${methods[@]}"; do
      run_wrapper "$method" "$model" "$batch"
    done
  done
done
