#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common_attack_args.sh"
cd "$REPO_ROOT"

extra_args=( "$@" )

models=( "gpt2" "gpt2-large" )
batches=( 128 64 32 16 4 2 1)
methods=( "dager" "hybrid" )
datasets=( "sst2" "cola" )

has_n_inputs_arg() {
  local arg
  if [ "${#extra_args[@]}" -eq 0 ]; then
    return 1
  fi
  for arg in "${extra_args[@]}"; do
    if [ "$arg" = "--n_inputs" ] || [[ "$arg" == --n_inputs=* ]]; then
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

run_wrapper() {
  local method="$1"
  local model="$2"
  local dataset="$3"
  local batch="$4"
  local run_args=()
  local max_hf_inputs
  local chosen_n_inputs

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
  if [ "${#run_args[@]}" -gt 0 ]; then
    set_default_rng_seed_arg 42 "${run_args[@]}"
  else
    set_default_rng_seed_arg 42
  fi
  run_args=( "${ATTACK_EXTRA_ARGS[@]}" )
  if [ "$dataset" = "sst2" ] || [ "$dataset" = "cola" ]; then
    if ! has_use_hf_split_arg; then
      run_args+=( --use_hf_split )
    fi
  elif ! has_split_arg; then
    run_args+=( --split val )
  fi
  if [ "$dataset" = "sst2" ] && ! has_n_inputs_arg; then
    # Official SST-2 validation split has 872 examples.
    max_hf_inputs=$(( 872 / batch ))
    if [ "$max_hf_inputs" -lt 1 ]; then
      max_hf_inputs=1
    fi
    chosen_n_inputs=100
    if [ "$max_hf_inputs" -lt "$chosen_n_inputs" ]; then
      chosen_n_inputs="$max_hf_inputs"
    fi
    run_args+=( --n_inputs "$chosen_n_inputs" )
  elif [ "$dataset" = "cola" ] && ! has_n_inputs_arg; then
    # Official CoLA validation split has 1043 examples.
    max_hf_inputs=$(( 1043 / batch ))
    if [ "$max_hf_inputs" -lt 1 ]; then
      max_hf_inputs=1
    fi
    chosen_n_inputs=100
    if [ "$max_hf_inputs" -lt "$chosen_n_inputs" ]; then
      chosen_n_inputs="$max_hf_inputs"
    fi
    run_args+=( --n_inputs "$chosen_n_inputs" )
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
    for batch in "${batches[@]}"; do
      for method in "${methods[@]}"; do
        run_wrapper "$method" "$model" "$dataset" "$batch"
      done
    done
  done
done
