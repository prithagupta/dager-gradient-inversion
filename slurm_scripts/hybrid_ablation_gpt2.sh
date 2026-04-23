#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common_attack_args.sh"
cd "$REPO_ROOT"

extra_args=( "$@" )
seeds=( 40 41 42 )
dataset="sst2"
batch="8"

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

run_variant() {
  local variant_name="$1"
  local seed="$2"
  shift 2
  local variant_args=( "$@" )

  local run_args=()
  if [ "${#variant_args[@]}" -gt 0 ]; then
    run_args+=( "${variant_args[@]}" )
  fi
  if [ "${#extra_args[@]}" -gt 0 ]; then
    run_args+=( "${extra_args[@]}" )
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
  run_args+=( --rng_seed "$seed" )
  if ! has_use_hf_split_arg; then
    run_args+=( --use_hf_split )
  fi
  echo ""
  echo "=================================================="
  echo "Running ${variant_name} | model=gpt2-large | dataset=${dataset} | batch_size=${batch} | rng_seed=${seed}"
  echo "=================================================="

  if [ "$variant_name" = "dager_only" ]; then
    echo "Command: scripts/dager_gpt2-large.sh ${dataset} ${batch} ${run_args[*]}"
    bash "${REPO_ROOT}/scripts/dager_gpt2-large.sh" \
      "$dataset" "$batch" "${run_args[@]}"
  else
    echo "Command: scripts/hybrid_gpt2-large.sh ${dataset} ${batch} ${run_args[*]}"
    bash "${REPO_ROOT}/scripts/hybrid_gpt2-large.sh" \
      "$dataset" "$batch" "${run_args[@]}"
  fi
}

for seed in "${seeds[@]}"; do
  run_variant "dager_only" "$seed"
  run_variant "hybrid_full" "$seed" \
    --hybrid_init_mode dager \
    --hybrid_use_lm_prior true \
    --hybrid_projection_mode candidate_final
  run_variant "hybrid_no_dager_init" "$seed" \
    --hybrid_init_mode candidate_random \
    --hybrid_use_lm_prior true \
    --hybrid_projection_mode candidate_final
  run_variant "hybrid_no_lm_prior" "$seed" \
    --hybrid_init_mode dager \
    --hybrid_use_lm_prior false
  run_variant "hybrid_no_candidate_projection" "$seed" \
    --hybrid_init_mode dager \
    --hybrid_use_lm_prior true \
    --hybrid_projection_mode none
done
