#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

extra_args=( "$@" )
dataset="sst2"
batch="8"

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

run_variant() {
  local variant_name="$1"
  shift
  local variant_args=( "$@" )

  local run_args=()
  if [ "${#variant_args[@]}" -gt 0 ]; then
    run_args+=( "${variant_args[@]}" )
  fi
  if [ "${#extra_args[@]}" -gt 0 ]; then
    run_args+=( "${extra_args[@]}" )
  fi
  if ! has_rank_tol_arg; then
    run_args+=( --rank_tol "$(default_rank_tol_for_batch "$batch")" )
  fi
  if ! has_use_hf_split_arg; then
    run_args+=( --use_hf_split )
  fi
  echo ""
  echo "=================================================="
  echo "Running ${variant_name} | model=gpt2-large | dataset=${dataset} | batch_size=${batch}"
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

run_variant "dager_only"
run_variant "hybrid_full" \
  --hybrid_init_mode dager \
  --hybrid_use_lm_prior true \
  --hybrid_projection_mode candidate_final
run_variant "hybrid_no_dager_init" \
  --hybrid_init_mode candidate_random \
  --hybrid_use_lm_prior true \
  --hybrid_projection_mode candidate_final
run_variant "hybrid_no_lm_prior" \
  --hybrid_init_mode dager \
  --hybrid_use_lm_prior false
run_variant "hybrid_no_candidate_projection" \
  --hybrid_init_mode dager \
  --hybrid_use_lm_prior true \
  --hybrid_projection_mode none
