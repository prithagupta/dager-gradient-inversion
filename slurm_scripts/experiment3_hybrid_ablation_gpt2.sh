#!/bin/bash

set -euo pipefail

extra_args=( "$@" )
dataset="sst2"
batches=( 1 2 4 8 )

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

run_variant() {
  local variant_name="$1"
  shift
  local variant_args=( "$@" )

  for batch in "${batches[@]}"; do
    local run_args=( "${variant_args[@]}" "${extra_args[@]}" )
    if ! has_rank_tol_arg; then
      run_args+=( --rank_tol "$(default_rank_tol_for_batch "$batch")" )
    fi
    echo ""
    echo "=================================================="
    echo "Running ${variant_name} | model=gpt2 | dataset=${dataset} | batch_size=${batch}"
    echo "=================================================="

    if [ "$variant_name" = "dager_only" ]; then
      echo "Command: scripts/dager_gpt2.sh ${dataset} ${batch} ${run_args[*]}"
      bash "/Users/prithagupta/projects/dager-gradient-inversion/scripts/dager_gpt2.sh" \
        "$dataset" "$batch" "${run_args[@]}"
    else
      echo "Command: scripts/hybrid_gpt2.sh ${dataset} ${batch} ${run_args[*]}"
      bash "/Users/prithagupta/projects/dager-gradient-inversion/scripts/hybrid_gpt2.sh" \
        "$dataset" "$batch" "${run_args[@]}"
    fi
  done
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
