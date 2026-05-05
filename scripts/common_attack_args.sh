#!/bin/bash

has_cli_arg() {
  local needle="$1"
  shift || true

  local arg
  for arg in "$@"; do
    if [ "$arg" = "$needle" ] || [[ "$arg" == "$needle="* ]]; then
      return 0
    fi
  done

  return 1
}

default_max_ids_for_batch() {
  local batch="$1"

  if [ "$batch" -ge 128 ]; then
    echo "32"
  elif [ "$batch" -ge 64 ]; then
    echo "64"
  else
    echo "-1"
  fi
}

set_default_max_ids_arg() {
  local batch="$1"
  shift || true

  ATTACK_EXTRA_ARGS=( "$@" )

  if has_cli_arg "--max_ids" "${ATTACK_EXTRA_ARGS[@]}"; then
    return
  fi

  local max_ids
  max_ids="$(default_max_ids_for_batch "$batch")"
  if [ "$max_ids" != "-1" ]; then
    ATTACK_EXTRA_ARGS+=( --max_ids "$max_ids" )
  fi
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

set_default_rank_tol_arg() {
  local batch="$1"
  shift || true

  ATTACK_EXTRA_ARGS=( "$@" )

  if has_cli_arg "--rank_tol" "${ATTACK_EXTRA_ARGS[@]}"; then
    return
  fi

  ATTACK_EXTRA_ARGS+=( --rank_tol "$(default_rank_tol_for_batch "$batch")" )
}

set_default_rng_seed_arg() {
  local seed="${1:-42}"
  shift || true

  ATTACK_EXTRA_ARGS=( "$@" )

  if has_cli_arg "--rng_seed" "${ATTACK_EXTRA_ARGS[@]}"; then
    return
  fi

  ATTACK_EXTRA_ARGS+=( --rng_seed "$seed" )
}

set_default_arg() {
  local flag="$1"
  local value="$2"
  shift 2 || true

  ATTACK_EXTRA_ARGS=( "$@" )

  if has_cli_arg "$flag" "${ATTACK_EXTRA_ARGS[@]}"; then
    return
  fi

  ATTACK_EXTRA_ARGS+=( "$flag" "$value" )
}

set_default_flag_arg() {
  local flag="$1"
  shift || true

  ATTACK_EXTRA_ARGS=( "$@" )

  if has_cli_arg "$flag" "${ATTACK_EXTRA_ARGS[@]}"; then
    return
  fi

  ATTACK_EXTRA_ARGS+=( "$flag" )
}

uses_hf_validation_split_dataset() {
  case "$1" in
    sst2|cola) return 0 ;;
    *) return 1 ;;
  esac
}

set_default_hf_split_arg() {
  local dataset="$1"
  shift || true

  ATTACK_EXTRA_ARGS=( "$@" )

  if ! uses_hf_validation_split_dataset "$dataset"; then
    return
  fi

  if has_cli_arg "--use_hf_split" "${ATTACK_EXTRA_ARGS[@]}"; then
    return
  fi

  ATTACK_EXTRA_ARGS+=( --use_hf_split )
}

set_default_idager_hybrid_args() {
  local lm_mode="$1"
  shift || true

  ATTACK_EXTRA_ARGS=( "$@" )

  set_default_arg "--n_steps" "300" "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg "--print_every" "50" "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg "--hybrid_init_mode" "dager" "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg "--hybrid_projection_mode" "candidate_periodic" "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg "--hybrid_project_every" "10" "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg "--iterative_rounds" "3" "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg "--iterative_steps_per_round" "0" "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg "--iterative_accept_margin" "1e-6" "${ATTACK_EXTRA_ARGS[@]}"
  set_default_arg "--iterative_stall_patience" "1" "${ATTACK_EXTRA_ARGS[@]}"
  set_default_flag_arg "--iterative_dager_lamp" "${ATTACK_EXTRA_ARGS[@]}"
  set_default_flag_arg "--iterative_refresh_candidates" "${ATTACK_EXTRA_ARGS[@]}"

  if [ "$lm_mode" = "gpt2" ]; then
    set_default_arg "--hybrid_use_lm_prior" "true" "${ATTACK_EXTRA_ARGS[@]}"
    set_default_arg "--coeff_perplexity" "0.2" "${ATTACK_EXTRA_ARGS[@]}"
  else
    set_default_arg "--hybrid_use_lm_prior" "false" "${ATTACK_EXTRA_ARGS[@]}"
    set_default_arg "--coeff_perplexity" "0.0" "${ATTACK_EXTRA_ARGS[@]}"
  fi
}
