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
