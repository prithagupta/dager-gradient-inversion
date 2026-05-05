#!/bin/bash

uses_hf_validation_split() {
  case "$1" in
    sst2|cola) return 0 ;;
    *) return 1 ;;
  esac
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

append_safe_eval_dataset_args() {
  local dataset="$1"
  shift || true
  if [ "$#" -gt 0 ]; then
    shift || true
  fi
  if [ "$#" -gt 0 ]; then
    shift || true
  fi

  ATTACK_EXTRA_ARGS=( "$@" )

  if uses_hf_validation_split "$dataset"; then
    if ! has_cli_arg "--use_hf_split" "${ATTACK_EXTRA_ARGS[@]}"; then
      ATTACK_EXTRA_ARGS+=( --use_hf_split )
    fi
  fi
}
