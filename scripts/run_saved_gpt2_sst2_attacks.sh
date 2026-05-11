#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export HF_HOME="${HF_HOME:-$REPO_ROOT/models_cache}"
source "$SCRIPT_DIR/common_attack_args.sh"
cd "$REPO_ROOT"

FINETUNED_PATH="${FINETUNED_PATH:-${1:-}}"
if [ -z "$FINETUNED_PATH" ]; then
  echo "Usage: FINETUNED_PATH=/path/to/model $0" >&2
  echo "   or: $0 /path/to/model" >&2
  exit 2
fi

if [ ! -d "$FINETUNED_PATH" ]; then
  echo "[ERROR] FINETUNED_PATH is not a directory: $FINETUNED_PATH" >&2
  exit 1
fi

IFS=' ' read -r -a batches <<< "${BATCHES:-8 16 32}"
IFS=' ' read -r -a seeds <<< "${SEEDS:-42}"
IFS=' ' read -r -a attacks <<< "${ATTACKS:-dager hybrid dager_canary hybrid_canary}"

N_INPUTS="${N_INPUTS:-50}"
DEVICE_GRAD="${DEVICE_GRAD:-cpu}"
EXTRA_ATTACK_ARGS="${EXTRA_ATTACK_ARGS:-}"
FINETUNED_BASE_MODEL="${FINETUNED_BASE_MODEL:-gpt2}"
ATTACK_DATASET="${ATTACK_DATASET:-sst2}"

dager_wrapper_for_model() {
  case "$FINETUNED_BASE_MODEL" in
    gpt2)
      printf '%s\n' "$REPO_ROOT/scripts/dager_gpt2.sh"
      ;;
    gpt2-large|openai-community/gpt2-large)
      printf '%s\n' "$REPO_ROOT/scripts/dager_gpt2-large.sh"
      ;;
    *)
      echo "[ERROR] Unsupported FINETUNED_BASE_MODEL for DAGER: $FINETUNED_BASE_MODEL" >&2
      exit 2
      ;;
  esac
}

hybrid_wrapper_for_model() {
  case "$FINETUNED_BASE_MODEL" in
    gpt2)
      printf '%s\n' "$REPO_ROOT/scripts/hybrid_gpt2.sh"
      ;;
    gpt2-large|openai-community/gpt2-large)
      printf '%s\n' "$REPO_ROOT/scripts/hybrid_gpt2-large.sh"
      ;;
    *)
      echo "[ERROR] Unsupported FINETUNED_BASE_MODEL for Hybrid: $FINETUNED_BASE_MODEL" >&2
      exit 2
      ;;
  esac
}

run_attack() {
  local attack="$1"
  local batch="$2"
  local seed="$3"
  local base_attack="$attack"
  local common_args=(
    --finetuned_path "$FINETUNED_PATH"
    --rng_seed "$seed"
    --n_inputs "$N_INPUTS"
    --device_grad "$DEVICE_GRAD"
    --cache_dir "$DAGER_CACHE_DIR"
  )
  case "$ATTACK_DATASET" in
    sst2|cola)
      common_args+=( --use_hf_split )
      ;;
  esac

  if [ -n "$EXTRA_ATTACK_ARGS" ]; then
    # shellcheck disable=SC2206
    common_args+=( $EXTRA_ATTACK_ARGS )
  fi

  case "$attack" in
    *_canary)
      base_attack="${attack%_canary}"
      common_args+=( --preprocess_unique_canary_markers --canary_marker_prefix "${CANARY_MARKER_PREFIX:-qxjkvcanary}" )
      ;;
  esac

  echo ""
  echo "=================================================="
  echo "Running saved-model attack=$attack | model=$FINETUNED_BASE_MODEL | dataset=$ATTACK_DATASET | batch_size=$batch | seed=$seed"
  echo "FINETUNED_PATH=$FINETUNED_PATH"
  echo "=================================================="

  case "$base_attack" in
    dager)
      bash "$(dager_wrapper_for_model)" "$ATTACK_DATASET" "$batch" "${common_args[@]}"
      ;;
    hybrid|iterative)
      bash "$(hybrid_wrapper_for_model)" "$ATTACK_DATASET" "$batch" "${common_args[@]}"
      ;;
    *)
      echo "[ERROR] Unknown attack: $attack" >&2
      exit 2
      ;;
  esac
}

for batch in "${batches[@]}"; do
  for seed in "${seeds[@]}"; do
    for attack in "${attacks[@]}"; do
      run_attack "$attack" "$batch" "$seed"
    done
  done
done
