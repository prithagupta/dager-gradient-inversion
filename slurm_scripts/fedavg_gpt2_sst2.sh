#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common_attack_args.sh"
source "$REPO_ROOT/slurm_scripts/common_benchmark_args.sh"
cd "$REPO_ROOT"

IFS=' ' read -r -a batches <<< "${BATCHES:-8 16 32}"
IFS=' ' read -r -a seeds <<< "${SEEDS:-40 41 42}"
IFS=' ' read -r -a attacks <<< "${ATTACKS:-dager hybrid dager_canary hybrid_canary}"
IFS=' ' read -r -a models <<< "${MODELS:-gpt2}"
IFS=' ' read -r -a datasets <<< "${DATASETS:-sst2}"

N_INPUTS="${N_INPUTS:-50}"
AVG_EPOCHS="${AVG_EPOCHS:-10}"
AVG_LR="${AVG_LR:-1e-4}"
B_MINI="${B_MINI:-4}"
DEVICE_GRAD="${DEVICE_GRAD:-cpu}"
EXTRA_ATTACK_ARGS="${EXTRA_ATTACK_ARGS:-}"

echo "[CONFIG] script=fedavg_gpt2_sst2.sh"
echo "[CONFIG] batches=${batches[*]}"
echo "[CONFIG] seeds=${seeds[*]}"
echo "[CONFIG] attacks=${attacks[*]}"
echo "[CONFIG] models=${models[*]}"
echo "[CONFIG] datasets=${datasets[*]}"
echo "[CONFIG] FedAvg: E=$AVG_EPOCHS lr=$AVG_LR b_mini=$B_MINI"
echo "[CONFIG] device_grad=$DEVICE_GRAD"

dager_wrapper_for_model() {
  case "$1" in
    gpt2)
      printf '%s\n' "$REPO_ROOT/scripts/dager_gpt2.sh"
      ;;
    gpt2-large|openai-community/gpt2-large)
      printf '%s\n' "$REPO_ROOT/scripts/dager_gpt2-large.sh"
      ;;
    *)
      echo "[ERROR] Unsupported FedAvg DAGER model: $1" >&2
      exit 2
      ;;
  esac
}

hybrid_wrapper_for_model() {
  case "$1" in
    gpt2)
      printf '%s\n' "$REPO_ROOT/scripts/hybrid_gpt2.sh"
      ;;
    gpt2-large|openai-community/gpt2-large)
      printf '%s\n' "$REPO_ROOT/scripts/hybrid_gpt2-large.sh"
      ;;
    *)
      echo "[ERROR] Unsupported FedAvg Hybrid model: $1" >&2
      exit 2
      ;;
  esac
}

run_attack() {
  local model="$1"
  local dataset="$2"
  local attack="$3"
  local batch="$4"
  local seed="$5"
  local base_attack="$attack"
  local args=(
    --algo fedavg
    --avg_epochs "$AVG_EPOCHS"
    --avg_lr "$AVG_LR"
    --b_mini "$B_MINI"
    --rng_seed "$seed"
    --n_inputs "$N_INPUTS"
    --device_grad "$DEVICE_GRAD"
    --rank_tol 5e-6
    --l1_span_thresh 5e-3
    --l2_span_thresh 5e-3
    --precision double
  )
  case "$dataset" in
    sst2|cola)
      args+=( --use_hf_split )
      ;;
  esac
  if [ -n "$EXTRA_ATTACK_ARGS" ]; then
    # shellcheck disable=SC2206
    args+=( $EXTRA_ATTACK_ARGS )
  fi

  case "$attack" in
    *_canary)
      base_attack="${attack%_canary}"
      args+=( --preprocess_unique_canary_markers --canary_marker_prefix "${CANARY_MARKER_PREFIX:-qxjkvcanary}" )
      ;;
  esac

  echo ""
  echo "=================================================="
  echo "Running FedAvg attack=$attack | model=$model | dataset=$dataset | batch_size=$batch | seed=$seed"
  echo "Args: ${args[*]}"
  echo "=================================================="

  case "$base_attack" in
    dager)
      bash "$(dager_wrapper_for_model "$model")" "$dataset" "$batch" "${args[@]}"
      ;;
    hybrid|iterative)
      bash "$(hybrid_wrapper_for_model "$model")" "$dataset" "$batch" "${args[@]}"
      ;;
    *)
      echo "[ERROR] Unknown attack: $attack" >&2
      exit 2
      ;;
  esac
}

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for batch in "${batches[@]}"; do
      for seed in "${seeds[@]}"; do
        for attack in "${attacks[@]}"; do
          run_attack "$model" "$dataset" "$attack" "$batch" "$seed"
        done
      done
    done
  done
done
