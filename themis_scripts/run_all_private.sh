#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_themis.sh"
LOG_DIR="$SCRIPT_DIR/logs"

GPU_FEDAVG_B16_A="${GPU_FEDAVG_B16_A:-3}"
GPU_FEDAVG_B16_B="${GPU_FEDAVG_B16_B:-4}"
GPU_FEDAVG_B16_C="${GPU_FEDAVG_B16_C:-3}"
GPU_TRAIN_CLEAN="${GPU_TRAIN_CLEAN:-2}"
GPU_TRAIN_0P1="${GPU_TRAIN_0P1:-3}"
GPU_TRAIN_0P5_1P0="${GPU_TRAIN_0P5_1P0:-4}"
GPU_TRAIN_0P5_1P0_B="${GPU_TRAIN_0P5_1P0_B:-3}"

RUN_FEDAVG_B16_A="${RUN_FEDAVG_B16_A:-1}"
RUN_FEDAVG_B16_B="${RUN_FEDAVG_B16_B:-1}"
RUN_FEDAVG_B16_C="${RUN_FEDAVG_B16_C:-0}"
RUN_TRAIN_CLEAN="${RUN_TRAIN_CLEAN:-0}"
RUN_TRAIN_0P1="${RUN_TRAIN_0P1:-0}"
RUN_TRAIN_0P5_1P0="${RUN_TRAIN_0P5_1P0:-1}"
RUN_TRAIN_0P5_1P0_B="${RUN_TRAIN_0P5_1P0_B:-1}"

MODELS="${MODELS:-gpt2}"
FEDAVG_DATASETS="${FEDAVG_DATASETS:-sst2 cola rotten_tomatoes}"
ATTACKS="${ATTACKS:-dager hybrid dager_canary hybrid_canary}"
FEDAVG_SEEDS="${FEDAVG_SEEDS:-40 41 42}"
TRAIN_ATTACK_SEEDS="${TRAIN_ATTACK_SEEDS:-42}"
TRAIN_ATTACK_BATCHES="${TRAIN_ATTACK_BATCHES:-8 16}"
TRAIN_DATASETS="${TRAIN_DATASETS:-sst2 cola rotten_tomatoes}"

AVG_EPOCHS="${AVG_EPOCHS:-10}"
AVG_LR="${AVG_LR:-1e-4}"
B_MINI="${B_MINI:-4}"
HARD_AVG_EPOCHS="${HARD_AVG_EPOCHS:-10}"
HARD_AVG_LR="${HARD_AVG_LR:-5e-4}"
HARD_B_MINI="${HARD_B_MINI:-4}"
N_INPUTS="${N_INPUTS:-50}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_LR="${TRAIN_LR:-5e-5}"
TRAIN_MAX_GRAD_NORM="${TRAIN_MAX_GRAD_NORM:-1.0}"
TRAIN_SEED="${TRAIN_SEED:-42}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

PIDS=()

mkdir -p "$LOG_DIR"

run_job() {
  local gpu="$1"
  local label="$2"
  shift 2
  local log_path="$LOG_DIR/gpu_${gpu}_private_${label}.log"

  echo "[THEMIS PRIVATE] Starting $label on CUDA_VISIBLE_DEVICES=$gpu"
  echo "[THEMIS PRIVATE] Log: $log_path"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$@"
  ) >"$log_path" 2>&1 &
  PIDS+=( "$!" )
}

run_fedavg_batch() {
  local batch="$1"
  local avg_epochs="${2:-$AVG_EPOCHS}"
  local avg_lr="${3:-$AVG_LR}"
  local b_mini="${4:-$B_MINI}"
  RUN_SCRIPT="fedavg_gpt2_sst2.sh" \
  BATCHES="$batch" \
  MODELS="$MODELS" \
  DATASETS="$FEDAVG_DATASETS" \
  SEEDS="$FEDAVG_SEEDS" \
  ATTACKS="$ATTACKS" \
  AVG_EPOCHS="$avg_epochs" \
  AVG_LR="$avg_lr" \
  B_MINI="$b_mini" \
  N_INPUTS="$N_INPUTS" \
  DEVICE_GRAD="cuda" \
    bash "$RUNNER"
}

run_fedavg_default_and_hard() {
  run_fedavg_batch "16" "$AVG_EPOCHS" "$AVG_LR" "$B_MINI"
  run_fedavg_batch "16" "$HARD_AVG_EPOCHS" "$HARD_AVG_LR" "$HARD_B_MINI"
}

run_train_then_attack_noises() {
  local noises="$1"
  RUN_SCRIPT="train_then_attack_gpt2_sst2.sh" \
  TRAIN_NOISES="$noises" \
  MODELS="$MODELS" \
  TRAIN_DATASETS="$TRAIN_DATASETS" \
  BATCHES="$TRAIN_ATTACK_BATCHES" \
  SEEDS="$TRAIN_ATTACK_SEEDS" \
  ATTACKS="$ATTACKS" \
  TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
  TRAIN_EPOCHS="$TRAIN_EPOCHS" \
  TRAIN_LR="$TRAIN_LR" \
  TRAIN_MAX_GRAD_NORM="$TRAIN_MAX_GRAD_NORM" \
  TRAIN_SEED="$TRAIN_SEED" \
  FORCE_TRAIN="$FORCE_TRAIN" \
  DEVICE_GRAD="cuda" \
  N_INPUTS="$N_INPUTS" \
    bash "$RUNNER"
}

echo "[THEMIS PRIVATE] models=$MODELS"
echo "[THEMIS PRIVATE] attacks=$ATTACKS"
echo "[THEMIS PRIVATE] FedAvg default: datasets=$FEDAVG_DATASETS batch16 gpu${GPU_FEDAVG_B16_A} enabled=$RUN_FEDAVG_B16_A gpu${GPU_FEDAVG_B16_B} enabled=$RUN_FEDAVG_B16_B gpu${GPU_FEDAVG_B16_C} extra=$RUN_FEDAVG_B16_C seeds=$FEDAVG_SEEDS E=$AVG_EPOCHS lr=$AVG_LR b_mini=$B_MINI"
echo "[THEMIS PRIVATE] FedAvg hard: E=$HARD_AVG_EPOCHS lr=$HARD_AVG_LR b_mini=$HARD_B_MINI"
echo "[THEMIS PRIVATE] Train/attack: datasets=$TRAIN_DATASETS clean=$RUN_TRAIN_CLEAN 0.1=$RUN_TRAIN_0P1 0.5+1.0 gpu${GPU_TRAIN_0P5_1P0} enabled=$RUN_TRAIN_0P5_1P0 gpu${GPU_TRAIN_0P5_1P0_B} enabled=$RUN_TRAIN_0P5_1P0_B batches=$TRAIN_ATTACK_BATCHES seeds=$TRAIN_ATTACK_SEEDS"

if [ "$RUN_FEDAVG_B16_A" = "1" ]; then
  run_job "$GPU_FEDAVG_B16_A" "fedavg_b16_all_datasets" run_fedavg_default_and_hard
fi
if [ "$RUN_FEDAVG_B16_B" = "1" ]; then
  run_job "$GPU_FEDAVG_B16_B" "fedavg_b16_all_datasets" run_fedavg_default_and_hard
fi
if [ "$RUN_FEDAVG_B16_C" = "1" ]; then
  run_job "$GPU_FEDAVG_B16_C" "fedavg_b16_all_datasets_extra" run_fedavg_default_and_hard
fi
if [ "$RUN_TRAIN_CLEAN" = "1" ]; then
  run_job "$GPU_TRAIN_CLEAN" "train_clean" run_train_then_attack_noises "clean"
fi
if [ "$RUN_TRAIN_0P1" = "1" ]; then
  run_job "$GPU_TRAIN_0P1" "train_0p1" run_train_then_attack_noises "0.1"
fi
if [ "$RUN_TRAIN_0P5_1P0" = "1" ]; then
  run_job "$GPU_TRAIN_0P5_1P0" "train_0p5_1p0" run_train_then_attack_noises "0.5 1.0"
fi
if [ "$RUN_TRAIN_0P5_1P0_B" = "1" ]; then
  run_job "$GPU_TRAIN_0P5_1P0_B" "train_0p5_1p0" run_train_then_attack_noises "0.5 1.0"
fi

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
