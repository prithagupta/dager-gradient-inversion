#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_themis.sh"
LOG_DIR="$SCRIPT_DIR/logs"
GPU_MAIN_CANARY="${GPU_MAIN_CANARY:-1}"
GPU_MAIN="${GPU_MAIN:-2}"
GPU_LLAMA="${GPU_LLAMA:-1}"
GPU_LLAMA_CANARY="${GPU_LLAMA_CANARY:-2}"
GPU_BATCH_CANARY_A="${GPU_BATCH_CANARY_A:-0}"
GPU_BATCH_CANARY_B="${GPU_BATCH_CANARY_B:-0}"
PIDS=()

mkdir -p "$LOG_DIR"

run_lane() {
  local gpu="$1"
  shift
  echo "[THEMIS] Starting lane on CUDA_VISIBLE_DEVICES=${gpu}: $*"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    lane_pids=()
    local lane_idx=0
    while [ "$#" -gt 0 ]; do
      local run_script="$1"
      local log_name=""
      local log_path=""
      shift
      lane_idx=$((lane_idx + 1))
      log_name="${run_script//[^A-Za-z0-9._-]/_}"
      log_path="$LOG_DIR/gpu${gpu}_${lane_idx}_${log_name}.log"
      case "$run_script" in
        main_benchmark_llama_canary)
          echo "[THEMIS][GPU ${gpu}] Running main_benchmark_llama.sh with USE_SYNTHETIC_CANARY=1"
          echo "[THEMIS][GPU ${gpu}] Log: $log_path"
          (
            RUN_SCRIPT="main_benchmark_llama.sh" USE_SYNTHETIC_CANARY=1 bash "$RUNNER"
          ) >"$log_path" 2>&1 &
          lane_pids+=( "$!" )
          ;;
        *)
          echo "[THEMIS][GPU ${gpu}] Running ${run_script}"
          echo "[THEMIS][GPU ${gpu}] Log: $log_path"
          (
            RUN_SCRIPT="$run_script" bash "$RUNNER"
          ) >"$log_path" 2>&1 &
          lane_pids+=( "$!" )
          ;;
      esac
    done
    lane_status=0
    for lane_pid in "${lane_pids[@]}"; do
      if ! wait "$lane_pid"; then
        lane_status=1
      fi
    done
    exit "$lane_status"
  ) &
  PIDS+=( "$!" )
}

#run_lane "$GPU_MAIN" "main_benchmark.sh"
#run_lane "$GPU_MAIN_CANARY" "main_benchmark_canary.sh"
run_lane "$GPU_LLAMA" "main_benchmark_llama.sh"
run_lane "$GPU_LLAMA_CANARY" "main_benchmark_llama_canary"
run_lane "$GPU_BATCH_CANARY_A" "batch_ablation.sh"
run_lane "$GPU_BATCH_CANARY_B" "batch_ablation_canary.sh"

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
