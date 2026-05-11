#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_RUNNERS=(
  #"$SCRIPT_DIR/run_dager_gpu.slurm"
  #"$SCRIPT_DIR/run_dager_fat_gpu.slurm"
  "$SCRIPT_DIR/run_dager_cpu.slurm"
)

run_exports=(
  #"RUN_SCRIPT=main_benchmark.sh"
  #"RUN_SCRIPT=main_benchmark_canary.sh"
  "RUN_SCRIPT=batch_ablation.sh,DATASETS=rotten_tomatoes,BATCHES=1,2,4,8,16,32,64"
  "RUN_SCRIPT=batch_ablation_canary.sh,DATASETS=rotten_tomatoes,BATCHES=1,2,4,8,16,32,64"
  #"RUN_SCRIPT=main_benchmark_llama.sh"
  #"RUN_SCRIPT=main_benchmark_llama.sh,USE_SYNTHETIC_CANARY=1"
  #"RUN_SCRIPT=ardager_main_benchmark.sh,USE_SYNTHETIC_CANARY=1"
  #"RUN_SCRIPT=ardager_main_benchmark.sh,USE_SYNTHETIC_CANARY=0"
  #"RUN_SCRIPT=ardager_batch_ablation.sh,USE_SYNTHETIC_CANARY=1"
  #"RUN_SCRIPT=ardager_batch_ablation.sh,USE_SYNTHETIC_CANARY=0"
)

for slurm_runner in "${SLURM_RUNNERS[@]}"; do
  if [ ! -f "$slurm_runner" ]; then
    echo "[ERROR] Slurm runner not found: $slurm_runner" >&2
    exit 1
  fi
done

for slurm_runner in "${SLURM_RUNNERS[@]}"; do
  for run_export in "${run_exports[@]}"; do
    echo "[INFO] Submitting ${run_export} via $(basename "$slurm_runner")"
    sbatch --export=ALL,"$run_export" "$slurm_runner"
  done
done
