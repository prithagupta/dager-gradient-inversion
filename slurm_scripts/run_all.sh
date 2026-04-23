#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_RUNNER="$SCRIPT_DIR/run_dager_gpu.slurm"

if [ ! -f "$SLURM_RUNNER" ]; then
  echo "[ERROR] Slurm runner not found: $SLURM_RUNNER" >&2
  exit 1
fi

run_scripts=(
  "main_benchmark.sh"
  "batch_ablation.sh"
  "hybrid_ablation_gpt2.sh"
)

for run_script in "${run_scripts[@]}"; do
  echo "[INFO] Submitting $run_script via $(basename "$SLURM_RUNNER")"
  sbatch --export=ALL,RUN_SCRIPT="$run_script" "$SLURM_RUNNER"
done
