#!/bin/bash
#SBATCH --job-name=dager
#SBATCH -N 1
#SBATCH --account=haberi2x_0000
#SBATCH --partition=fat_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/%u/logs/%x_%j.out
#SBATCH --output=/lustre/%u/logs/%x_%j.out
#SBATCH --error=/lustre/%u/logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

#hybrid_ablation_gpt2.sh, main_benchmark.sh
# Project & script
PROJECT_DIR="/lustre/$USER/projects/dager-gradient-inversion"
RUN_SCRIPT="${RUN_SCRIPT:-main_benchmark.sh}"
SCRIPT_PATH="$PROJECT_DIR/slurm_scripts/$RUN_SCRIPT"
#export MKL_INTERFACE_LAYER=LP64
# Optional extra args to pass through to the experiment script.
SCRIPT_ARGS="${SCRIPT_ARGS:-}"

# Env
source ~/.bashrc || true
export MKL_INTERFACE_LAYER=LP64
set +u
conda activate dager-cuda
set -u

# HuggingFace cache: keep everything under $HF_HOME/gia_exp_cache
export HF_HOME="${HF_HOME:-${DATA_FOLDER}/.cache/huggingface}"
export CACHE_DIR="${HF_HOME}/gia_exp_cache"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CACHE_DIR}/.hf_hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_DIR}/.hf_datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_DIR}/.transformers}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${CACHE_DIR}/.hf_modules}"
export HF_EVALUATE_CACHE="${HF_EVALUATE_CACHE:-${CACHE_DIR}/.hf_evaluate}"
export HF_METRICS_CACHE="${HF_METRICS_CACHE:-${CACHE_DIR}/.hf_metrics}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_EVALUATE_OFFLINE="${HF_EVALUATE_OFFLINE:-1}"
export HF_UPDATE_DOWNLOAD_COUNTS="${HF_UPDATE_DOWNLOAD_COUNTS:-0}"
export PYTHONUNBUFFERED=1

# Threads
export OMP_NUM_THREADS="${SLURM_CPUS_PER_GPU:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_GPU:-4}"

mkdir -p "/lustre/$USER/logs"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_MODULES_CACHE" "$HF_EVALUATE_CACHE" "$HF_METRICS_CACHE"

echo "[INFO] Host: $(hostname)"
echo "[INFO] PWD: $(pwd)"
echo "[INFO] Project dir: $PROJECT_DIR"
echo "[INFO] Script path: $SCRIPT_PATH"
echo "[INFO] Script args: $SCRIPT_ARGS"
echo "[INFO] HF_HOME: $HF_HOME"
echo "[INFO] CACHE_DIR: $CACHE_DIR"
echo "[INFO] HF_HUB_CACHE: $HF_HUB_CACHE"
echo "[INFO] HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "[INFO] TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "[INFO] HF_MODULES_CACHE: $HF_MODULES_CACHE"
echo "[INFO] HF_EVALUATE_CACHE: $HF_EVALUATE_CACHE"
echo "[INFO] HF_METRICS_CACHE: $HF_METRICS_CACHE"
echo "[INFO] HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo "[INFO] HF_DATASETS_OFFLINE: $HF_DATASETS_OFFLINE"
echo "[INFO] TRANSFORMERS_OFFLINE: $TRANSFORMERS_OFFLINE"
echo "[INFO] HF_EVALUATE_OFFLINE: $HF_EVALUATE_OFFLINE"
ls -la "$PROJECT_DIR"
ls -la "$PROJECT_DIR/slurm_scripts"

cd "$PROJECT_DIR"

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "[ERROR] Script not found: $SCRIPT_PATH" >&2
  exit 1
fi

if [ -n "$SCRIPT_ARGS" ]; then
  bash "$SCRIPT_PATH" $SCRIPT_ARGS
else
  bash "$SCRIPT_PATH"
fi
