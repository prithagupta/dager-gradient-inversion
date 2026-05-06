#!/bin/bash

set -euo pipefail

if [ -f "/media/data/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "/media/data/$USER/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  if [ "${CONDA_DEFAULT_ENV:-}" != "dager-cuda" ]; then
    conda activate dager-cuda
  fi
else
  echo "[ERROR] conda is not available; cannot activate the dager-cuda environment." >&2
  exit 1
fi


# On Themis we want a stable writable cache root. Prefer an explicit override,
# otherwise prefer the shared HF cache already used on the cluster.
export HF_HOME="/media/data/shared/hf_cache"
# Force a consistent Themis cache layout instead of inheriting potentially
# inconsistent interactive-shell values.
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_EVALUATE_CACHE="$HF_HOME/evaluate"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export HF_MODULES_CACHE="$HF_HOME/modules"
export DAGER_CACHE_DIR="${DAGER_CACHE_DIR:-$HF_HOME/gia_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_EVALUATE_CACHE" "$HF_METRICS_CACHE" "$HF_MODULES_CACHE" "$DAGER_CACHE_DIR"

echo "[THEMIS] Using conda env: dager-cuda"
echo "[THEMIS] HF_HOME=$HF_HOME"
echo "[THEMIS] DAGER_CACHE_DIR=$DAGER_CACHE_DIR"
echo "[THEMIS] HF_HUB_CACHE=$HF_HUB_CACHE"
echo "[THEMIS] TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "[THEMIS] HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "[THEMIS] HF_EVALUATE_CACHE=$HF_EVALUATE_CACHE"
echo "[THEMIS] HF_METRICS_CACHE=$HF_METRICS_CACHE"
echo "[THEMIS] HF_MODULES_CACHE=$HF_MODULES_CACHE"
