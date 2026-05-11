#!/bin/bash

set -euo pipefail

# Conda activation hooks are not always nounset-clean. In particular, MKL's
# activate.d scripts may read MKL_INTERFACE_LAYER before it exists. Keep the
# rest of our scripts strict, but temporarily relax `set -u` around Conda.
_THEMIS_RESTORE_NOUNSET=0
case "$-" in
  *u*)
    _THEMIS_RESTORE_NOUNSET=1
    set +u
    ;;
esac

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

if [ "$_THEMIS_RESTORE_NOUNSET" = "1" ]; then
  set -u
fi
unset _THEMIS_RESTORE_NOUNSET

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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export DAGER_TORCH_NUM_THREADS="${DAGER_TORCH_NUM_THREADS:-$OMP_NUM_THREADS}"
export DAGER_TORCH_INTEROP_THREADS="${DAGER_TORCH_INTEROP_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *expandable_segments* ]]; then
  echo "[THEMIS] Removing expandable_segments from inherited PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
  unset PYTORCH_CUDA_ALLOC_CONF
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

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
echo "[THEMIS] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "[THEMIS] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[THEMIS] MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "[THEMIS] OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "[THEMIS] NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
echo "[THEMIS] DAGER_TORCH_NUM_THREADS=$DAGER_TORCH_NUM_THREADS"
echo "[THEMIS] DAGER_TORCH_INTEROP_THREADS=$DAGER_TORCH_INTEROP_THREADS"
echo "[THEMIS] TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
