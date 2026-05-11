#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export HF_HOME="${HF_HOME:-$REPO_ROOT/models_cache}"
source "$SCRIPT_DIR/common_attack_args.sh"
cd "$REPO_ROOT"

TRAIN_NOISE="${TRAIN_NOISE:-clean}"
TRAIN_SEED="${TRAIN_SEED:-42}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_LR="${TRAIN_LR:-5e-5}"
TRAIN_MAX_GRAD_NORM="${TRAIN_MAX_GRAD_NORM:-1.0}"
TRAIN_SAVE_EVERY="${TRAIN_SAVE_EVERY:-1000000}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
TRAIN_MODEL_PATH="${TRAIN_MODEL_PATH:-gpt2}"
TRAIN_DATASET="${TRAIN_DATASET:-sst2}"

if [ "$TRAIN_NOISE" = "clean" ] || [ "$TRAIN_NOISE" = "none" ]; then
  noise_tag="clean"
  noise_args=()
else
  noise_tag="noise_${TRAIN_NOISE//./p}"
  noise_args=( --noise "$TRAIN_NOISE" )
fi

model_tag="${TRAIN_MODEL_PATH//\//__}"
output_dir="${FINETUNED_OUTPUT_DIR:-$DAGER_CACHE_DIR/finetuned/${model_tag}_${TRAIN_DATASET}_${noise_tag}_seed${TRAIN_SEED}}"

echo "[TRAIN] dataset=$TRAIN_DATASET model=$TRAIN_MODEL_PATH noise=$TRAIN_NOISE seed=$TRAIN_SEED"
echo "[TRAIN] output_dir=$output_dir"

python train.py \
  --dataset "$TRAIN_DATASET" \
  --model_path "$TRAIN_MODEL_PATH" \
  --train_method full \
  --batch_size "$TRAIN_BATCH_SIZE" \
  --num_epochs "$TRAIN_EPOCHS" \
  --learning_rate "$TRAIN_LR" \
  --max_grad_norm "$TRAIN_MAX_GRAD_NORM" \
  --rng_seed "$TRAIN_SEED" \
  --save_every "$TRAIN_SAVE_EVERY" \
  --cache_dir "$DAGER_CACHE_DIR" \
  --output_dir "$output_dir" \
  --device "$TRAIN_DEVICE" \
  "${noise_args[@]}"

latest_path="$(
  find "$output_dir" -mindepth 1 -maxdepth 1 -type d -name "${TRAIN_DATASET}_*_full_steps*" -print |
  sort -V |
  tail -n 1
)"

if [ -z "$latest_path" ]; then
  echo "[ERROR] No trained model directory found under $output_dir" >&2
  exit 1
fi

printf '%s\n' "$latest_path" > "$output_dir/latest_path.txt"
echo "[TRAIN] FINETUNED_PATH=$latest_path"
