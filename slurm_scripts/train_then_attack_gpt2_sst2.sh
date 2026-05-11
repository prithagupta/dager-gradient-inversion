#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common_attack_args.sh"
cd "$REPO_ROOT"

IFS=' ' read -r -a noises <<< "${TRAIN_NOISES:-clean 0.1 0.5 1.0}"
IFS=' ' read -r -a attack_batches <<< "${BATCHES:-8 16 32}"
IFS=' ' read -r -a attack_seeds <<< "${SEEDS:-42}"
IFS=' ' read -r -a models <<< "${MODELS:-gpt2}"
IFS=' ' read -r -a datasets <<< "${TRAIN_DATASETS:-sst2}"
ATTACKS="${ATTACKS:-dager hybrid dager_canary hybrid_canary}"

TRAIN_SEED="${TRAIN_SEED:-42}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_LR="${TRAIN_LR:-5e-5}"
TRAIN_MAX_GRAD_NORM="${TRAIN_MAX_GRAD_NORM:-1.0}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
DEVICE_GRAD="${DEVICE_GRAD:-cpu}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

echo "[CONFIG] script=train_then_attack_gpt2_sst2.sh"
echo "[CONFIG] train_noises=${noises[*]}"
echo "[CONFIG] attack_batches=${attack_batches[*]}"
echo "[CONFIG] attack_seeds=${attack_seeds[*]}"
echo "[CONFIG] models=${models[*]}"
echo "[CONFIG] datasets=${datasets[*]}"
echo "[CONFIG] attacks=$ATTACKS"

latest_model_for_output_dir() {
  local output_dir="$1"
  if [ -f "$output_dir/latest_path.txt" ]; then
    local recorded
    recorded="$(cat "$output_dir/latest_path.txt")"
    if [ -d "$recorded" ]; then
      printf '%s\n' "$recorded"
      return 0
    fi
  fi
  find "$output_dir" -mindepth 1 -maxdepth 1 -type d -name '*_full_steps*' -print 2>/dev/null |
    sort -V |
    tail -n 1
}

for model in "${models[@]}"; do
  model_tag="${model//\//__}"
  for dataset in "${datasets[@]}"; do
    for noise in "${noises[@]}"; do
      if [ "$noise" = "clean" ] || [ "$noise" = "none" ]; then
        noise_tag="clean"
        family="fine_tune"
      else
        noise_tag="noise_${noise//./p}"
        family="dp"
      fi
      output_dir="$DAGER_CACHE_DIR/$family/${model_tag}_${dataset}_${noise_tag}_seed${TRAIN_SEED}"
      finetuned_path="$(latest_model_for_output_dir "$output_dir" || true)"

      if [ "$FORCE_TRAIN" = "1" ] || [ -z "$finetuned_path" ] || [ ! -d "$finetuned_path" ]; then
        echo "[TRAIN-THEN-ATTACK] Training model=$model dataset=$dataset noise=$noise"
        TRAIN_MODEL_PATH="$model" \
        TRAIN_DATASET="$dataset" \
        TRAIN_NOISE="$noise" \
        TRAIN_SEED="$TRAIN_SEED" \
        TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
        TRAIN_EPOCHS="$TRAIN_EPOCHS" \
        TRAIN_LR="$TRAIN_LR" \
        TRAIN_MAX_GRAD_NORM="$TRAIN_MAX_GRAD_NORM" \
        TRAIN_DEVICE="$TRAIN_DEVICE" \
        FINETUNED_OUTPUT_DIR="$output_dir" \
          bash "$REPO_ROOT/scripts/train_gpt2_sst2_model.sh"
        finetuned_path="$(latest_model_for_output_dir "$output_dir")"
      else
        echo "[TRAIN-THEN-ATTACK] Reusing finetuned model: $finetuned_path"
      fi

      if [ -z "$finetuned_path" ] || [ ! -d "$finetuned_path" ]; then
        echo "[ERROR] No finetuned model available for model=$model dataset=$dataset noise=$noise under $output_dir" >&2
        exit 1
      fi

      FINETUNED_PATH="$finetuned_path" \
      FINETUNED_BASE_MODEL="$model" \
      ATTACK_DATASET="$dataset" \
      BATCHES="${attack_batches[*]}" \
      SEEDS="${attack_seeds[*]}" \
      ATTACKS="$ATTACKS" \
      DEVICE_GRAD="$DEVICE_GRAD" \
        bash "$REPO_ROOT/scripts/run_saved_gpt2_sst2_attacks.sh"
    done
  done
done
