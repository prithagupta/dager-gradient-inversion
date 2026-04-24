#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common_attack_args.sh"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/logs}"
mkdir -p "$OUTPUT_DIR"

MAIN_HASH_FILE="$OUTPUT_DIR/main_benchmark_hashes.txt"
MAIN_COMPARISON_FILE="$OUTPUT_DIR/main_benchmark_comparison_hashes.txt"
BATCH_HASH_FILE="$OUTPUT_DIR/batch_ablation_hashes.txt"
HYBRID_HASH_FILE="$OUTPUT_DIR/hybrid_ablation_hashes.txt"

: > "$MAIN_HASH_FILE"
: > "$MAIN_COMPARISON_FILE"
: > "$BATCH_HASH_FILE"
: > "$HYBRID_HASH_FILE"

CACHE_DIR="${CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/gia_exp_cache}"

main_count=0
main_comparison_count=0
batch_count=0
hybrid_count=0

apply_attack_defaults() {
  local batch="$1"
  shift || true

  local run_args=( "$@" )

  if [ "${#run_args[@]}" -gt 0 ]; then
    set_default_max_ids_arg "$batch" "${run_args[@]}"
  else
    set_default_max_ids_arg "$batch"
  fi
  run_args=( "${ATTACK_EXTRA_ARGS[@]}" )

  if [ "${#run_args[@]}" -gt 0 ]; then
    set_default_rank_tol_arg "$batch" "${run_args[@]}"
  else
    set_default_rank_tol_arg "$batch"
  fi
  run_args=( "${ATTACK_EXTRA_ARGS[@]}" )

  if [ "${#run_args[@]}" -gt 0 ]; then
    set_default_rng_seed_arg 42 "${run_args[@]}"
  else
    set_default_rng_seed_arg 42
  fi
}

write_hash_record() {
  local outfile="$1"
  local category="$2"
  local method="$3"
  local model="$4"
  local dataset="$5"
  local batch="$6"
  local seed="$7"
  local hash="$8"
  shift 8
  local args=( "$@" )

  local args_string=""
  printf -v args_string '%q ' "${args[@]}"
  printf 'category=%s method=%s model=%s dataset=%s batch=%s seed=%s hash=%s args=%s\n' \
    "$category" "$method" "$model" "$dataset" "$batch" "$seed" "$hash" "$args_string" >> "$outfile"
}

hash_dager_args() {
  python "$REPO_ROOT/print_job_hash.py" "$@"
}

write_main_comparison_record() {
  local outfile="$1"
  local model="$2"
  local dataset="$3"
  local batch="$4"
  local seed="$5"
  local dager_hash="$6"
  local hybrid_hash="$7"

  printf 'category=main_benchmark_compare model=%s dataset=%s batch=%s seed=%s dager_hash=%s hybrid_hash=%s dager_results=%s hybrid_results=%s dager_summary=%s hybrid_summary=%s\n' \
    "$model" \
    "$dataset" \
    "$batch" \
    "$seed" \
    "$dager_hash" \
    "$hybrid_hash" \
    "$REPO_ROOT/results/dager_ce/results_${dager_hash}" \
    "$REPO_ROOT/results/hybrid_ce/results_${hybrid_hash}" \
    "$REPO_ROOT/results/dager_ce/results_${dager_hash}/run_summary.json" \
    "$REPO_ROOT/results/hybrid_ce/results_${hybrid_hash}/run_summary.json" >> "$outfile"
}

collect_main_benchmark_hashes() {
  local datasets=( "sst2" "cola" "rotten_tomatoes" )
  local models=( "gpt2" "gpt2-large" )
  local methods=( "dager" "hybrid" )
  local seeds=( 40 41 42 )
  local batch="8"

  local dataset model seed model_path dager_hash hybrid_hash
  local run_args dager_args hybrid_args
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      for seed in "${seeds[@]}"; do
        run_args=( --rng_seed "$seed" )
        if [ "$dataset" = "sst2" ] || [ "$dataset" = "cola" ]; then
          run_args+=( --use_hf_split )
        fi

        apply_attack_defaults "$batch" "${run_args[@]}"
        run_args=( "${ATTACK_EXTRA_ARGS[@]}" )

        if [ "$model" = "gpt2" ]; then
          model_path="gpt2"
        else
          model_path="openai-community/gpt2-large"
        fi

        dager_args=(
          --dataset "$dataset"
          --split val
          --n_inputs 100
          --batch_size "$batch"
          --l1_filter all
          --l2_filter non-overlap
          --model_path "$model_path"
          --device auto
          --task seq_class
          --cache_dir "$CACHE_DIR"
          "${run_args[@]}"
        )
        dager_hash="$(hash_dager_args "${dager_args[@]}")"
        write_hash_record "$MAIN_HASH_FILE" "main_benchmark" "dager" "$model" "$dataset" "$batch" "$seed" "$dager_hash" "${dager_args[@]}"
        main_count=$((main_count + 1))

        hybrid_args=(
          --dataset "$dataset"
          --split val
          --n_inputs 100
          --batch_size "$batch"
          --l1_filter all
          --l2_filter non-overlap
          --model_path "$model_path"
          --device auto
          --task seq_class
          --cache_dir "$CACHE_DIR"
        )
        if [ "$model" = "gpt2" ]; then
          hybrid_args+=( --n_steps 10 )
        else
          hybrid_args+=( --n_steps 50 )
        fi
        hybrid_args+=( "${run_args[@]}" )
        hybrid_hash="$(hash_dager_args "${hybrid_args[@]}")"
        write_hash_record "$MAIN_HASH_FILE" "main_benchmark" "hybrid" "$model" "$dataset" "$batch" "$seed" "$hybrid_hash" "${hybrid_args[@]}"
        main_count=$((main_count + 1))

        write_main_comparison_record "$MAIN_COMPARISON_FILE" "$model" "$dataset" "$batch" "$seed" "$dager_hash" "$hybrid_hash"
        main_comparison_count=$((main_comparison_count + 1))
      done
    done
  done
}

collect_batch_ablation_hashes() {
  local datasets=( "sst2" "cola" )
  local models=( "gpt2" "gpt2-large" )
  local methods=( "dager" "hybrid" )
  local batches=( 128 64 32 16 4 2 1 )
  local seed="42"

  local dataset model method batch model_path hash max_hf_inputs chosen_n_inputs
  local run_args final_args
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      for batch in "${batches[@]}"; do
        for method in "${methods[@]}"; do
          run_args=( --rng_seed "$seed" --use_hf_split )

          if [ "$batch" -gt 64 ]; then
            run_args+=( --max_ids 96 --rank_tol 1e-8 --l1_span_thresh 5e-5 --l2_span_thresh 2e-3 --distinct_thresh 0.6 )
          fi

          max_hf_inputs=1
          if [ "$dataset" = "sst2" ]; then
            max_hf_inputs=$(( 872 / batch ))
          elif [ "$dataset" = "cola" ]; then
            max_hf_inputs=$(( 1043 / batch ))
          fi
          if [ "$max_hf_inputs" -lt 1 ]; then
            max_hf_inputs=1
          fi
          chosen_n_inputs=100
          if [ "$max_hf_inputs" -lt "$chosen_n_inputs" ]; then
            chosen_n_inputs="$max_hf_inputs"
          fi
          run_args+=( --n_inputs "$chosen_n_inputs" )

          apply_attack_defaults "$batch" "${run_args[@]}"
          run_args=( "${ATTACK_EXTRA_ARGS[@]}" )

          if [ "$model" = "gpt2" ]; then
            model_path="gpt2"
          else
            model_path="openai-community/gpt2-large"
          fi

          if [ "$method" = "dager" ]; then
            final_args=(
              --dataset "$dataset"
              --split val
              --n_inputs 100
              --batch_size "$batch"
              --l1_filter all
              --l2_filter non-overlap
              --model_path "$model_path"
              --device auto
              --task seq_class
              --cache_dir "$CACHE_DIR"
              "${run_args[@]}"
            )
          else
            final_args=(
              --dataset "$dataset"
              --split val
              --n_inputs 100
              --batch_size "$batch"
              --l1_filter all
              --l2_filter non-overlap
              --model_path "$model_path"
              --device auto
              --task seq_class
              --cache_dir "$CACHE_DIR"
            )
            if [ "$model" = "gpt2" ]; then
              final_args+=( --n_steps 10 )
            else
              final_args+=( --n_steps 50 )
            fi
            final_args+=( "${run_args[@]}" )
          fi

          hash="$(hash_dager_args "${final_args[@]}")"
          write_hash_record "$BATCH_HASH_FILE" "batch_ablation" "$method" "$model" "$dataset" "$batch" "$seed" "$hash" "${final_args[@]}"
          batch_count=$((batch_count + 1))
        done
      done
    done
  done
}

collect_hybrid_ablation_hashes() {
  local seeds=( 40 41 42 )
  local dataset="sst2"
  local batch="8"
  local model="gpt2-large"
  local model_path="openai-community/gpt2-large"

  local variant_names=( "dager_only" "hybrid_full" "hybrid_no_dager_init" "hybrid_no_lm_prior" "hybrid_no_candidate_projection" )
  local seed variant hash
  local run_args final_args

  for seed in "${seeds[@]}"; do
    for variant in "${variant_names[@]}"; do
      run_args=( --rng_seed "$seed" --use_hf_split )
      case "$variant" in
        dager_only)
          ;;
        hybrid_full)
          run_args+=( --hybrid_init_mode dager --hybrid_use_lm_prior true --hybrid_projection_mode candidate_final )
          ;;
        hybrid_no_dager_init)
          run_args+=( --hybrid_init_mode candidate_random --hybrid_use_lm_prior true --hybrid_projection_mode candidate_final )
          ;;
        hybrid_no_lm_prior)
          run_args+=( --hybrid_init_mode dager --hybrid_use_lm_prior false )
          ;;
        hybrid_no_candidate_projection)
          run_args+=( --hybrid_init_mode dager --hybrid_use_lm_prior true --hybrid_projection_mode none )
          ;;
      esac

      apply_attack_defaults "$batch" "${run_args[@]}"
      run_args=( "${ATTACK_EXTRA_ARGS[@]}" )

      if [ "$variant" = "dager_only" ]; then
        final_args=(
          --dataset "$dataset"
          --split val
          --n_inputs 100
          --batch_size "$batch"
          --l1_filter all
          --l2_filter non-overlap
          --model_path "$model_path"
          --device auto
          --task seq_class
          --cache_dir "$CACHE_DIR"
          "${run_args[@]}"
        )
        hash="$(hash_dager_args "${final_args[@]}")"
        write_hash_record "$HYBRID_HASH_FILE" "hybrid_ablation" "dager" "$model" "$dataset" "$batch" "$seed" "$hash" "${final_args[@]}"
      else
        final_args=(
          --dataset "$dataset"
          --split val
          --n_inputs 100
          --batch_size "$batch"
          --l1_filter all
          --l2_filter non-overlap
          --model_path "$model_path"
          --device auto
          --task seq_class
          --cache_dir "$CACHE_DIR"
          --n_steps 50
          "${run_args[@]}"
        )
        hash="$(hash_dager_args "${final_args[@]}")"
        write_hash_record "$HYBRID_HASH_FILE" "hybrid_ablation" "hybrid" "$model" "$dataset" "$batch" "$seed" "$hash" "${final_args[@]}"
      fi
      hybrid_count=$((hybrid_count + 1))
    done
  done
}

collect_main_benchmark_hashes
collect_batch_ablation_hashes
collect_hybrid_ablation_hashes

echo "Wrote $main_count hashes to $MAIN_HASH_FILE"
echo "Wrote $main_comparison_count paired comparison rows to $MAIN_COMPARISON_FILE"
echo "Wrote $batch_count hashes to $BATCH_HASH_FILE"
echo "Wrote $hybrid_count hashes to $HYBRID_HASH_FILE"
