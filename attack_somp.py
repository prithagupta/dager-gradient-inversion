import argparse
import datetime
import json
import os
import sys
import time

import evaluate
import numpy as np
import pandas as pd
import torch

from args_factory import get_args
from utils.data import TextDataset
from utils.experiment import _repo_root, cleanup_memory
from utils.experiment import setup_experiment_logging
from utils.functional import evaluate_prediction
from utils.functional import print_single_metric_dict
from utils.functional import print_summary_table
from utils.functional import summarize_metrics
from utils.somp_core import ensure_somp_args
from utils.somp_core import reconstruct_with_omp
from utils.somp_models import SOMPModelWrapper


def get_somp_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target_pool", type=int, default=1600)
    parser.add_argument("--k_per_head_max", type=int, default=4096)
    parser.add_argument("--wte_chunk", type=int, default=4096)
    parser.add_argument("--sparse_q", type=float, default=0.25)
    parser.add_argument("--frac_active_heads", type=float, default=0.5)
    parser.add_argument("--lambda_sub", type=float, default=0.8)
    parser.add_argument("--lambda_cons", type=float, default=0.5)
    parser.add_argument("--lambda_sparse", type=float, default=0.5)
    parser.add_argument("--booster_front_positions", type=int, default=5)
    parser.add_argument("--booster_topm", type=int, default=800)
    parser.add_argument("--pos_topk", type=int, default=256)
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--beam_groups", type=int, default=4)
    parser.add_argument("--beam_max_steps", type=int, default=None)
    parser.add_argument("--diversity_lambda", type=float, default=0.35)
    parser.add_argument("--ngram_diversity", type=int, default=2)
    parser.add_argument("--ngram_lambda", type=float, default=0.25)
    parser.add_argument("--beta_glm", type=float, default=0.33)
    parser.add_argument("--cluster_rouge_l", type=float, default=0.7)
    parser.add_argument("--length_bonus_gamma", type=float, default=0.2)
    parser.add_argument("--max_omp_candidates", type=int, default=128)
    parser.add_argument("--somp_add_special_tokens", action="store_true")
    parser.add_argument("--disable_headwise_factorization", action="store_true")

    somp_args, base_argv = parser.parse_known_args()
    args = get_args(base_argv)
    for key, value in vars(somp_args).items():
        setattr(args, key, value)
    args.headwise_factorization = not args.disable_headwise_factorization
    if args.l1_span_thresh == 1e-5:
        args.l1_span_thresh = 0.2
    if args.l2_span_thresh == 1e-3:
        args.l2_span_thresh = 1e-7
    if args.max_len == 1e10:
        args.max_len = 1024
    return ensure_somp_args(args)


args = get_somp_args()
logger, log_path, job_hash = setup_experiment_logging(args, "somp_attack")
logger.info("\n\n\nCommand: %s\n\n\n", " ".join(sys.argv))


def main():
    metric = evaluate.load("rouge", cache_dir=args.cache_dir)
    dataset = TextDataset(
        args.device,
        args.dataset,
        args.split,
        args.n_inputs,
        args.batch_size,
        args.cache_dir,
        use_hf_split=args.use_hf_split,
    )
    model_wrapper = SOMPModelWrapper(args)

    logger.info("\n\nAttacking with SOMP..\n")
    predictions = []
    references = []
    final_sentence_results = []
    final_input_results = []
    input_times = []
    sentence_rows = []
    input_rows = []
    attack_name = "somp"
    results_dir = os.path.join(_repo_root(), "results", attack_name, f"results_{job_hash}")
    os.makedirs(results_dir, exist_ok=True)
    t_start = time.time()

    for input_idx in range(args.start_input, min(args.n_inputs, args.end_input)):
        t_input_start = time.time()
        sample = dataset[input_idx]
        logger.info("Running input #%d of %d.", input_idx, args.n_inputs)
        logger.info("reference:")
        for seq in sample[0]:
            logger.info("========================")
            logger.info(seq)
        logger.info("========================")

        prediction, reference = reconstruct_with_omp(args, sample, metric, model_wrapper)
        predictions.extend(prediction)
        references.extend(reference)

        curr_metrics = []
        logger.info("Done with input #%d of %d.", input_idx, args.n_inputs)
        for sentence_idx, (ref, pred) in enumerate(zip(reference, prediction)):
            logger.info("========================")
            logger.info("Reference: %s", ref)
            logger.info("Prediction: %s", pred)
            metrics = evaluate_prediction(pred, ref, model_wrapper.tokenizer, metric)
            curr_metrics.append(metrics)
            sentence_rows.append(
                {
                    "run_id": job_hash,
                    "attack": attack_name,
                    "model": args.model_path,
                    "dataset": args.dataset,
                    "input_index": input_idx,
                    "sentence_index": sentence_idx,
                    "reference": ref,
                    "prediction": pred,
                    **metrics,
                }
            )

        summary = summarize_metrics(curr_metrics)
        final_sentence_results.extend(curr_metrics)
        logger.info("[Curr input metrics]:")
        logger.info("%s", print_summary_table(summary))

        joined_metrics = evaluate_prediction(
            " ".join(prediction),
            " ".join(reference),
            model_wrapper.tokenizer,
            metric,
        )
        final_input_results.append(joined_metrics)
        logger.info("[Curr joined-input metrics]:")
        logger.info("%s", print_single_metric_dict(joined_metrics))

        input_time_sec = datetime.timedelta(seconds=time.time() - t_input_start).total_seconds()
        input_times.append(input_time_sec)
        logger.info(
            "input #%d time: %s | total time: %s",
            input_idx,
            str(datetime.timedelta(seconds=input_time_sec)).split(".")[0],
            str(datetime.timedelta(seconds=time.time() - t_start)).split(".")[0],
        )
        input_rows.append(
            {
                "run_id": job_hash,
                "attack": attack_name,
                "model": args.model_path,
                "dataset": args.dataset,
                "input_index": input_idx,
                "num_sentences": len(reference),
                "joined_reference": " ".join(reference),
                "joined_prediction": " ".join(prediction),
                "reconstruction_time_sec": input_time_sec,
                **joined_metrics,
            }
        )
        del sample, prediction, reference, curr_metrics, joined_metrics
        cleanup_memory()

    overall = evaluate_prediction(" ".join(predictions), " ".join(references), model_wrapper.tokenizer, metric)
    overall["reconstruction_time_mean"] = float(np.mean(input_times)) if input_times else 0.0
    overall["reconstruction_time_std"] = float(np.std(input_times)) if input_times else 0.0
    sentence_summary = summarize_metrics(final_sentence_results) if final_sentence_results else {}
    input_summary = summarize_metrics(final_input_results) if final_input_results else {}

    logger.info("Overall %s", print_single_metric_dict(overall))
    logger.info("Per Sentence %s", print_summary_table(sentence_summary) if sentence_summary else "{}")
    logger.info("Per Input Results %s", print_summary_table(input_summary) if input_summary else "{}")

    pd.DataFrame(sentence_rows).to_csv(os.path.join(results_dir, "sentence_results.csv"), index=False)
    pd.DataFrame(input_rows).to_csv(os.path.join(results_dir, "input_results.csv"), index=False)
    with open(os.path.join(results_dir, "run_summary.json"), "w") as f:
        json.dump(
            {
                "Overall Results": overall,
                "Per Sentence Results": sentence_summary,
                "Per Input Results": input_summary,
            },
            f,
            indent=2,
        )
    logger.info("Results directory: %s", results_dir)
    logger.info("Done with all.")


if __name__ == "__main__":
    main()
