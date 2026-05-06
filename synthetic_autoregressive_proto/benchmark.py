import argparse
import atexit
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthetic_autoregressive_proto.batch_reranker import rerank_batch, rerank_candidate_batches
from synthetic_autoregressive_proto.config import PrototypeConfig
from synthetic_autoregressive_proto.constrained_decoder import decode_slot_candidates
from synthetic_autoregressive_proto.evidence_extractor import SyntheticEvidenceExtractor
from synthetic_autoregressive_proto.evidence_io import load_evidence_file
from synthetic_autoregressive_proto.slot_inference import infer_slots
from synthetic_autoregressive_proto.token_graph import build_token_graph
from utils.data import TextDataset
from utils.experiment import load_rouge_metric, setup_random_seed, write_attack_artifacts
from utils.functional import (
    _safe_aggregated_metrics,
    evaluate_prediction,
    extract_canary_metric_means,
    maybe_add_canary_audit_metrics,
    print_summary_table,
    summarize_metrics,
)
from utils.models import _resolve_local_model_path

logger = logging.getLogger("synthetic_autoregressive_proto")
_CONSOLE_STREAM = None
_PUNCT_ONLY_RE = re.compile(r"^[^\w]+$", re.UNICODE)


def _is_offline_mode():
    return any(
        str(os.environ.get(flag, "")).lower() in {"1", "true", "yes"}
        for flag in ["HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_EVALUATE_OFFLINE"]
    )


def _load_tokenizer(model_path: str, cache_dir: str | None, pad: str) -> AutoTokenizer:
    resolved = _resolve_local_model_path(model_path, cache_dir)
    cache_dir_accessible = cache_dir is not None and os.path.isdir(cache_dir)
    if cache_dir is not None and not cache_dir_accessible:
        warnings.warn(
            f"cache_dir '{cache_dir}' is not accessible from this machine. "
            f"Falling back to the local/default Hugging Face cache.",
            RuntimeWarning,
        )

    load_attempts = []
    kwargs = {"use_fast": True}
    if cache_dir_accessible:
        kwargs["cache_dir"] = cache_dir
    if _is_offline_mode():
        kwargs["local_files_only"] = True
    load_attempts.append((resolved, kwargs))

    fallback_kwargs = {"use_fast": True}
    if _is_offline_mode():
        fallback_kwargs["local_files_only"] = True
    if (resolved != model_path) or kwargs != fallback_kwargs:
        load_attempts.append((model_path, fallback_kwargs))

    last_exc = None
    for candidate_path, candidate_kwargs in load_attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate_path, **candidate_kwargs)
            break
        except Exception as exc:
            last_exc = exc
    else:
        raise RuntimeError(
            f"Could not load tokenizer for model_path='{model_path}' with cache_dir='{cache_dir}'. "
            f"Resolved path was '{resolved}'. If you are on Themis, use the shared synthetic cache view such as "
            f"'--cache_dir /media/data/shared/hf_cache/gia_cache'. If you are on your Mac, either omit "
            f"'--cache_dir' or point it at a local model cache."
        ) from last_exc
    tokenizer.model_max_length = 512
    if pad == "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        elif tokenizer.unk_token is not None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token})
    return tokenizer


def _args_to_dict(args):
    return {k: getattr(args, k) for k in sorted(vars(args))}


def _build_output_dir(args) -> str:
    safe_model = args.model_path.replace("/", "__")
    safe_dataset = args.dataset.replace("/", "__")
    variant = "canary" if args.preprocess_unique_canary_markers else "baseline"
    evidence_suffix = ""
    if args.evidence_file:
        evidence_suffix = "_" + Path(args.evidence_file).stem.replace("/", "__")
    run_name = (
        f"{safe_dataset}_{safe_model}_b{args.batch_size}_n{args.n_inputs}_"
        f"seed{args.rng_seed}_slots{args.n_slots or args.batch_size}_{variant}{evidence_suffix}"
    )
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        run_name,
    )


def _setup_logging(output_dir: str) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "benchmark.log")
    console_path = os.path.join(output_dir, "console_output.log")

    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    logger.propagate = True
    return log_path, console_path


def _suppress_noisy_logs() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    for noisy_name in [
        "absl",
        "absl.logging",
        "evaluate",
        "evaluate.loading",
        "datasets",
        "transformers",
        "transformers.tokenization_utils_base",
        "urllib3.connectionpool",
    ]:
        noisy_logger = logging.getLogger(noisy_name)
        noisy_logger.setLevel(logging.ERROR)
        noisy_logger.propagate = False
        noisy_logger.handlers.clear()
        noisy_logger.disabled = True
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
        absl_logging.set_stderrthreshold("fatal")
    except Exception:
        pass


def _redirect_console(console_path: str) -> None:
    global _CONSOLE_STREAM
    _CONSOLE_STREAM = open(console_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = _CONSOLE_STREAM
    sys.stderr = _CONSOLE_STREAM
    atexit.register(_CONSOLE_STREAM.close)


def _token_set_score(pred: str, ref: str) -> float:
    pred_toks = set(pred.split())
    ref_toks = set(ref.split())
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    inter = len(pred_toks & ref_toks)
    union = len(pred_toks | ref_toks)
    return inter / max(union, 1)


def _align_predictions(predictions, references):
    if not predictions:
        return [""] * len(references), []

    score_matrix = []
    for pred_idx, pred in enumerate(predictions):
        row = []
        for ref_idx, ref in enumerate(references):
            row.append(_token_set_score(pred, ref))
        score_matrix.append(row)

    scores = torch.tensor(score_matrix, dtype=torch.float64)
    cost = (1.0 - scores).cpu().numpy()
    pred_ids, ref_ids = linear_sum_assignment(cost)

    aligned_predictions = [""] * len(references)
    for pred_idx, ref_idx in zip(pred_ids.tolist(), ref_ids.tolist()):
        aligned_predictions[ref_idx] = predictions[pred_idx]

    return aligned_predictions


def _tokenize_for_evidence(text: str, tokenizer, mode: str) -> list[str]:
    if mode == "model":
        pieces = tokenizer.tokenize(text)
        return pieces if pieces else [text]
    pieces = text.split()
    return pieces if pieces else [text]


def _normalize_evidence_tokens(tokens: list[str], args) -> list[str]:
    normalized = []
    canary_re = re.compile(rf"^{re.escape(args.canary_marker_prefix)}\d{{6}}$")
    for tok in tokens:
        if args.evidence_strip_canary_markers and canary_re.match(tok):
            continue
        if args.evidence_drop_punctuation and _PUNCT_ONLY_RE.match(tok):
            continue
        normalized.append(tok)
    return normalized if normalized else tokens


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Synthetic autoregressive prototype benchmark")
    parser.add_argument("--dataset", required=True,
                        choices=["cola", "sst2", "rte", "rotten_tomatoes", "stanfordnlp/imdb", "glnmario/ECHR"])
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_inputs", type=int, default=10)
    parser.add_argument("--start_input", type=int, default=0)
    parser.add_argument("--end_input", type=int, default=100000)
    parser.add_argument("--model_path", type=str, default="gpt2")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--rng_seed", type=int, default=42)
    parser.add_argument("--pad", type=str, default="right", choices=["right", "left"])
    parser.add_argument("--use_hf_split", action="store_true")
    parser.add_argument("--preprocess_numbered_markers", action="store_true")
    parser.add_argument("--preprocess_boundary_markers", action="store_true")
    parser.add_argument("--preprocess_unique_canary_markers", action="store_true")
    parser.add_argument("--canary_marker_prefix", type=str, default="qxjkvcanary")
    parser.add_argument("--n_slots", type=int, default=None)
    parser.add_argument("--top_k_per_position", type=int, default=4)
    parser.add_argument("--graph_edge_threshold", type=float, default=0.20)
    parser.add_argument("--assignment_temperature", type=float, default=0.8)
    parser.add_argument("--slot_stride", type=int, default=5)
    parser.add_argument("--slot_candidate_width", type=int, default=3)
    parser.add_argument("--slot_source_bonus", type=float, default=0.8)
    parser.add_argument("--decoder_beam_size", type=int, default=3)
    parser.add_argument("--max_sequence_length", type=int, default=32)
    parser.add_argument("--decoder_graph_weight", type=float, default=0.75)
    parser.add_argument("--decoder_repetition_penalty", type=float, default=0.25)
    parser.add_argument("--decoder_common_token_penalty", type=float, default=0.5)
    parser.add_argument("--decoder_source_bonus", type=float, default=0.9)
    parser.add_argument("--evidence_frequency_penalty", type=float, default=0.35)
    parser.add_argument("--rerank_weight_support", type=float, default=1.0)
    parser.add_argument("--rerank_weight_coherence", type=float, default=0.5)
    parser.add_argument("--rerank_weight_source_coverage", type=float, default=0.75)
    parser.add_argument("--rerank_duplicate_penalty", type=float, default=0.8)
    parser.add_argument("--rerank_batch_beam_size", type=int, default=8)
    parser.add_argument("--evidence_strip_canary_markers", action="store_true")
    parser.add_argument("--evidence_drop_punctuation", action="store_true")
    parser.add_argument("--evidence_tokenization", type=str, default="whitespace", choices=["whitespace", "model"])
    parser.add_argument(
        "--evidence_file",
        type=str,
        default=None,
        help="Optional JSON file with precomputed per-input token evidence and optional graph edges.",
    )
    parser.add_argument("--debug_top_positions", type=int, default=8)
    parser.add_argument("--debug_top_candidates", type=int, default=3)
    parser.add_argument("--debug_top_slots", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser


def main(argv=None):
    warnings.filterwarnings("ignore")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.preprocess_unique_canary_markers and not args.evidence_strip_canary_markers:
        args.evidence_strip_canary_markers = True
    if not args.evidence_drop_punctuation:
        args.evidence_drop_punctuation = True
    setup_random_seed(args.rng_seed)

    requested_slots = args.n_slots or args.batch_size
    slot_clamp_message = None
    if requested_slots != args.batch_size:
        slot_clamp_message = (
            f"Clamping n_slots from {requested_slots} to batch_size={args.batch_size} "
            "for synthetic slot reconstruction."
        )
    args.n_slots = args.batch_size
    args.top_k_per_position = max(args.top_k_per_position, min(args.batch_size, 16))
    args.slot_candidate_width = max(2, min(args.slot_candidate_width, args.top_k_per_position))
    output_dir = args.output_dir or _build_output_dir(args)
    log_path, console_path = _setup_logging(output_dir)
    _redirect_console(console_path)
    _suppress_noisy_logs()
    logger.info("Starting synthetic_autoregressive_proto benchmark")
    logger.info("Arguments: %s", json.dumps(_args_to_dict(args), sort_keys=True))
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Output dir: %s", output_dir)
    logger.info("Log path: %s", log_path)
    logger.info("Console path: %s", console_path)
    if slot_clamp_message:
        logger.warning(slot_clamp_message)

    config = PrototypeConfig(
        n_slots=args.n_slots,
        top_k_per_position=args.top_k_per_position,
        graph_edge_threshold=args.graph_edge_threshold,
        assignment_temperature=args.assignment_temperature,
        slot_stride=args.slot_stride,
        slot_candidate_width=args.slot_candidate_width,
        slot_source_bonus=args.slot_source_bonus,
        decoder_beam_size=args.decoder_beam_size,
        max_sequence_length=args.max_sequence_length,
        decoder_graph_weight=args.decoder_graph_weight,
        decoder_repetition_penalty=args.decoder_repetition_penalty,
        decoder_common_token_penalty=args.decoder_common_token_penalty,
        decoder_source_bonus=args.decoder_source_bonus,
        evidence_frequency_penalty=args.evidence_frequency_penalty,
        rerank_weight_support=args.rerank_weight_support,
        rerank_weight_coherence=args.rerank_weight_coherence,
        rerank_weight_source_coverage=args.rerank_weight_source_coverage,
        rerank_duplicate_penalty=args.rerank_duplicate_penalty,
        rerank_batch_beam_size=args.rerank_batch_beam_size,
        random_seed=args.rng_seed,
    )

    tokenizer = _load_tokenizer(args.model_path, args.cache_dir, args.pad)
    rouge_metric = load_rouge_metric(cache_dir=args.cache_dir)
    evidence_records = load_evidence_file(args.evidence_file) if args.evidence_file else {}
    dataset = TextDataset(
        torch.device("cpu"),
        args.dataset,
        args.split,
        args.n_inputs,
        args.batch_size,
        cache_dir=args.cache_dir,
        use_hf_split=args.use_hf_split,
        preprocess_numbered_markers=args.preprocess_numbered_markers,
        preprocess_boundary_markers=args.preprocess_boundary_markers,
        preprocess_unique_canary_markers=args.preprocess_unique_canary_markers,
        canary_marker_prefix=args.canary_marker_prefix,
    )

    extractor = SyntheticEvidenceExtractor(rng_seed=args.rng_seed)
    sentence_rows = []
    input_rows = []
    per_sentence_metrics = []
    per_input_metrics = []
    all_predictions = []
    all_references = []
    graph_nodes = []
    graph_edges = []

    end_index = min(len(dataset.seqs), args.end_input)

    for input_index in range(args.start_input, end_index):
        input_start = time.time()
        logger.info("Processing input %s/%s", input_index + 1, len(dataset.seqs))
        references = list(dataset.seqs[input_index])
        stage_start = time.time()
        evidence_record = evidence_records.get(input_index)
        if evidence_record is not None:
            if evidence_record.references is not None and len(evidence_record.references) != len(references):
                logger.warning(
                    "Input %s evidence_file references length=%s differs from dataset batch length=%s; using dataset references for metrics.",
                    input_index,
                    len(evidence_record.references),
                    len(references),
                )
            evidences = evidence_record.evidences
            logger.info(
                "Input %s loaded evidence_file positions=%s references_in_file=%s",
                input_index,
                len(evidences),
                evidence_record.references is not None,
            )
        else:
            evidence_sequences = [
                _normalize_evidence_tokens(
                    _tokenize_for_evidence(ref, tokenizer, args.evidence_tokenization),
                    args,
                )
                for ref in references
            ]
            logger.info(
                "Input %s evidence_tokenization=%s mean_seq_len=%.2f",
                input_index,
                args.evidence_tokenization,
                sum(len(seq) for seq in evidence_sequences) / max(len(evidence_sequences), 1),
            )
            evidences = extractor.extract(
                evidence_sequences,
                top_k_per_position=config.top_k_per_position,
                frequency_penalty=config.evidence_frequency_penalty,
            )
        logger.info(
            "Input %s evidence extraction finished in %.2fs",
            input_index,
            time.time() - stage_start,
        )
        for ev in evidences[:args.debug_top_positions]:
            candidate_summary = ", ".join(
                f"{cand.token}:{cand.support:.2f}"
                for cand in ev.candidates[:args.debug_top_candidates]
            )
            logger.info("Evidence pos %s -> %s", ev.position, candidate_summary)
        stage_start = time.time()
        graph = evidence_record.graph if evidence_record is not None and evidence_record.graph is not None else build_token_graph(evidences, edge_threshold=config.graph_edge_threshold)
        slots = infer_slots(
            evidences,
            n_slots=config.n_slots,
            slot_stride=config.slot_stride,
            candidate_width=config.slot_candidate_width,
            source_bonus=config.slot_source_bonus,
        )
        candidate_groups = [
            decode_slot_candidates(
                slot,
                graph,
                max_sequence_length=config.max_sequence_length,
                beam_size=config.decoder_beam_size,
                graph_weight=config.decoder_graph_weight,
                repetition_penalty=config.decoder_repetition_penalty,
                common_token_penalty=config.decoder_common_token_penalty,
                source_bonus=config.decoder_source_bonus,
            )
            for slot in slots
        ]
        decoded = [group[0] for group in candidate_groups if group]
        reranked = rerank_candidate_batches(candidate_groups, evidences, config)
        if not reranked:
            reranked = rerank_batch(decoded, evidences, config)
        raw_predictions = [seq.text for seq in reranked if seq.text]
        logger.info(
            "Input %s graph/slot/decode finished in %.2fs",
            input_index,
            time.time() - stage_start,
        )

        stage_start = time.time()
        predictions = _align_predictions(raw_predictions, references)
        logger.info(
            "Input %s alignment finished in %.2fs",
            input_index,
            time.time() - stage_start,
        )
        logger.info(
            "Input %s graph_nodes=%s graph_edges=%s decoded=%s",
            input_index,
            len(graph),
            sum(len(v) for v in graph.values()),
            len(raw_predictions),
        )
        for seq in reranked[:args.debug_top_slots]:
            logger.info(
                "Slot %s support=%.3f coherence=%.3f text=%s",
                seq.slot_id,
                seq.support_score,
                seq.coherence_score,
                seq.text,
            )

        graph_nodes.append(float(len(graph)))
        graph_edges.append(float(sum(len(v) for v in graph.values())))

        input_sentence_metrics = []
        for sentence_index, (pred, ref) in enumerate(zip(predictions, references)):
            metrics = evaluate_prediction(pred, ref, tokenizer, rouge_metric)
            metrics = maybe_add_canary_audit_metrics(
                metrics,
                pred,
                ref,
                tokenizer,
                rouge_metric,
                enabled=args.preprocess_unique_canary_markers,
                canary_prefix=args.canary_marker_prefix,
            )
            input_sentence_metrics.append(metrics)
            per_sentence_metrics.append(metrics)
            all_predictions.append(pred)
            all_references.append(ref)

            row = {
                "attack": "synthetic_autoregressive_proto",
                "model": args.model_path,
                "dataset": args.dataset,
                "input_index": input_index,
                "sentence_index": sentence_index,
                "reference": ref,
                "prediction": pred,
            }
            row.update(metrics)
            sentence_rows.append(row)
            logger.info("========================")
            logger.info("Reference: %s", ref)
            logger.info("Prediction: %s", pred)
            logger.info(
                "Input %s sentence %s rouge2=%.4f non_canary_rouge2=%.4f exact=%s",
                input_index,
                sentence_index,
                float(metrics.get("rouge2_fm", 0.0)),
                float(metrics.get("non_canary_rouge2_fm", metrics.get("rouge2_fm", 0.0))),
                int(metrics.get("exact_match", 0)),
            )

        joined_pred = " ".join(predictions)
        joined_ref = " ".join(references)
        input_metric = _safe_aggregated_metrics(
            predictions,
            references,
            tokenizer,
            rouge_metric,
            input_sentence_metrics,
            scope=f"synthetic_proto_input_{input_index}",
        )
        input_metric = maybe_add_canary_audit_metrics(
            input_metric,
            joined_pred,
            joined_ref,
            tokenizer,
            rouge_metric,
            enabled=args.preprocess_unique_canary_markers,
            canary_prefix=args.canary_marker_prefix,
        )
        input_metric["graph_nodes"] = float(len(graph))
        input_metric["graph_edges"] = float(sum(len(v) for v in graph.values()))
        input_metric["num_sentences"] = float(len(references))
        per_input_metrics.append(input_metric)
        logger.info("========================")

        curr_summary = summarize_metrics(input_sentence_metrics)
        logger.info("[Curr input metrics]:")
        logger.info("%s", print_summary_table(curr_summary))
        if args.preprocess_unique_canary_markers:
            non_canary_summary = {
                key: value
                for key, value in curr_summary.items()
                if "non_canary_" in key
            }
            if non_canary_summary:
                logger.info("[Curr input non-canary metrics]:")
                logger.info("%s", print_summary_table(non_canary_summary))
        logger.info("[Aggregate metrics]:")
        logger.info("%s", json.dumps(input_metric, sort_keys=True, indent=2))

        input_row = {
            "attack": "synthetic_autoregressive_proto",
            "model": args.model_path,
            "dataset": args.dataset,
            "input_index": input_index,
            "num_sentences": len(references),
            "joined_reference": joined_ref,
            "joined_prediction": joined_pred,
            "reconstruction_time_sec": float(time.time() - input_start),
        }
        input_row.update(input_metric)
        input_rows.append(input_row)
        logger.info(
            "Input %s elapsed_time_sec=%.2f",
            input_index,
            input_row["reconstruction_time_sec"],
        )

        partial_overall = _safe_aggregated_metrics(
            all_predictions,
            all_references,
            tokenizer,
            rouge_metric,
            per_sentence_metrics,
            scope=f"synthetic_proto_partial_overall_{input_index}",
        )
        partial_summary = {
            "Arguments": _args_to_dict(args),
            "Attack": "synthetic_autoregressive_proto",
            "Overall Results": partial_overall,
            "Per Input Results": summarize_metrics(per_input_metrics) if per_input_metrics else {},
            "Per Sentence Results": summarize_metrics(per_sentence_metrics) if per_sentence_metrics else {},
        }
        partial_canary_means = extract_canary_metric_means(partial_summary["Per Input Results"])
        if partial_canary_means:
            partial_summary["Canary Audit Results"] = partial_canary_means
        write_attack_artifacts(output_dir, sentence_rows, input_rows, summary_results=partial_summary, status="incomplete")

    overall_results = _safe_aggregated_metrics(
        all_predictions,
        all_references,
        tokenizer,
        rouge_metric,
        per_sentence_metrics,
        scope="synthetic_proto_overall",
    )
    overall_results["graph_nodes_mean"] = float(sum(graph_nodes) / max(len(graph_nodes), 1))
    overall_results["graph_edges_mean"] = float(sum(graph_edges) / max(len(graph_edges), 1))

    per_input_summary = summarize_metrics(per_input_metrics) if per_input_metrics else {}
    per_sentence_summary = summarize_metrics(per_sentence_metrics) if per_sentence_metrics else {}

    summary = {
        "Arguments": _args_to_dict(args),
        "Attack": "synthetic_autoregressive_proto",
        "Overall Results": overall_results,
        "Per Input Results": per_input_summary,
        "Per Sentence Results": per_sentence_summary,
    }
    canary_means = extract_canary_metric_means(per_input_summary)
    if canary_means:
        summary["Canary Audit Results"] = canary_means

    write_attack_artifacts(output_dir, sentence_rows, input_rows, summary_results=summary, status="complete")
    logger.info("Finished benchmark. overall_rouge2_fm=%.4f", float(overall_results.get("rouge2_fm", 0.0)))
    logger.info("Artifacts written to %s", output_dir)

    print(json.dumps({
        "output_dir": output_dir,
        "log_path": log_path,
        "console_path": console_path,
        "n_inputs_effective": len(dataset.seqs),
        "batch_size": args.batch_size,
        "dataset": args.dataset,
        "model_path": args.model_path,
        "overall_rouge2_fm": overall_results.get("rouge2_fm"),
    }, indent=2))


if __name__ == "__main__":
    main()
