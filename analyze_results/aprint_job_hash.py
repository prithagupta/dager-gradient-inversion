#!/usr/bin/env python
from pathlib import Path

import argparse
import csv
import json
import shutil
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from args_factory import get_args
from utils.experiment import args_to_dict, get_hash_value_for_args, is_attack_complete

OLD_RESULTS_ROOT = REPO_ROOT / "aresults"
RESULTS_ROOT = REPO_ROOT / "results"
LOGS_ROOT = REPO_ROOT / "logs"
FINAL_RESULTS_DIR = REPO_ROOT / "analyze_results" / "final_results"
SLURM_SCRIPTS_DIR = REPO_ROOT / "slurm_scripts"

RESULT_PARENT_TO_LOG_PREFIX = {
    "dager_ce": "dager_attack",
    "dager_canary_ce": "dager_attack",
    "hybrid_ce": "hybrid_attack",
    "hybrid_canary_ce": "hybrid_attack",
    "iterative_dager_lamp_ce": "hybrid_attack",
    "iterative_dager_lamp_canary_ce": "hybrid_attack",
}


def infer_attack_name(args, attack_kind):
    if attack_kind == "dager":
        attack_prefix = "dager_canary" if args.preprocess_unique_canary_markers else "dager"
    else:
        attack_prefix = "iterative_dager_lamp" if args.iterative_dager_lamp else "hybrid"
        if args.preprocess_unique_canary_markers:
            attack_prefix = f"{attack_prefix}_canary"
    return f"{attack_prefix}_{args.loss}"


def log_prefix_for_attack_kind(attack_kind):
    return "dager_attack" if attack_kind == "dager" else "hybrid_attack"


def default_rank_tol(batch):
    if batch <= 2:
        return "1e-7"
    if batch <= 16:
        return "1e-8"
    return "1e-9"


def default_max_ids(batch):
    if batch >= 128:
        return "32"
    if batch >= 64:
        return "64"
    return None


def dataset_size(dataset):
    return {
        "sst2": 872,
        "cola": 1043,
        "rotten_tomatoes": 1066,
    }.get(dataset, 100000000)


def safe_n_inputs(dataset, batch, requested=50):
    return min(requested, max(1, dataset_size(dataset) // batch))


def append_if_missing(args, flag, value=None):
    if any(a == flag or a.startswith(flag + "=") for a in args):
        return list(args)
    out = list(args)
    out.append(flag)
    if value is not None:
        out.append(value)
    return out


def add_hybrid_defaults(args, lm_mode):
    out = list(args)
    defaults = [
        ("--n_steps", "300"),
        ("--print_every", "50"),
        ("--hybrid_init_mode", "dager"),
        ("--hybrid_projection_mode", "candidate_periodic"),
        ("--hybrid_project_every", "10"),
        ("--iterative_rounds", "3"),
        ("--iterative_steps_per_round", "0"),
        ("--iterative_accept_margin", "1e-6"),
        ("--iterative_stall_patience", "1"),
    ]
    for flag, value in defaults:
        out = append_if_missing(out, flag, value)
    out = append_if_missing(out, "--iterative_dager_lamp")
    out = append_if_missing(out, "--iterative_refresh_candidates")
    if lm_mode == "gpt2":
        out = append_if_missing(out, "--hybrid_use_lm_prior", "true")
        out = append_if_missing(out, "--coeff_perplexity", "0.2")
    else:
        out = append_if_missing(out, "--hybrid_use_lm_prior", "false")
        out = append_if_missing(out, "--coeff_perplexity", "0.0")
    return out


def base_eval_args(dataset, batch, seed=None):
    args = ["--rank_tol", default_rank_tol(batch)]
    if seed is not None:
        args.extend(["--rng_seed", str(seed)])
    max_ids = default_max_ids(batch)
    if max_ids is not None:
        args.extend(["--max_ids", max_ids])
    if dataset in {"sst2", "cola"}:
        args.append("--use_hf_split")
    return args


def build_gpt_cli_args(method, model, dataset, batch, seed=None, canary=False, variant=None):
    model_path = "gpt2" if model == "gpt2" else "openai-community/gpt2-large"
    args = [
        "--dataset", dataset,
        "--split", "val",
        "--n_inputs", "50",
        "--batch_size", str(batch),
        "--l1_filter", "all",
        "--l2_filter", "non-overlap",
        "--model_path", model_path,
        "--device", "auto",
        "--task", "seq_class",
        "--cache_dir", "$HF_HOME/gia_cache",
    ]
    extra = base_eval_args(dataset, batch, seed)
    extra = append_if_missing(extra, "--device_grad", "cpu")
    if method == "hybrid":
        extra = add_hybrid_defaults(extra, "gpt2")
    if variant == "hybrid_no_dager_init":
        extra.extend(["--hybrid_init_mode", "candidate_random"])
    elif variant == "hybrid_no_lm_prior":
        extra.extend(["--hybrid_init_mode", "dager", "--hybrid_use_lm_prior", "false", "--coeff_perplexity", "0.0"])
    elif variant == "hybrid_no_candidate_projection":
        extra.extend(
            ["--hybrid_init_mode", "dager", "--hybrid_use_lm_prior", "true", "--hybrid_projection_mode", "none"])
    if canary:
        extra = append_if_missing(extra, "--preprocess_unique_canary_markers")
        extra = append_if_missing(extra, "--canary_marker_prefix", "qxjkvcanary")
    return args + extra


def build_llama_cli_args(method, dataset, batch, seed, canary=False):
    args = [
        "--dataset", dataset,
        "--split", "val",
        "--n_inputs", "50",
        "--batch_size", str(batch),
        "--l1_filter", "all",
        "--l2_filter", "non-overlap",
        "--model_path", "meta-llama/Meta-Llama-3.1-8B",
        "--device", "auto",
        "--task", "seq_class",
        "--cache_dir", "$HF_HOME/gia_cache",
        "--rank_tol", "1e-8",
        "--l1_span_thresh", "1e-4",
        "--l2_span_thresh", "5e-5",
        "--pad", "left",
        "--precision", "full",
        "--n_incorrect", "8",
        "--attn_implementation", "sdpa",
        "--rng_seed", str(seed),
        "--device_grad", "cuda",
        "--parallel", "4",
    ]
    max_ids = default_max_ids(batch)
    if max_ids is not None:
        args.extend(["--max_ids", max_ids])
    if dataset in {"sst2", "cola"}:
        args.append("--use_hf_split")
    if method == "hybrid":
        args = add_hybrid_defaults(args, "no_lm")
    if canary:
        args = append_if_missing(args, "--preprocess_unique_canary_markers")
        args = append_if_missing(args, "--canary_marker_prefix", "qxjkvcanary")
    return args


def parse_runtime_args(cli_args):
    return get_args(cli_args)


def expected_current_runs():
    rows = []

    def add(script_name, category, method, model, dataset, batch, seed, cli_args, variant=""):
        attack_kind = "dager" if method == "dager" else "hybrid"
        args = parse_runtime_args(cli_args)
        attack_name = infer_attack_name(args, attack_kind)
        hash_value = get_hash_value_for_args(args)
        is_complete, results_dir = is_attack_complete(attack_name, hash_value)
        run_summary_path = Path(results_dir) / "run_summary.json"
        run_summary_status = ""
        if run_summary_path.exists():
            try:
                summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
                run_summary_status = summary.get("status", "complete")
            except Exception:
                run_summary_status = "unreadable"
        result_parent = Path(results_dir).parent.name
        log_path = LOGS_ROOT / f"{log_prefix_for_attack_kind(attack_kind)}_{hash_value}.log"
        rows.append({
            "hash_value": hash_value,
            "attack_kind": attack_kind,
            "attack_name": attack_name,
            "method": method,
            "model": model,
            "dataset": dataset,
            "batch_size": str(batch),
            "rng_seed": "" if seed is None else str(seed),
            "associated_experiment_scripts": script_name,
            "associated_categories": category,
            "associated_variants": variant,
            "done_or_remaining": "done" if is_complete else "remaining",
            "result_parent": result_parent,
            "result_dir": str(results_dir),
            "run_summary_path": str(run_summary_path),
            "run_summary_status": run_summary_status,
            "log_file_path": str(log_path),
            "log_exists": str(log_path.exists()),
            "arguments_json": json.dumps(args_to_dict(args), sort_keys=True, default=str),
        })

    for dataset in ["sst2", "cola", "rotten_tomatoes"]:
        for batch in [16, 32, 64]:
            for model in ["gpt2", "gpt2-large"]:
                for seed in [40, 41, 42]:
                    add("main_benchmark.sh", "main_benchmark", "dager", model, dataset, batch, seed,
                        build_gpt_cli_args("dager", model, dataset, batch, seed))
                    add("main_benchmark.sh", "main_benchmark", "hybrid", model, dataset, batch, seed,
                        build_gpt_cli_args("hybrid", model, dataset, batch, seed))
                    add("main_benchmark_canary.sh", "main_benchmark_canary", "dager", model, dataset, batch, seed,
                        build_gpt_cli_args("dager", model, dataset, batch, seed, canary=True))
                    add("main_benchmark_canary.sh", "main_benchmark_canary", "hybrid", model, dataset, batch, seed,
                        build_gpt_cli_args("hybrid", model, dataset, batch, seed, canary=True))

    for dataset in ["sst2", "cola"]:
        for batch in [1, 2, 4, 8, 16, 32, 64]:
            for model in ["gpt2", "gpt2-large"]:
                add("batch_ablation.sh", "batch_ablation", "dager", model, dataset, batch, 42,
                    build_gpt_cli_args("dager", model, dataset, batch, 42))
                add("batch_ablation.sh", "batch_ablation", "hybrid", model, dataset, batch, 42,
                    build_gpt_cli_args("hybrid", model, dataset, batch, 42))
                add("batch_ablation_canary.sh", "batch_ablation_canary", "dager", model, dataset, batch, 42,
                    build_gpt_cli_args("dager", model, dataset, batch, 42, canary=True))
                add("batch_ablation_canary.sh", "batch_ablation_canary", "hybrid", model, dataset, batch, 42,
                    build_gpt_cli_args("hybrid", model, dataset, batch, 42, canary=True))

    for batch in [8, 16, 32]:
        for seed in [40, 41, 42]:
            add("hybrid_ablation_gpt2.sh", "hybrid_ablation", "dager", "gpt2-large", "sst2", batch, seed,
                build_gpt_cli_args("dager", "gpt2-large", "sst2", batch, seed), variant="dager_only")
            add("hybrid_ablation_gpt2.sh", "hybrid_ablation", "hybrid", "gpt2-large", "sst2", batch, seed,
                build_gpt_cli_args("hybrid", "gpt2-large", "sst2", batch, seed), variant="hybrid_full")
            add("hybrid_ablation_gpt2.sh", "hybrid_ablation", "hybrid", "gpt2-large", "sst2", batch, seed,
                build_gpt_cli_args("hybrid", "gpt2-large", "sst2", batch, seed, variant="hybrid_no_dager_init"),
                variant="hybrid_no_dager_init")
            add("hybrid_ablation_gpt2.sh", "hybrid_ablation", "hybrid", "gpt2-large", "sst2", batch, seed,
                build_gpt_cli_args("hybrid", "gpt2-large", "sst2", batch, seed, variant="hybrid_no_lm_prior"),
                variant="hybrid_no_lm_prior")
            add("hybrid_ablation_gpt2.sh", "hybrid_ablation", "hybrid", "gpt2-large", "sst2", batch, seed,
                build_gpt_cli_args("hybrid", "gpt2-large", "sst2", batch, seed,
                                   variant="hybrid_no_candidate_projection"), variant="hybrid_no_candidate_projection")

            add("hybrid_ablation_gpt2_canary.sh", "hybrid_ablation_canary", "dager", "gpt2-large", "sst2", batch, seed,
                build_gpt_cli_args("dager", "gpt2-large", "sst2", batch, seed, canary=True), variant="dager_only")
            add("hybrid_ablation_gpt2_canary.sh", "hybrid_ablation_canary", "hybrid", "gpt2-large", "sst2", batch, seed,
                build_gpt_cli_args("hybrid", "gpt2-large", "sst2", batch, seed, canary=True), variant="hybrid_full")
            add("hybrid_ablation_gpt2_canary.sh", "hybrid_ablation_canary", "hybrid", "gpt2-large", "sst2", batch, seed,
                build_gpt_cli_args("hybrid", "gpt2-large", "sst2", batch, seed, canary=True,
                                   variant="hybrid_no_dager_init"), variant="hybrid_no_dager_init")
            add("hybrid_ablation_gpt2_canary.sh", "hybrid_ablation_canary", "hybrid", "gpt2-large", "sst2", batch, seed,
                build_gpt_cli_args("hybrid", "gpt2-large", "sst2", batch, seed, canary=True,
                                   variant="hybrid_no_lm_prior"), variant="hybrid_no_lm_prior")
            add("hybrid_ablation_gpt2_canary.sh", "hybrid_ablation_canary", "hybrid", "gpt2-large", "sst2", batch, seed,
                build_gpt_cli_args("hybrid", "gpt2-large", "sst2", batch, seed, canary=True,
                                   variant="hybrid_no_candidate_projection"), variant="hybrid_no_candidate_projection")

    for dataset in ["sst2", "cola", "rotten_tomatoes"]:
        for batch in [8, 16, 32, 64]:
            for seed in [40, 41, 42]:
                add("main_benchmark_llama.sh", "main_benchmark_llama", "dager", "llama_3.1", dataset, batch, seed,
                    build_llama_cli_args("dager", dataset, batch, seed))
                add("main_benchmark_llama.sh", "main_benchmark_llama", "hybrid", "llama_3.1", dataset, batch, seed,
                    build_llama_cli_args("hybrid", dataset, batch, seed))
                add("main_benchmark_llama.sh", "main_benchmark_llama_canary", "dager", "llama_3.1", dataset, batch,
                    seed,
                    build_llama_cli_args("dager", dataset, batch, seed, canary=True))
                add("main_benchmark_llama.sh", "main_benchmark_llama_canary", "hybrid", "llama_3.1", dataset, batch,
                    seed,
                    build_llama_cli_args("hybrid", dataset, batch, seed, canary=True))

    return rows


def aggregate_expected_rows(rows):
    grouped = {}
    for row in rows:
        key = row["hash_value"]
        if key not in grouped:
            grouped[key] = dict(row)
            grouped[key]["associated_experiment_scripts"] = [row["associated_experiment_scripts"]]
            grouped[key]["associated_categories"] = [row["associated_categories"]]
            grouped[key]["associated_variants"] = [row["associated_variants"]] if row["associated_variants"] else []
        else:
            grouped[key]["associated_experiment_scripts"].append(row["associated_experiment_scripts"])
            grouped[key]["associated_categories"].append(row["associated_categories"])
            if row["associated_variants"]:
                grouped[key]["associated_variants"].append(row["associated_variants"])
            if row["done_or_remaining"] == "done":
                grouped[key]["done_or_remaining"] = "done"
    out = []
    for row in grouped.values():
        row["associated_experiment_scripts"] = ";".join(sorted(set(row["associated_experiment_scripts"])))
        row["associated_categories"] = ";".join(sorted(set(row["associated_categories"])))
        row["associated_variants"] = ";".join(sorted(set(row["associated_variants"])))
        out.append(row)
    out.sort(key=lambda r: (r["done_or_remaining"], r["associated_experiment_scripts"], r["method"], r["model"],
                            r["dataset"], r["batch_size"], r["rng_seed"]))
    return out


def inventory_command(argv):
    parser = argparse.ArgumentParser(
        description="Collect current Slurm experiment hashes and identify obsolete result folders.")
    parser.add_argument("--current-output", default=str(FINAL_RESULTS_DIR / "current_slurm_runs.csv"))
    parser.add_argument("--remove-output", default=str(FINAL_RESULTS_DIR / "result_folders_to_remove.csv"))
    args = parser.parse_args(argv)

    FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    current_rows = aggregate_expected_rows(expected_current_runs())
    current_hashes = {row["hash_value"] for row in current_rows}

    current_path = Path(args.current_output)
    remove_path = Path(args.remove_output)

    current_fields = [
        "hash_value",
        "attack_kind",
        "attack_name",
        "method",
        "model",
        "dataset",
        "batch_size",
        "rng_seed",
        "associated_experiment_scripts",
        "associated_categories",
        "associated_variants",
        "done_or_remaining",
        "result_parent",
        "result_dir",
        "run_summary_path",
        "run_summary_status",
        "log_file_path",
        "log_exists",
        "arguments_json",
    ]
    with current_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=current_fields)
        writer.writeheader()
        writer.writerows(current_rows)

    remove_rows = []
    for summary_path in RESULTS_ROOT.glob("**/run_summary.json"):
        summary = _load_summary(summary_path)
        if not summary:
            continue
        canonical_hash = _hash_from_summary_args(summary)
        status = summary.get("status", "complete")
        if canonical_hash in current_hashes:
            continue
        if status == "incomplete":
            continue
        results_dir = summary_path.parent
        parent_dir = results_dir.parent.name
        log_prefix = RESULT_PARENT_TO_LOG_PREFIX.get(parent_dir, "")
        log_path = LOGS_ROOT / f"{log_prefix}_{canonical_hash}.log" if log_prefix else Path("")
        remove_rows.append({
            "hash_value": canonical_hash,
            "result_parent": parent_dir,
            "result_dir": str(results_dir),
            "run_summary_path": str(summary_path),
            "run_summary_status": status,
            "log_file_path": str(log_path) if log_prefix else "",
            "log_exists": str(log_path.exists()) if log_prefix else "",
            "reason": "not_in_current_slurm_experiments",
            "arguments_json": json.dumps(args_to_dict(SimpleNamespace(**(summary.get("Arguments") or {}))),
                                         sort_keys=True, default=str),
        })

    remove_fields = [
        "hash_value",
        "result_parent",
        "result_dir",
        "run_summary_path",
        "run_summary_status",
        "log_file_path",
        "log_exists",
        "reason",
        "arguments_json",
    ]
    with remove_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=remove_fields)
        writer.writeheader()
        writer.writerows(sorted(remove_rows, key=lambda r: (r["result_parent"], r["hash_value"])))

    print(json.dumps({
        "current_csv": str(current_path),
        "current_rows": len(current_rows),
        "remove_csv": str(remove_path),
        "remove_rows": len(remove_rows),
    }, indent=2, sort_keys=True))


def status_command(argv):
    parser = argparse.ArgumentParser(description="Print canonical attack hash and completion status.")
    parser.add_argument("--attack-kind", choices=["dager", "hybrid"], required=True)
    ns, remaining = parser.parse_known_args(argv)

    args = get_args(remaining)
    attack_name = infer_attack_name(args, ns.attack_kind)
    job_hash = get_hash_value_for_args(args)
    is_complete, results_dir = is_attack_complete(attack_name, job_hash)

    payload = {
        "attack_kind": ns.attack_kind,
        "attack_name": attack_name,
        "hash_value": job_hash,
        "is_complete": bool(is_complete),
        "results_dir": results_dir,
        "arguments_json": json.dumps(args_to_dict(args), sort_keys=True, default=str),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _scrub_run_id_column(csv_path, apply_changes):
    if not csv_path.exists():
        return False
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "run_id" not in fieldnames:
            return False
        rows = list(reader)

    new_fieldnames = [name for name in fieldnames if name != "run_id"]
    if apply_changes:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            for row in rows:
                row.pop("run_id", None)
                writer.writerow(row)
    return True


def _load_summary(summary_path):
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _hash_from_summary_args(summary):
    args = dict(summary.get("Arguments") or {})
    args.pop("status", None)
    return get_hash_value_for_args(SimpleNamespace(**args))


def _rename_path(src, dst, apply_changes):
    if not src.exists():
        return "missing_source"
    if src == dst:
        return "already_canonical"
    if dst.exists():
        return "target_exists"
    if apply_changes:
        src.rename(dst)
    return "renamed"


def _mtime_for_path(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def _delete_tree_and_logs(results_dir, log_prefix, hash_value, apply_changes):
    result_status = "missing_source"
    log_status = "missing_source"
    if results_dir.exists():
        result_status = "deleted"
        if apply_changes:
            shutil.rmtree(results_dir)
    log_path = LOGS_ROOT / f"{log_prefix}_{hash_value}.log" if log_prefix else Path("")
    lock_path = LOGS_ROOT / f"{log_prefix}_{hash_value}.log.lock" if log_prefix else Path("")
    if log_prefix:
        if log_path.exists() or lock_path.exists():
            log_status = "deleted"
            if apply_changes:
                try:
                    if log_path.exists():
                        log_path.unlink()
                except OSError:
                    pass
                try:
                    if lock_path.exists():
                        lock_path.unlink()
                except OSError:
                    pass
    return result_status, log_status


def repair_results_command(argv):
    parser = argparse.ArgumentParser(
        description="Recompute canonical hashes from run_summary.json, scrub run_id, and optionally rename completed result/log paths.")
    parser.add_argument("--apply", action="store_true", help="Apply renames and CSV cleanup. Default is dry-run.")
    parser.add_argument("--output", default=str(FINAL_RESULTS_DIR / "hash_repair_report.csv"))
    args = parser.parse_args(argv)

    FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    rows = []

    for summary_path in RESULTS_ROOT.glob("**/run_summary.json"):
        summary = _load_summary(summary_path)
        if not summary:
            continue

        results_dir = summary_path.parent
        parent_dir = results_dir.parent.name
        log_prefix = RESULT_PARENT_TO_LOG_PREFIX.get(parent_dir, "")
        status = summary.get("status", "complete")
        current_hash = results_dir.name.replace("results_", "")
        canonical_hash = _hash_from_summary_args(summary)
        sentence_csv = results_dir / "sentence_results.csv"
        input_csv = results_dir / "input_results.csv"

        scrubbed_sentence = False
        scrubbed_input = False
        if status != "incomplete":
            scrubbed_sentence = _scrub_run_id_column(sentence_csv, args.apply)
            scrubbed_input = _scrub_run_id_column(input_csv, args.apply)

        result_rename_status = "skipped_incomplete"
        log_rename_status = "skipped_incomplete"
        target_results_dir = results_dir.parent / f"results_{canonical_hash}"

        if status != "incomplete":
            if target_results_dir.exists() and results_dir != target_results_dir:
                src_mtime = _mtime_for_path(summary_path)
                target_summary = target_results_dir / "run_summary.json"
                dst_mtime = _mtime_for_path(target_summary)
                if src_mtime > dst_mtime:
                    deleted_result_status, deleted_log_status = _delete_tree_and_logs(
                        target_results_dir, log_prefix, canonical_hash, args.apply
                    )
                    result_rename_status = f"replaced_target_{deleted_result_status}"
                    current_log = LOGS_ROOT / f"{log_prefix}_{current_hash}.log"
                    target_log = LOGS_ROOT / f"{log_prefix}_{canonical_hash}.log"
                    log_rename_status = f"replaced_target_{deleted_log_status}"
                    if apply_changes and current_log.exists():
                        current_log.rename(target_log)
                    current_lock = LOGS_ROOT / f"{log_prefix}_{current_hash}.log.lock"
                    target_lock = LOGS_ROOT / f"{log_prefix}_{canonical_hash}.log.lock"
                    if apply_changes and current_lock.exists():
                        current_lock.rename(target_lock)
                    if apply_changes:
                        results_dir.rename(target_results_dir)
                else:
                    deleted_result_status, deleted_log_status = _delete_tree_and_logs(
                        results_dir, log_prefix, current_hash, args.apply
                    )
                    result_rename_status = f"deleted_older_source_{deleted_result_status}"
                    log_rename_status = f"deleted_older_source_{deleted_log_status}"
            else:
                result_rename_status = _rename_path(results_dir, target_results_dir, args.apply)
                if log_prefix:
                    current_log = LOGS_ROOT / f"{log_prefix}_{current_hash}.log"
                    target_log = LOGS_ROOT / f"{log_prefix}_{canonical_hash}.log"
                    log_rename_status = _rename_path(current_log, target_log, args.apply)
                    current_lock = LOGS_ROOT / f"{log_prefix}_{current_hash}.log.lock"
                    target_lock = LOGS_ROOT / f"{log_prefix}_{canonical_hash}.log.lock"
                    _rename_path(current_lock, target_lock, args.apply)
        else:
            target_log = ""

        rows.append({
            "result_parent": parent_dir,
            "current_hash": current_hash,
            "canonical_hash": canonical_hash,
            "status": status,
            "results_dir": str(results_dir),
            "target_results_dir": str(target_results_dir),
            "result_rename_status": result_rename_status,
            "log_prefix": log_prefix,
            "log_rename_status": log_rename_status,
            "scrubbed_sentence_results_run_id": scrubbed_sentence,
            "scrubbed_input_results_run_id": scrubbed_input,
            "arguments_json": json.dumps(args_to_dict(SimpleNamespace(**(summary.get("Arguments") or {}))),
                                         sort_keys=True, default=str),
        })

    fieldnames = [
        "result_parent",
        "current_hash",
        "canonical_hash",
        "status",
        "results_dir",
        "target_results_dir",
        "result_rename_status",
        "log_prefix",
        "log_rename_status",
        "scrubbed_sentence_results_run_id",
        "scrubbed_input_results_run_id",
        "arguments_json",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "apply": args.apply,
        "rows_written": len(rows),
        "report_path": str(output_path),
    }, indent=2, sort_keys=True))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        status_command(["--attack-kind", "dager"])
        return

    command = argv[0]
    if command == "status":
        status_command(argv[1:])
    elif command == "inventory":
        inventory_command(argv[1:])
    elif command == "repair-results":
        repair_results_command(argv[1:])
    else:
        # Backwards-compatible simple mode:
        # python aprint_job_hash.py --attack-kind dager --dataset ...
        status_command(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
