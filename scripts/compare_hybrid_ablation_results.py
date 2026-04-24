#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


SECTION_SPECS = {
    "Overall Results": {
        "metrics": [
            "rouge1_fm",
            "rouge2_fm",
            "rougeL_fm",
            "exact_match",
            "token_acc",
            "padded_token_acc",
            "experiment_time_mean",
        ],
    },
    "Per Input Results": {
        "metrics": [
            "rouge1_fm_mean",
            "rouge2_fm_mean",
            "rougeL_fm_mean",
            "exact_match_mean",
            "token_acc_mean",
            "padded_token_acc_mean",
            "reconstruction_time_mean",
        ],
    },
    "Per Sentence Results": {
        "metrics": [
            "rouge1_fm_mean",
            "rouge2_fm_mean",
            "rougeL_fm_mean",
            "exact_match_mean",
            "token_acc_mean",
            "padded_token_acc_mean",
            "reconstruction_time_mean",
        ],
    },
}

DISPLAY_METRIC_NAMES = {
    "rouge1_fm": "rouge1_fm",
    "rouge2_fm": "rouge2_fm",
    "rougeL_fm": "rougeL_fm",
    "exact_match": "exact_match",
    "token_acc": "token_acc",
    "padded_token_acc": "padded_token_acc",
    "experiment_time_mean": "experiment_time_mean",
    "rouge1_fm_mean": "rouge1_fm",
    "rouge2_fm_mean": "rouge2_fm",
    "rougeL_fm_mean": "rougeL_fm",
    "exact_match_mean": "exact_match",
    "token_acc_mean": "token_acc",
    "padded_token_acc_mean": "padded_token_acc",
    "reconstruction_time_mean": "reconstruction_time_mean",
}

VARIANT_ORDER = [
    "dager_only",
    "hybrid_full",
    "hybrid_no_dager_init",
    "hybrid_no_lm_prior",
    "hybrid_no_candidate_projection",
]


def parse_manifest_line(line: str) -> dict[str, str]:
    record = {}
    for part in line.strip().split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        record[key] = value
    return record


def load_run_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else math.nan


def safe_std(values: list[float]) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return 0.0
    return pstdev(values)


def infer_variant(method: str, args: dict) -> str:
    if method == "dager":
        return "dager_only"

    init_mode = str(args.get("hybrid_init_mode"))
    use_lm_prior = str(args.get("hybrid_use_lm_prior")).lower()
    projection_mode = str(args.get("hybrid_projection_mode"))

    if init_mode == "dager" and use_lm_prior == "true" and projection_mode == "candidate_final":
        return "hybrid_full"
    if init_mode == "candidate_random" and use_lm_prior == "true" and projection_mode == "candidate_final":
        return "hybrid_no_dager_init"
    if init_mode == "dager" and use_lm_prior == "false":
        return "hybrid_no_lm_prior"
    if init_mode == "dager" and use_lm_prior == "true" and projection_mode == "none":
        return "hybrid_no_candidate_projection"
    return f"unknown_{method}_{init_mode}_{use_lm_prior}_{projection_mode}"


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(manifest_path: Path):
    grouped = defaultdict(lambda: defaultdict(list))

    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = parse_manifest_line(raw_line)
        hash_value = record["hash"]
        method = record["method"]
        attack_dir = "dager_ce" if method == "dager" else "hybrid_ce"
        summary_path = Path("results") / attack_dir / f"results_{hash_value}" / "run_summary.json"
        if not summary_path.exists():
            continue

        summary = load_run_summary(summary_path)
        args = summary.get("Arguments", {})
        variant = infer_variant(method, args)

        key = (
            record["model"],
            record["dataset"],
            record["batch"],
        )
        grouped[key][variant].append(summary)

    long_rows = []
    wide_rows = []

    for (model, dataset, batch), variant_runs in sorted(grouped.items()):
        for section_name, spec in SECTION_SPECS.items():
            wide_row = {
                "model": model,
                "dataset": dataset,
                "batch": batch,
                "section": section_name,
            }

            baseline_metric_means = {}
            baseline_runs = variant_runs.get("dager_only", [])
            baseline_n = len(baseline_runs)

            for metric in spec["metrics"]:
                display_name = DISPLAY_METRIC_NAMES[metric]
                baseline_vals = [
                    float(run.get(section_name, {}).get(metric))
                    for run in baseline_runs
                    if metric in run.get(section_name, {})
                ]
                baseline_metric_means[display_name] = safe_mean(baseline_vals)

            for variant in VARIANT_ORDER:
                runs = variant_runs.get(variant, [])
                wide_row[f"{variant}_n_seeds"] = len(runs)

                for metric in spec["metrics"]:
                    display_name = DISPLAY_METRIC_NAMES[metric]
                    vals = [
                        float(run.get(section_name, {}).get(metric))
                        for run in runs
                        if metric in run.get(section_name, {})
                    ]
                    metric_mean = safe_mean(vals)
                    metric_std = safe_std(vals)
                    wide_row[f"{variant}_{display_name}_seed_mean"] = metric_mean
                    wide_row[f"{variant}_{display_name}_seed_std"] = metric_std

                    if variant != "dager_only":
                        baseline_mean = baseline_metric_means.get(display_name, math.nan)
                        delta = (
                            metric_mean - baseline_mean
                            if not math.isnan(metric_mean) and not math.isnan(baseline_mean)
                            else math.nan
                        )
                        wide_row[f"{variant}_minus_dager_only_{display_name}"] = delta

                    long_rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "batch": batch,
                            "section": section_name,
                            "variant": variant,
                            "metric": display_name,
                            "mean_across_seeds": metric_mean,
                            "std_across_seeds": metric_std,
                            "n_seeds": len(vals),
                        }
                    )

            wide_rows.append(wide_row)

    return long_rows, wide_rows


def main():
    parser = argparse.ArgumentParser(description="Compare hybrid ablation results across variants.")
    parser.add_argument(
        "--manifest",
        default="logs/hybrid_ablation_hashes.txt",
        help="Path to hybrid ablation hash manifest.",
    )
    parser.add_argument(
        "--output-long",
        default="logs/hybrid_ablation_results_long.csv",
        help="Output CSV with one row per variant/metric/section.",
    )
    parser.add_argument(
        "--output-wide",
        default="logs/hybrid_ablation_results_wide.csv",
        help="Output CSV with all variants side-by-side.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Hybrid ablation manifest not found: {manifest_path}")

    long_rows, wide_rows = build_rows(manifest_path)
    write_csv(Path(args.output_long), long_rows)
    write_csv(Path(args.output_wide), wide_rows)

    print(f"Wrote {len(long_rows)} rows to {args.output_long}")
    print(f"Wrote {len(wide_rows)} rows to {args.output_wide}")


if __name__ == "__main__":
    main()
