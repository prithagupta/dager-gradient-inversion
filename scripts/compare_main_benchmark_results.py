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
        "time_metric": "experiment_time_mean",
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
        "time_metric": "reconstruction_time_mean",
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
        "time_metric": "reconstruction_time_mean",
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


def parse_manifest_line(line: str) -> dict[str, str]:
    parts = line.strip().split()
    record = {}
    for part in parts:
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


def build_rows(comparison_manifest: Path):
    grouped = defaultdict(list)

    for raw_line in comparison_manifest.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = parse_manifest_line(raw_line)
        dager_summary = Path(record["dager_summary"])
        hybrid_summary = Path(record["hybrid_summary"])
        if not dager_summary.exists() or not hybrid_summary.exists():
            continue

        grouped[(record["model"], record["dataset"])].append(
            {
                "seed": record["seed"],
                "dager": load_run_summary(dager_summary),
                "hybrid": load_run_summary(hybrid_summary),
            }
        )

    long_rows = []
    wide_rows = []

    for (model, dataset), runs in sorted(grouped.items()):
        for section_name, spec in SECTION_SPECS.items():
            dager_metric_values = defaultdict(list)
            hybrid_metric_values = defaultdict(list)

            for run in runs:
                dager_section = run["dager"].get(section_name, {})
                hybrid_section = run["hybrid"].get(section_name, {})
                for metric in spec["metrics"]:
                    if metric in dager_section:
                        dager_metric_values[metric].append(float(dager_section[metric]))
                    if metric in hybrid_section:
                        hybrid_metric_values[metric].append(float(hybrid_section[metric]))

            wide_row = {
                "model": model,
                "dataset": dataset,
                "section": section_name,
                "n_seeds": len(runs),
            }

            for metric in spec["metrics"]:
                display_name = DISPLAY_METRIC_NAMES[metric]
                d_vals = dager_metric_values[metric]
                h_vals = hybrid_metric_values[metric]

                d_mean = safe_mean(d_vals)
                d_std = safe_std(d_vals)
                h_mean = safe_mean(h_vals)
                h_std = safe_std(h_vals)
                delta = h_mean - d_mean if not math.isnan(d_mean) and not math.isnan(h_mean) else math.nan

                long_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "section": section_name,
                        "method": "dager",
                        "metric": display_name,
                        "mean_across_seeds": d_mean,
                        "std_across_seeds": d_std,
                        "n_seeds": len(d_vals),
                    }
                )
                long_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "section": section_name,
                        "method": "hybrid",
                        "metric": display_name,
                        "mean_across_seeds": h_mean,
                        "std_across_seeds": h_std,
                        "n_seeds": len(h_vals),
                    }
                )

                wide_row[f"dager_{display_name}_seed_mean"] = d_mean
                wide_row[f"dager_{display_name}_seed_std"] = d_std
                wide_row[f"hybrid_{display_name}_seed_mean"] = h_mean
                wide_row[f"hybrid_{display_name}_seed_std"] = h_std
                wide_row[f"hybrid_minus_dager_{display_name}"] = delta

            wide_rows.append(wide_row)

    return long_rows, wide_rows


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


def main():
    parser = argparse.ArgumentParser(description="Compare DAGER and Hybrid main benchmark results.")
    parser.add_argument(
        "--comparison-manifest",
        default="logs/main_benchmark_comparison_hashes.txt",
        help="Path to main benchmark paired hash manifest.",
    )
    parser.add_argument(
        "--output-long",
        default="logs/main_benchmark_results_long.csv",
        help="Output CSV with one row per method/metric/section.",
    )
    parser.add_argument(
        "--output-wide",
        default="logs/main_benchmark_results_wide.csv",
        help="Output CSV with DAGER and Hybrid side-by-side.",
    )
    args = parser.parse_args()

    comparison_manifest = Path(args.comparison_manifest)
    if not comparison_manifest.exists():
        raise FileNotFoundError(f"Comparison manifest not found: {comparison_manifest}")

    long_rows, wide_rows = build_rows(comparison_manifest)
    write_csv(Path(args.output_long), long_rows)
    write_csv(Path(args.output_wide), wide_rows)

    print(f"Wrote {len(long_rows)} rows to {args.output_long}")
    print(f"Wrote {len(wide_rows)} rows to {args.output_wide}")


if __name__ == "__main__":
    main()
