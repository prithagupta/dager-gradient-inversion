#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


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

TIME_METRICS = {"experiment_time_mean", "reconstruction_time_mean"}
QUALITY_METRICS = {
    "rouge1_fm",
    "rouge2_fm",
    "rougeL_fm",
    "exact_match",
    "token_acc",
    "padded_token_acc",
}


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


def classify_metric(metric: str, dager_value: float, hybrid_value: float, tolerance: float) -> tuple[str, float]:
    if math.isnan(dager_value) or math.isnan(hybrid_value):
        return "missing", math.nan

    delta = hybrid_value - dager_value
    if abs(delta) <= tolerance:
        return "equal", delta

    if metric in TIME_METRICS:
        return ("hybrid_better", delta) if delta < 0 else ("hybrid_worse", delta)

    return ("hybrid_better", delta) if delta > 0 else ("hybrid_worse", delta)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
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


def collect_main_benchmark_pairs(manifest_path: Path) -> list[dict]:
    grouped = defaultdict(list)

    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = parse_manifest_line(raw_line)
        dager_summary = Path(record["dager_summary"])
        hybrid_summary = Path(record["hybrid_summary"])
        if not dager_summary.exists() or not hybrid_summary.exists():
            continue

        grouped[(record["model"], record["dataset"], int(record.get("batch", 8)))].append(
            {
                "seed": int(record["seed"]),
                "dager": load_run_summary(dager_summary),
                "hybrid": load_run_summary(hybrid_summary),
            }
        )

    pairs = []
    for (model, dataset, batch), runs in sorted(grouped.items()):
        pairs.append(
            {
                "source": "main_benchmark",
                "model": model,
                "dataset": dataset,
                "batch": batch,
                "n_seeds": len(runs),
                "runs": runs,
            }
        )
    return pairs


def collect_batch_ablation_pairs(manifest_path: Path) -> list[dict]:
    grouped = defaultdict(lambda: {"dager": [], "hybrid": []})

    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = parse_manifest_line(raw_line)
        attack_dir = "dager_ce" if record["method"] == "dager" else "hybrid_ce"
        summary_path = Path("results") / attack_dir / f"results_{record['hash']}" / "run_summary.json"
        if not summary_path.exists():
            continue

        grouped[(record["model"], record["dataset"], int(record["batch"]))][record["method"]].append(
            {
                "seed": int(record["seed"]),
                "summary": load_run_summary(summary_path),
            }
        )

    pairs = []
    for (model, dataset, batch), methods in sorted(grouped.items()):
        if not methods["dager"] or not methods["hybrid"]:
            continue
        pairs.append(
            {
                "source": "batch_ablation",
                "model": model,
                "dataset": dataset,
                "batch": batch,
                "n_seeds": min(len(methods["dager"]), len(methods["hybrid"])),
                "runs": methods,
            }
        )
    return pairs


def aggregate_metric_values(pair: dict, section_name: str, metric: str) -> tuple[float, float]:
    if pair["source"] == "main_benchmark":
        dager_values = []
        hybrid_values = []
        for run in pair["runs"]:
            dager_section = run["dager"].get(section_name, {})
            hybrid_section = run["hybrid"].get(section_name, {})
            if metric in dager_section:
                dager_values.append(float(dager_section[metric]))
            if metric in hybrid_section:
                hybrid_values.append(float(hybrid_section[metric]))
        return safe_mean(dager_values), safe_mean(hybrid_values)

    dager_values = []
    hybrid_values = []
    for run in pair["runs"]["dager"]:
        section = run["summary"].get(section_name, {})
        if metric in section:
            dager_values.append(float(section[metric]))
    for run in pair["runs"]["hybrid"]:
        section = run["summary"].get(section_name, {})
        if metric in section:
            hybrid_values.append(float(section[metric]))
    return safe_mean(dager_values), safe_mean(hybrid_values)


def build_rows(main_manifest: Path, batch_manifest: Path, tolerance: float) -> tuple[list[dict], list[dict], list[dict]]:
    detailed_rows = []
    summary_rows = []
    count_rows = []
    pairs = []

    if main_manifest.exists():
        pairs.extend(collect_main_benchmark_pairs(main_manifest))
    if batch_manifest.exists():
        pairs.extend(collect_batch_ablation_pairs(batch_manifest))

    for pair in pairs:
        source = pair["source"]
        model = pair["model"]
        dataset = pair["dataset"]
        batch = pair["batch"]

        quality_counts = defaultdict(int)
        time_counts = defaultdict(int)
        quality_advantages = []

        for section_name, spec in SECTION_SPECS.items():
            for metric in spec["metrics"]:
                display_name = DISPLAY_METRIC_NAMES[metric]
                dager_value, hybrid_value = aggregate_metric_values(pair, section_name, metric)
                comparison, delta = classify_metric(display_name, dager_value, hybrid_value, tolerance)
                metric_type = "time" if display_name in TIME_METRICS else "quality"

                detailed_rows.append(
                    {
                        "source": source,
                        "model": model,
                        "dataset": dataset,
                        "batch": batch,
                        "section": section_name,
                        "metric": display_name,
                        "metric_type": metric_type,
                        "n_seeds": pair["n_seeds"],
                        "dager_value": dager_value,
                        "hybrid_value": hybrid_value,
                        "hybrid_minus_dager": delta,
                        "abs_diff": abs(delta) if not math.isnan(delta) else math.nan,
                        "comparison": comparison,
                    }
                )

                if comparison == "missing":
                    continue

                if metric_type == "quality":
                    quality_counts[comparison] += 1
                    if not math.isnan(delta):
                        quality_advantages.append(delta)
                else:
                    time_counts[comparison] += 1

        mean_quality_delta = safe_mean(quality_advantages)
        if quality_counts["hybrid_better"] > quality_counts["hybrid_worse"]:
            quality_case_by_counts = "hybrid_better"
        elif quality_counts["hybrid_worse"] > quality_counts["hybrid_better"]:
            quality_case_by_counts = "hybrid_worse"
        elif quality_counts["hybrid_better"] == 0 and quality_counts["hybrid_worse"] == 0:
            quality_case_by_counts = "equal"
        else:
            quality_case_by_counts = "mixed"

        if math.isnan(mean_quality_delta) or abs(mean_quality_delta) <= tolerance:
            quality_case_by_mean_delta = "equal"
        elif mean_quality_delta > 0:
            quality_case_by_mean_delta = "hybrid_better"
        else:
            quality_case_by_mean_delta = "hybrid_worse"

        summary_rows.append(
            {
                "source": source,
                "model": model,
                "dataset": dataset,
                "batch": batch,
                "n_seeds": pair["n_seeds"],
                "quality_hybrid_better_count": quality_counts["hybrid_better"],
                "quality_equal_count": quality_counts["equal"],
                "quality_hybrid_worse_count": quality_counts["hybrid_worse"],
                "time_hybrid_better_count": time_counts["hybrid_better"],
                "time_equal_count": time_counts["equal"],
                "time_hybrid_worse_count": time_counts["hybrid_worse"],
                "quality_net_count": quality_counts["hybrid_better"] - quality_counts["hybrid_worse"],
                "mean_quality_delta_hybrid_minus_dager": mean_quality_delta,
                "quality_case_by_counts": quality_case_by_counts,
                "quality_case_by_mean_delta": quality_case_by_mean_delta,
            }
        )

    detail_counts = defaultdict(int)
    for row in detailed_rows:
        detail_counts[(row["source"], row["metric_type"], row["comparison"])] += 1
    for (source, metric_type, comparison), count in sorted(detail_counts.items()):
        count_rows.append(
            {
                "row_type": "metric_rows",
                "source": source,
                "metric_type": metric_type,
                "comparison": comparison,
                "count": count,
            }
        )

    summary_case_counts = defaultdict(int)
    for row in summary_rows:
        summary_case_counts[(row["source"], row["quality_case_by_counts"])] += 1
    for (source, comparison), count in sorted(summary_case_counts.items()):
        count_rows.append(
            {
                "row_type": "config_rows",
                "source": source,
                "metric_type": "quality",
                "comparison": comparison,
                "count": count,
            }
        )

    summary_rows.sort(key=lambda row: (row["source"], row["model"], row["dataset"], int(row["batch"])))
    return detailed_rows, summary_rows, count_rows


def main():
    parser = argparse.ArgumentParser(description="Classify paired Hybrid-vs-DAGER results as better, equal, or worse.")
    parser.add_argument(
        "--main-manifest",
        default="logs/main_benchmark_comparison_hashes.txt",
        help="Path to main benchmark paired manifest.",
    )
    parser.add_argument(
        "--batch-manifest",
        default="logs/batch_ablation_hashes.txt",
        help="Path to batch ablation manifest.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance for treating two values as equal.",
    )
    parser.add_argument(
        "--output-detailed",
        default="logs/dager_vs_hybrid_case_details.csv",
        help="Detailed output CSV with one row per section/metric/config.",
    )
    parser.add_argument(
        "--output-summary",
        default="logs/dager_vs_hybrid_case_summary.csv",
        help="Summary output CSV with one row per config.",
    )
    parser.add_argument(
        "--output-counts",
        default="logs/dager_vs_hybrid_case_counts.csv",
        help="Counts CSV summarizing better/equal/worse tallies.",
    )
    args = parser.parse_args()

    detailed_rows, summary_rows, count_rows = build_rows(Path(args.main_manifest), Path(args.batch_manifest), args.tolerance)
    write_csv(Path(args.output_detailed), detailed_rows)
    write_csv(Path(args.output_summary), summary_rows)
    write_csv(Path(args.output_counts), count_rows)

    print(f"Wrote {len(detailed_rows)} rows to {args.output_detailed}")
    print(f"Wrote {len(summary_rows)} rows to {args.output_summary}")
    print(f"Wrote {len(count_rows)} rows to {args.output_counts}")


if __name__ == "__main__":
    main()
