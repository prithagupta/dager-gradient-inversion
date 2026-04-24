#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


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
    grouped = defaultdict(dict)

    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = parse_manifest_line(raw_line)
        attack_dir = "dager_ce" if record["method"] == "dager" else "hybrid_ce"
        summary_path = Path("results") / attack_dir / f"results_{record['hash']}" / "run_summary.json"
        if not summary_path.exists():
            continue
        grouped[(record["model"], record["dataset"], record["batch"])][record["method"]] = load_run_summary(summary_path)

    long_rows = []
    wide_rows = []
    ranking_rows = []

    for (model, dataset, batch), methods in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], int(x[0][2]))):
        dager = methods.get("dager")
        hybrid = methods.get("hybrid")
        if dager is None or hybrid is None:
            continue

        for section_name, spec in SECTION_SPECS.items():
            wide_row = {
                "model": model,
                "dataset": dataset,
                "batch": int(batch),
                "section": section_name,
            }
            quality_deltas = []

            for metric in spec["metrics"]:
                display_name = DISPLAY_METRIC_NAMES[metric]
                d_value = dager.get(section_name, {}).get(metric)
                h_value = hybrid.get(section_name, {}).get(metric)
                d_value = float(d_value) if d_value is not None else math.nan
                h_value = float(h_value) if h_value is not None else math.nan
                delta = h_value - d_value if not math.isnan(d_value) and not math.isnan(h_value) else math.nan
                abs_delta = abs(delta) if not math.isnan(delta) else math.nan

                long_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "batch": int(batch),
                        "section": section_name,
                        "metric": display_name,
                        "dager_value": d_value,
                        "hybrid_value": h_value,
                        "hybrid_minus_dager": delta,
                        "abs_diff": abs_delta,
                    }
                )

                wide_row[f"dager_{display_name}"] = d_value
                wide_row[f"hybrid_{display_name}"] = h_value
                wide_row[f"hybrid_minus_dager_{display_name}"] = delta
                wide_row[f"abs_diff_{display_name}"] = abs_delta

                if display_name in QUALITY_METRICS and not math.isnan(abs_delta):
                    quality_deltas.append(abs_delta)

            wide_row["mean_abs_quality_gap"] = sum(quality_deltas) / len(quality_deltas) if quality_deltas else math.nan
            wide_rows.append(wide_row)

        overall = next(row for row in wide_rows if row["model"] == model and row["dataset"] == dataset and row["batch"] == int(batch) and row["section"] == "Overall Results")
        per_input = next(row for row in wide_rows if row["model"] == model and row["dataset"] == dataset and row["batch"] == int(batch) and row["section"] == "Per Input Results")
        per_sentence = next(row for row in wide_rows if row["model"] == model and row["dataset"] == dataset and row["batch"] == int(batch) and row["section"] == "Per Sentence Results")

        ranking_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "batch": int(batch),
                "overall_mean_abs_quality_gap": overall["mean_abs_quality_gap"],
                "per_input_mean_abs_quality_gap": per_input["mean_abs_quality_gap"],
                "per_sentence_mean_abs_quality_gap": per_sentence["mean_abs_quality_gap"],
                "overall_token_acc_delta": overall.get("hybrid_minus_dager_token_acc"),
                "per_input_token_acc_delta": per_input.get("hybrid_minus_dager_token_acc"),
                "per_sentence_token_acc_delta": per_sentence.get("hybrid_minus_dager_token_acc"),
                "overall_rougeL_delta": overall.get("hybrid_minus_dager_rougeL_fm"),
                "per_input_rougeL_delta": per_input.get("hybrid_minus_dager_rougeL_fm"),
                "per_sentence_rougeL_delta": per_sentence.get("hybrid_minus_dager_rougeL_fm"),
                "overall_time_delta": overall.get("hybrid_minus_dager_experiment_time_mean"),
                "per_input_time_delta": per_input.get("hybrid_minus_dager_reconstruction_time_mean"),
                "per_sentence_time_delta": per_sentence.get("hybrid_minus_dager_reconstruction_time_mean"),
            }
        )

    ranking_rows.sort(key=lambda row: (row["model"], row["dataset"], -row["overall_mean_abs_quality_gap"], -row["per_input_mean_abs_quality_gap"]))
    return long_rows, wide_rows, ranking_rows


def main():
    parser = argparse.ArgumentParser(description="Compare DAGER and Hybrid batch ablation results by batch size.")
    parser.add_argument(
        "--manifest",
        default="logs/batch_ablation_hashes.txt",
        help="Path to batch ablation hash manifest.",
    )
    parser.add_argument(
        "--output-long",
        default="logs/batch_ablation_results_long.csv",
        help="Output CSV with one row per metric/section/batch.",
    )
    parser.add_argument(
        "--output-wide",
        default="logs/batch_ablation_results_wide.csv",
        help="Output CSV with DAGER and Hybrid side-by-side by batch size.",
    )
    parser.add_argument(
        "--output-ranking",
        default="logs/batch_ablation_batch_ranking.csv",
        help="Output CSV ranking batch sizes by the size of the DAGER-vs-Hybrid gap.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Batch ablation manifest not found: {manifest_path}")

    long_rows, wide_rows, ranking_rows = build_rows(manifest_path)
    write_csv(Path(args.output_long), long_rows)
    write_csv(Path(args.output_wide), wide_rows)
    write_csv(Path(args.output_ranking), ranking_rows)

    print(f"Wrote {len(long_rows)} rows to {args.output_long}")
    print(f"Wrote {len(wide_rows)} rows to {args.output_wide}")
    print(f"Wrote {len(ranking_rows)} rows to {args.output_ranking}")


if __name__ == "__main__":
    main()
