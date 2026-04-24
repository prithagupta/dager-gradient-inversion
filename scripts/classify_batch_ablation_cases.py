#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


HIGHER_IS_BETTER = {
    "rouge1_fm",
    "rouge2_fm",
    "rougeL_fm",
    "exact_match",
    "token_acc",
    "padded_token_acc",
}

LOWER_IS_BETTER = {
    "experiment_time_mean",
    "reconstruction_time_mean",
}


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


def classify(metric: str, delta: float, tol: float) -> str:
    if abs(delta) <= tol:
        return "equal"
    if metric in HIGHER_IS_BETTER:
        return "hybrid_better" if delta > 0 else "hybrid_worse"
    if metric in LOWER_IS_BETTER:
        return "hybrid_better" if delta < 0 else "hybrid_worse"
    return "equal"


def main():
    parser = argparse.ArgumentParser(description="Classify batch ablation cases into Hybrid better/equal/worse.")
    parser.add_argument(
        "--input",
        default="logs/batch_ablation_results_long.csv",
        help="Input long-format batch ablation comparison CSV.",
    )
    parser.add_argument(
        "--output-detail",
        default="logs/batch_ablation_case_labels.csv",
        help="Output CSV with one row per batch/section/metric labeled better/equal/worse.",
    )
    parser.add_argument(
        "--output-summary",
        default="logs/batch_ablation_case_summary.csv",
        help="Output CSV summarizing better/equal/worse counts by model/dataset/batch/section.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute delta tolerance for treating two results as equal.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    detail_rows = []
    summary = defaultdict(lambda: {"hybrid_better": 0, "equal": 0, "hybrid_worse": 0})

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row["metric"]
            delta = float(row["hybrid_minus_dager"])
            label = classify(metric, delta, args.tolerance)
            detail_row = {
                "model": row["model"],
                "dataset": row["dataset"],
                "batch": int(row["batch"]),
                "section": row["section"],
                "metric": metric,
                "dager_value": float(row["dager_value"]),
                "hybrid_value": float(row["hybrid_value"]),
                "hybrid_minus_dager": delta,
                "case_label": label,
            }
            detail_rows.append(detail_row)

            summary_key = (
                row["model"],
                row["dataset"],
                int(row["batch"]),
                row["section"],
            )
            summary[summary_key][label] += 1

    detail_rows.sort(key=lambda r: (r["model"], r["dataset"], r["batch"], r["section"], r["metric"]))

    summary_rows = []
    for (model, dataset, batch, section), counts in sorted(summary.items()):
        total = counts["hybrid_better"] + counts["equal"] + counts["hybrid_worse"]
        summary_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "batch": batch,
                "section": section,
                "hybrid_better_count": counts["hybrid_better"],
                "equal_count": counts["equal"],
                "hybrid_worse_count": counts["hybrid_worse"],
                "total_metrics": total,
            }
        )

    write_csv(Path(args.output_detail), detail_rows)
    write_csv(Path(args.output_summary), summary_rows)

    print(f"Wrote {len(detail_rows)} rows to {args.output_detail}")
    print(f"Wrote {len(summary_rows)} rows to {args.output_summary}")


if __name__ == "__main__":
    main()
