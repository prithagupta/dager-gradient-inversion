#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


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


def classify_failure_shape(prediction: str, reference: str) -> str:
    pred = prediction.strip()
    ref = reference.strip()
    if pred == "":
        return "empty"
    pred_words = pred.split()
    if len(pred_words) <= 2:
        return "very_short"
    if ref.startswith(pred):
        return "prefix_like"
    if len(set(pred_words)) <= max(1, len(pred_words) // 3):
        return "repetitive"
    return "other"


def load_rows(results_root: Path) -> list[dict]:
    rows = []
    for csv_path in sorted(results_root.glob("**/sentence_results.csv")):
        summary_path = csv_path.with_name("run_summary.json")
        if not summary_path.exists():
            continue
        data = json.load(summary_path.open())
        args = data.get("Arguments", {})
        for row in csv.DictReader(csv_path.open()):
            rows.append(
                {
                    "results_dir": str(csv_path.parent),
                    "attack": row["attack"],
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "batch_size": int(args.get("batch_size", 0) or 0),
                    "rng_seed": args.get("rng_seed"),
                    "reference": row["reference"],
                    "prediction": row["prediction"],
                    "rouge1_fm": float(row["rouge1_fm"]),
                    "rouge2_fm": float(row["rouge2_fm"]),
                    "rougeL_fm": float(row["rougeL_fm"]),
                    "exact_match": float(row["exact_match"]),
                    "token_acc": float(row["token_acc"]),
                    "padded_token_acc": float(row["padded_token_acc"]),
                    "pred_len": int(float(row["pred_len"])),
                    "ref_len": int(float(row["ref_len"])),
                }
            )
    return rows


def build_outputs(rows: list[dict], top_k: int):
    summary = defaultdict(lambda: {"n": 0, "failures": 0, "rougeL": 0.0, "exact_match": 0.0, "token_acc": 0.0, "pred_len": 0.0, "ref_len": 0.0, "shapes": Counter(), "predictions": Counter()})
    failures = []

    for row in rows:
        key = (row["attack"], row["model"], row["dataset"], row["batch_size"])
        agg = summary[key]
        agg["n"] += 1
        agg["rougeL"] += row["rougeL_fm"]
        agg["exact_match"] += row["exact_match"]
        agg["token_acc"] += row["token_acc"]
        agg["pred_len"] += row["pred_len"]
        agg["ref_len"] += row["ref_len"]

        if row["exact_match"] < 1.0:
            agg["failures"] += 1
            shape = classify_failure_shape(row["prediction"], row["reference"])
            agg["shapes"][shape] += 1
            agg["predictions"][row["prediction"].strip()] += 1
            failures.append(
                {
                    **row,
                    "failure_shape": shape,
                }
            )

    summary_rows = []
    for key in sorted(summary):
        attack, model, dataset, batch_size = key
        agg = summary[key]
        n = agg["n"]
        top_preds = " | ".join(f"{pred}:{count}" for pred, count in agg["predictions"].most_common(5))
        summary_rows.append(
            {
                "attack": attack,
                "model": model,
                "dataset": dataset,
                "batch_size": batch_size,
                "n_sentences": n,
                "n_failures": agg["failures"],
                "failure_rate": agg["failures"] / n if n else 0.0,
                "mean_rougeL_fm": agg["rougeL"] / n if n else 0.0,
                "mean_exact_match": agg["exact_match"] / n if n else 0.0,
                "mean_token_acc": agg["token_acc"] / n if n else 0.0,
                "mean_pred_len": agg["pred_len"] / n if n else 0.0,
                "mean_ref_len": agg["ref_len"] / n if n else 0.0,
                "very_short_failures": agg["shapes"]["very_short"],
                "prefix_like_failures": agg["shapes"]["prefix_like"],
                "repetitive_failures": agg["shapes"]["repetitive"],
                "empty_failures": agg["shapes"]["empty"],
                "other_failures": agg["shapes"]["other"],
                "top_failed_predictions": top_preds,
            }
        )

    detailed_failures = sorted(
        failures,
        key=lambda row: (row["rougeL_fm"], row["token_acc"], row["exact_match"], row["pred_len"]),
    )[:top_k]
    return summary_rows, detailed_failures


def main():
    parser = argparse.ArgumentParser(description="Analyze sentence-level failures across saved result files.")
    parser.add_argument("--results-root", default="results", help="Root directory containing attack result folders.")
    parser.add_argument("--output-summary", default="logs/sentence_failure_summary.csv", help="Summary CSV by attack/model/dataset/batch.")
    parser.add_argument("--output-hardest", default="logs/sentence_failure_hardest.csv", help="Detailed CSV of the hardest sentence-level failures.")
    parser.add_argument("--top-k", type=int, default=500, help="Number of hardest failures to keep in detailed CSV.")
    args = parser.parse_args()

    rows = load_rows(Path(args.results_root))
    summary_rows, detailed_failures = build_outputs(rows, args.top_k)
    write_csv(Path(args.output_summary), summary_rows)
    write_csv(Path(args.output_hardest), detailed_failures)
    print(f"Wrote {len(summary_rows)} rows to {args.output_summary}")
    print(f"Wrote {len(detailed_failures)} rows to {args.output_hardest}")


if __name__ == "__main__":
    main()
