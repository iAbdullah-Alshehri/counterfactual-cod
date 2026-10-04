"""Summarize class-wise residual distributions and bootstrap AUROC (RQ1)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SIGNALS = ("residual_appearance", "residual_feature", "residual_structural")


def auc_score(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.concatenate((positive, negative))
    labels = np.r_[np.ones(len(positive), dtype=bool), np.zeros(len(negative), dtype=bool)]
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc(positive: np.ndarray, negative: np.ndarray, *, iterations: int,
                  seed: int) -> tuple[float, float, float]:
    point = auc_score(positive, negative)
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        pos = positive[rng.integers(0, len(positive), size=len(positive))]
        neg = negative[rng.integers(0, len(negative), size=len(negative))]
        values[index] = auc_score(pos, neg)
    low, high = np.quantile(values, [0.025, 0.975])
    return point, float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.per_image.open(newline="", encoding="utf-8-sig")))
    if not rows:
        parser.error("Per-image results CSV has no rows")
    labels = np.array([row["label"] for row in rows])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs = {
        "CO_vs_negative": (labels == "CO", labels != "CO"),
        "CO_vs_BG": (labels == "CO", labels == "BG"),
        "CO_vs_NOCOD": (labels == "CO", labels == "NOCOD"),
        "NOCOD_vs_BG": (labels == "NOCOD", labels == "BG"),
    }
    report = {"n": len(rows), "label_counts": {k: int((labels == k).sum()) for k in ("CO", "BG", "NOCOD")},
              "bootstrap_iterations": args.bootstrap,
              "note": "AUROC is threshold-independent. Do not select thresholds on the official evaluation split.",
              "signals": {}}
    output_rows = []
    for signal in SIGNALS:
        try:
            scores = np.array([float(row[signal]) for row in rows], dtype=np.float64)
        except (KeyError, ValueError):
            parser.error(f"Missing numeric score column: {signal}")
        groups = {}
        for label in ("CO", "BG", "NOCOD"):
            part = scores[labels == label]
            groups[label] = {"n": int(len(part)), "mean": float(part.mean()) if len(part) else None,
                             "std": float(part.std(ddof=1)) if len(part) > 1 else None,
                             "median": float(np.median(part)) if len(part) else None}
        pair_report = {}
        for index, (pair_name, (positive_mask, negative_mask)) in enumerate(pairs.items()):
            positive, negative = scores[positive_mask], scores[negative_mask]
            if not len(positive) or not len(negative):
                pair_report[pair_name] = {"n_positive": int(len(positive)), "n_negative": int(len(negative)), "auc": None}
                continue
            auc, low, high = bootstrap_auc(positive, negative, iterations=args.bootstrap,
                                            seed=args.seed + index)
            pair_report[pair_name] = {"n_positive": int(len(positive)), "n_negative": int(len(negative)),
                                      "auc": auc, "bootstrap_95ci": [low, high]}
            output_rows.append({"signal": signal, "comparison": pair_name, **pair_report[pair_name]})
        report["signals"][signal] = {"by_label": groups, "separation": pair_report}
    (output / "residual_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "residual_auc.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["signal", "comparison", "n_positive", "n_negative", "auc", "bootstrap_95ci"])
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
