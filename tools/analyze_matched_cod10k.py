"""Measure low-residual CO discrimination on same-source COD10K CAM/NonCAM samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SIGNALS = ("residual_appearance", "residual_feature", "residual_structural", "verification_score")


def auc_high(positive: np.ndarray, negative: np.ndarray) -> float:
    differences = positive[:, None] - negative[None, :]
    return float(((differences > 0).sum() + 0.5 * (differences == 0).sum()) / differences.size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--direction", choices=("low", "high"), default="low")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    with args.per_image.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    co = [row for row in rows if row["label"] == "CO"]
    bg = [row for row in rows if row["label"] == "BG"]
    if not co or not bg:
        parser.error("Expected matched CO and BG rows")
    rng = np.random.default_rng(args.seed)
    signal_results, distributions = {}, {}
    for signal in SIGNALS:
        pos = np.array([float(row[signal]) for row in co])
        neg = np.array([float(row[signal]) for row in bg])
        high_auc = auc_high(pos, neg)
        samples = np.empty(args.bootstrap)
        for i in range(args.bootstrap):
            a = rng.choice(pos, size=len(pos), replace=True)
            b = rng.choice(neg, size=len(neg), replace=True)
            samples[i] = auc_high(a, b)
        ci_high = [float(x) for x in np.quantile(samples, [0.025, 0.975])]
        positive_score_auc = high_auc if args.direction == "high" else 1.0 - high_auc
        positive_ci = ci_high if args.direction == "high" else [1.0 - ci_high[1], 1.0 - ci_high[0]]
        signal_results[signal] = {"n_CO": len(co), "n_BG": len(bg),
                                  "auc_high_means_CO": high_auc,
                                  "auc_high_bootstrap_95ci": ci_high,
                                  "auc_selected_direction": positive_score_auc,
                                  "auc_selected_direction_bootstrap_95ci": positive_ci}
        distributions[signal] = {
            "CO_mean": float(pos.mean()), "CO_median": float(np.median(pos)),
            "BG_mean": float(neg.mean()), "BG_median": float(np.median(neg)),
        }
    score_col = "verification_score"
    co_scores = np.array([float(row[score_col]) for row in co])
    bg_scores = np.array([float(row[score_col]) for row in bg])
    co_accept = co_scores <= args.threshold if args.direction == "low" else co_scores >= args.threshold
    bg_accept = bg_scores <= args.threshold if args.direction == "low" else bg_scores >= args.threshold
    selected_auc = signal_results[score_col]["auc_selected_direction"]
    summary = {"n_CO": len(co), "n_same_source_BG": len(bg), "source_family": "COD10K",
               "threshold": args.threshold, "acceptance_direction": args.direction,
               "co_acceptance_rate": float(co_accept.mean()),
               "same_source_BG_false_accept_rate": float(bg_accept.mean()),
               "co_vs_same_source_BG_accuracy": float((co_accept.sum() + (~bg_accept).sum()) / (len(co) + len(bg))),
               "selected_score_AUROC_CO_positive": signal_results[score_col]["auc_selected_direction"],
               "signals": signal_results, "score_distributions": distributions,
               "caveat": "Small source-matched diagnostic, not the official USC12K test. COD10K CAM positives are from COD-TestDataset; COD10K NonCAM negatives are from the USC12K training partition. Exact filename overlap with SINet-V2 COD training images was excluded. SINet-V2 was trained on related COD data, so candidate generation may still carry dataset/task bias."}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "matched_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# COD10K source-matched diagnostic", "",
             f"Compared {len(co)} COD10K-CAM positives with {len(bg)} COD10K-NonCAM negatives; both share the COD10K source family. Selected rule: accept CO when score is {args.direction} {args.threshold:.2f}.", "",
             "| Signal | CO mean (median) | BG mean (median) | AUROC if high means CO (95% bootstrap CI) | AUROC if low means CO |", "|---|---:|---:|---:|---:|"]
    for signal, result in signal_results.items():
        dist = distributions[signal]
        low, high = result["auc_selected_direction_bootstrap_95ci"]
        lines.append(f"| {signal} | {dist['CO_mean']:.3f} ({dist['CO_median']:.3f}) | {dist['BG_mean']:.3f} ({dist['BG_median']:.3f}) | {result['auc_high_means_CO']:.3f} ({result['auc_high_bootstrap_95ci'][0]:.3f}–{result['auc_high_bootstrap_95ci'][1]:.3f}) | {1 - result['auc_high_means_CO']:.3f} |")
    lines += ["", f"At the frozen 0.10 threshold: CO acceptance={summary['co_acceptance_rate']:.3f}; same-source BG false acceptance={summary['same_source_BG_false_accept_rate']:.3f}; accuracy={summary['co_vs_same_source_BG_accuracy']:.3f}.", "",
              f"For the selected combined score, AUROC in its configured `{args.direction}` CO-positive direction is {selected_auc:.3f} (95% bootstrap CI {signal_results[score_col]['auc_selected_direction_bootstrap_95ci'][0]:.3f}–{signal_results[score_col]['auc_selected_direction_bootstrap_95ci'][1]:.3f}). The opposite score direction is also shown in the table; individual residual directions are reported both ways and were not retuned.", "",
              "This is a small source-matched diagnostic, not official USC12K test performance. Positive COD10K-CAM examples come from the authors' COD-TestDataset and negatives from USC12K's training-derived COD10K-NonCAM pool. Exact filename overlaps with SINet-V2's COD training archive were excluded. Since SINet-V2 was trained on related COD data, a remaining baseline/task bias is possible. Do not tune the threshold on this diagnostic.", ""]
    (output / "MATCHED_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"CO acceptance={summary['co_acceptance_rate']:.3f}; matched BG false acceptance={summary['same_source_BG_false_accept_rate']:.3f}; AUROC={selected_auc:.3f}")
    print(f"Wrote {output / 'MATCHED_DIAGNOSTICS.md'}")


if __name__ == "__main__":
    main()
