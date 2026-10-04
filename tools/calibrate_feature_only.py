"""Calibrate a ResNet-feature-only CO rule on the fresh COD10K development set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cod_recon.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/experiments/cod10k_feature_only/dev_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/cod10k_feature_only/calibration"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/experiments/cod10k_feature_only/residual_cache"))
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20,
                                 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70])
    parser.add_argument("--backend", choices=("lama", "opencv"), default="lama")
    parser.add_argument("--inpainting-model", type=Path, default=Path("data/downloads/inpainting_lama_2025jan.onnx"))
    parser.add_argument("--feature-weights", type=Path, default=Path("data/downloads/resnet18-f37072fd.pth"))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = evaluate(args.manifest, out / "feature_score_pass", threshold=0.5,
                      weights={"appearance": 0.0, "feature": 1.0, "structural": 0.0},
                      backend=args.backend, ablation="feature",
                      inpainting_model=args.inpainting_model, feature_weights=args.feature_weights,
                      cache_dir=args.cache_dir, decision_rule="binary_threshold")
    per_image = out / "feature_score_pass" / "per_image.csv"
    with per_image.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    positive = np.array([float(r["residual_feature"]) for r in rows if r["label"] == "CO"])
    negative = np.array([float(r["residual_feature"]) for r in rows if r["label"] != "CO"])
    if not len(positive) or not len(negative):
        parser.error("Development manifest must contain both CO and negative examples")
    table = []
    for threshold in args.thresholds:
        for direction in ("high", "low"):
            pa = positive >= threshold if direction == "high" else positive <= threshold
            na = negative >= threshold if direction == "high" else negative <= threshold
            co_recall, false_accept = float(pa.mean()), float(na.mean())
            table.append({"threshold": threshold, "direction": direction,
                          "co_acceptance_rate": co_recall,
                          "negative_false_accept_rate": false_accept,
                          "co_vs_negative_accuracy": (float(pa.sum()) + float((~na).sum())) / (len(pa) + len(na))})
    # Deterministic tie break prefers higher balanced accuracy (same here because
    # the dev set is balanced), then higher CO recall, then simpler low threshold.
    selected = max(table, key=lambda r: (r["co_vs_negative_accuracy"], r["co_acceptance_rate"], -r["threshold"], r["direction"] == "high"))
    with (out / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    rule = {"signal": "residual_feature (frozen ImageNet ResNet-18 layer2 cosine distance)",
            "weights": {"appearance": 0.0, "feature": 1.0, "structural": 0.0},
            "threshold": selected["threshold"], "acceptance_direction": selected["direction"],
            "selection_metric": "co_vs_negative_accuracy", "development_n_CO": len(positive),
            "development_n_negative": len(negative), "development_accuracy": selected["co_vs_negative_accuracy"],
            "development_co_acceptance_rate": selected["co_acceptance_rate"],
            "development_negative_false_accept_rate": selected["negative_false_accept_rate"],
            "is_tuning_score_not_unbiased_estimate": True,
            "threshold_candidates": args.thresholds}
    (out / "selected_feature_rule.json").write_text(json.dumps(rule, indent=2) + "\n", encoding="utf-8")
    lines = ["# Feature-only threshold calibration", "",
             "The rule uses only the frozen pretrained ResNet-18 layer2 feature residual. Appearance and structural residual weights are exactly zero. The fresh, balanced development set contains COD10K-CAM positives and USC12K training-derived COD10K-NonCAM negatives; its IDs were selected disjointly from earlier diagnostic/validation manifests and SINet-V2 training stems.", "",
             "Thresholds were swept in both high-score-means-CO and low-score-means-CO directions. The selected point maximizes CO-vs-negative accuracy. Ties prefer higher CO acceptance, then the lower threshold, then high direction. No USC sanity rows or official USC test data were used for selection.", "",
             f"Selected rule: accept CO when feature score is **{selected['direction']} {selected['threshold']:.3f}**.", "",
             f"Development tuning accuracy: {selected['co_vs_negative_accuracy']:.3f} ({len(positive)} CO, {len(negative)} negatives); CO acceptance {selected['co_acceptance_rate']:.3f}; negative false acceptance {selected['negative_false_accept_rate']:.3f}.", "",
             "This is a tuning score, not an unbiased estimate. The operating point is frozen before the independent USC12K-source sanity check. Full threshold results are in `threshold_sweep.csv`.", ""]
    (out / "FEATURE_RULE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Selected feature-only rule: {selected['direction']} {selected['threshold']:.3f}; accuracy={selected['co_vs_negative_accuracy']:.3f}", flush=True)
    # Apply the selected rule to development rows for a directly auditable report.
    applied = evaluate(args.manifest, out / "selected_rule_on_dev", threshold=selected["threshold"],
                       weights=rule["weights"], backend=args.backend, ablation="feature",
                       inpainting_model=args.inpainting_model, feature_weights=args.feature_weights,
                       cache_dir=args.cache_dir, acceptance_direction=selected["direction"],
                       decision_rule="binary_threshold")
    print(f"Applied selected rule; summary accuracy={applied['co_vs_negative_accuracy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
