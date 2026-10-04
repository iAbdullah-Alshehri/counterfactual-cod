"""Sweep thresholds and residual weights on a training-derived validation manifest only."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from cod_recon.evaluate import evaluate


THRESHOLDS = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)
SIGNALS = ("appearance", "feature", "structural")


def _weights_grid() -> list[dict[str, float]]:
    # The 15 unique points on a 0.25-step three-component simplex include equal,
    # single-signal, and mixed appearance/feature/structure conditions.
    return [{name: value / 4 for name, value in zip(SIGNALS, units)}
            for units in itertools.product(range(5), repeat=3) if sum(units) == 4]


def _accuracy(records: list[dict], scores: list[float], threshold: float,
              direction: str) -> dict[str, float]:
    truth = [r["label"] == "CO" for r in records]
    predicted = [score >= threshold if direction == "high" else score <= threshold for score in scores]
    negatives = [not label for label in truth]
    co = [label for label in truth]
    accuracy = float(np.mean([a == b for a, b in zip(truth, predicted)]))
    return {"co_vs_negative_accuracy": accuracy,
            "co_acceptance_rate": float(np.mean([p for p, y in zip(predicted, truth) if y])) if any(co) else 0.0,
            "negative_rejection_accuracy": float(np.mean([not p for p, y in zip(predicted, truth) if not y])) if any(negatives) else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("lama", "opencv"), default="lama")
    parser.add_argument("--inpainting-model", type=Path, default=Path("data/downloads/inpainting_lama_2025jan.onnx"))
    parser.add_argument("--feature-weights", type=Path)
    parser.add_argument("--dilation", type=int, default=5)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "_residual_cache"
    base = evaluate(args.manifest, output / "feature_maps", threshold=0.5, ablation="baseline",
                    backend=args.backend, inpainting_model=args.inpainting_model,
                    feature_weights=args.feature_weights, dilation=args.dilation, cache_dir=cache,
                    decision_rule="binary_threshold")
    with (output / "feature_maps" / "per_image.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records or {r["label"] for r in records} != {"CO", "BG", "NOCOD"}:
        parser.error("Validation manifest must contain at least one CO, BG, and NOCOD sample")

    results: list[dict] = []
    directions = ("high", "low")
    for signal in SIGNALS:
        scores = [float(r[f"residual_{signal}"]) for r in records]
        for direction in directions:
            for threshold in THRESHOLDS:
                metrics = _accuracy(records, scores, threshold, direction)
                results.append({"condition": signal, "threshold": threshold,
                                "acceptance_direction": direction, **metrics,
                                "appearance_weight": None, "feature_weight": None, "structural_weight": None})
    for weights in _weights_grid():
        scores = [sum(float(r[f"residual_{signal}"]) * weights[signal] for signal in SIGNALS) for r in records]
        label = "combined_" + "_".join(f"{k[0]}{v:.2f}" for k, v in weights.items())
        for direction in directions:
            for threshold in THRESHOLDS:
                metrics = _accuracy(records, scores, threshold, direction)
                results.append({"condition": label, "threshold": threshold,
                                "acceptance_direction": direction, **metrics,
                                "appearance_weight": weights["appearance"],
                                "feature_weight": weights["feature"],
                                "structural_weight": weights["structural"]})

    # Accuracy is the stated selection objective; ties prefer retaining real CO,
    # then rejecting negatives, and finally the higher threshold.
    selected = {}
    for condition in sorted({r["condition"] for r in results}):
        group = [r for r in results if r["condition"] == condition]
        best = max(group, key=lambda r: (r["co_vs_negative_accuracy"], r["co_acceptance_rate"],
                                         r["negative_rejection_accuracy"], r["threshold"]))
        selected[condition] = best
    combined_rows = [row for row in selected.values() if row["appearance_weight"] is not None]
    selected["combined_best"] = max(
        combined_rows, key=lambda r: (r["co_vs_negative_accuracy"], r["co_acceptance_rate"],
                                      r["negative_rejection_accuracy"], r["threshold"]))

    csv_path = output / "sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    chosen = {"validation_manifest": str(args.manifest.resolve()),
              "official_test_split_used": False,
              "validation_n": base["n"], "validation_counts": base["metrics_by_label"],
              "thresholds": list(THRESHOLDS), "weight_grid_step": 0.25,
              "selection_objective": "maximize CO-vs-negative accuracy; ties prefer higher CO acceptance, then higher negative rejection, then higher threshold",
              "selected": selected}
    (output / "selected_calibration.json").write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    counts = ", ".join(f"{label}={base['metrics_by_label'][label]['n']}" for label in ("CO", "BG", "NOCOD"))
    lines = ["# Validation-only calibration", "",
             f"Manifest: `{args.manifest.resolve()}` ({base['n']} images; {counts}). The manifest is drawn from USC12K `train.txt`; the official `val.txt` test split was not used.", "",
             "Thresholds swept in both score directions: " + ", ".join(str(t) for t in THRESHOLDS) + ". Combined weights swept over the 0.25-step simplex (15 configurations). The threshold/weight/direction tuple for each condition maximizes validation CO-vs-negative accuracy. Ties prefer higher CO acceptance, then higher negative rejection, then a higher threshold.", "",
             "## Selected settings", "", "| Condition | Direction | Threshold | Weights (appearance, feature, structural) | Accuracy | CO acceptance | Negative rejection |", "|---|---|---:|---|---:|---:|---:|"]
    for condition, row in selected.items():
        weights = "—" if row["appearance_weight"] is None else f"{row['appearance_weight']:.2f}, {row['feature_weight']:.2f}, {row['structural_weight']:.2f}"
        lines.append(f"| {condition} | {row['acceptance_direction']} | {row['threshold']:.3f} | {weights} | {row['co_vs_negative_accuracy']:.3f} | {row['co_acceptance_rate']:.3f} | {row['negative_rejection_accuracy']:.3f} |")
    best_combined = selected["combined_best"]
    if best_combined["feature_weight"] == 0:
        lines += ["", "**Feature note:** the selected combined weights assign zero weight to the learned ResNet-18 feature residual. The embedding pipeline runs, but it did not add validation accuracy in this sample."]
    if best_combined["co_acceptance_rate"] == 0 or best_combined["negative_rejection_accuracy"] == 0:
        lines += ["", "**Calibration warning:** the best-accuracy setting is degenerate: it accepts no CO candidates or rejects no negatives. The residual verifier has not demonstrated useful separation on this validation slice; do not describe the selected settings as an effective detector. The setting is recorded to satisfy reproducible model selection, and further method development is needed before a blinded test evaluation."]
    elif best_combined["co_acceptance_rate"] < 0.5 or best_combined["negative_rejection_accuracy"] < 0.5:
        lines += ["", "**Calibration warning:** although the selected setting is the best by validation accuracy, it has a strongly imbalanced CO-acceptance/negative-rejection tradeoff. Treat it as exploratory, not a validated detector."]
    lines += ["", "These are tuning-set results on a small held-out slice of USC12K's training partition, not an unbiased validation or test estimate: the same labels selected the threshold, direction, and weights. Expect selection optimism. Do not report them as benchmark results. Freeze the selected settings before any later untouched evaluation; do not tune them on the official test split. See `sweep_results.csv` and `selected_calibration.json` for all values.", ""]
    (output / "CALIBRATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}, selected settings, and CALIBRATION.md")
    for condition, row in selected.items():
        print(condition, row["acceptance_direction"], row["threshold"], row["co_vs_negative_accuracy"], flush=True)


if __name__ == "__main__":
    main()
