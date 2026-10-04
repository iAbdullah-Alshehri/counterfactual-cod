"""Calibrate a three-way BG / CO / NOCOD feature-score band on USC12K train data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cod_recon.evaluate import evaluate


LABELS = ("BG", "CO", "NOCOD")


def classify(score: float, low: float, high: float) -> str:
    if score < low:
        return "BG"
    if score > high:
        return "NOCOD"
    return "CO"


def confusion(rows: list[dict], low: float, high: float) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=int)
    idx = {label: i for i, label in enumerate(LABELS)}
    for row in rows:
        matrix[idx[row["label"]], idx[classify(float(row["residual_feature"]), low, high)]] += 1
    return matrix


def metrics(matrix: np.ndarray) -> tuple[float, float]:
    accuracy = float(np.trace(matrix) / matrix.sum())
    tp = np.diag(matrix).astype(float)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(3), where=(2 * tp + fp + fn) > 0)
    return accuracy, float(f1.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/experiments/usc12k_feature_band/calibration_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/usc12k_feature_band/calibration"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/experiments/usc12k_feature_band/residual_cache"))
    parser.add_argument("--backend", choices=("lama", "opencv"), default="lama")
    parser.add_argument("--inpainting-model", type=Path, default=Path("data/downloads/inpainting_lama_2025jan.onnx"))
    parser.add_argument("--feature-weights", type=Path, default=Path("data/downloads/resnet18-f37072fd.pth"))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    evaluate(args.manifest, out / "feature_score_pass", threshold=0.5,
             weights={"appearance": 0.0, "feature": 1.0, "structural": 0.0},
             backend=args.backend, ablation="feature", inpainting_model=args.inpainting_model,
             feature_weights=args.feature_weights, cache_dir=args.cache_dir,
             decision_rule="binary_threshold")
    with (out / "feature_score_pass" / "per_image.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if {r["label"] for r in rows} != set(LABELS) or any(sum(r["label"] == label for r in rows) < 30 for label in LABELS):
        parser.error("Calibration manifest must contain at least 30 samples of each of BG, CO, and NOCOD")

    # Fixed, predeclared grid over the residual's [0,1] scale, with 0.005 steps.
    cuts = np.round(np.arange(0.0, 0.6001, 0.005), 3)
    sweep = []
    for low in cuts[:-1]:
        for high in cuts[cuts > low]:
            matrix = confusion(rows, float(low), float(high))
            accuracy, macro_f1 = metrics(matrix)
            sweep.append({"low_cut": float(low), "high_cut": float(high),
                          "three_way_accuracy": accuracy, "macro_f1": macro_f1,
                          "band_width": float(high - low)})
    # Primary criterion is three-way accuracy. Resolve exact ties using macro-F1,
    # then a wider CO band, then lower low_cut for deterministic reproducibility.
    selected = max(sweep, key=lambda r: (r["three_way_accuracy"], r["macro_f1"],
                                         r["band_width"], -r["low_cut"]))
    matrix = confusion(rows, selected["low_cut"], selected["high_cut"])
    with (out / "two_cut_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    rule = {"signal": "residual_feature (frozen ImageNet ResNet-18 layer2 cosine distance)",
            "weights": {"appearance": 0.0, "feature": 1.0, "structural": 0.0},
            "decision_rule": "score < low_cut => BG; low_cut <= score <= high_cut => CO; score > high_cut => NOCOD",
            "low_cut": selected["low_cut"], "high_cut": selected["high_cut"],
            "selection_metric": "three_way_accuracy", "tie_breaks": ["macro_f1", "wider_CO_band", "lower_low_cut"],
            "calibration_n_by_class": {label: sum(r["label"] == label for r in rows) for label in LABELS},
            "calibration_accuracy": selected["three_way_accuracy"], "calibration_macro_f1": selected["macro_f1"],
            "calibration_confusion_order": list(LABELS),
            "calibration_confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
            "calibration_is_training_tuning_score_not_unbiased_estimate": True,
            "grid": {"min": 0.0, "max": 0.6, "step": 0.005}}
    (out / "selected_band_rule.json").write_text(json.dumps(rule, indent=2) + "\n", encoding="utf-8")
    lines = ["# USC12K train-derived two-cut feature calibration", "",
             "This calibration uses only fresh USC12K training-partition samples (40 each of CO, BG, NOCOD). No COD10K proxy or official USC12K test image was used. Exact IDs from previous experiment manifests and SINet-V2 COD training stems were excluded. The frozen signal is ResNet-18 layer2 feature residual only; appearance and structural weights are zero.", "",
             "The predeclared grid searched every ordered pair of cut points from 0.000 to 0.600 in increments of 0.005. Scores below low_cut are labeled BG; scores above high_cut are labeled NOCOD; scores in the inclusive band are labeled CO. Selection maximized three-way accuracy. Exact ties were resolved by macro-F1, then wider CO band, then lower low_cut.", "",
             f"Selected rule: BG if score < **{selected['low_cut']:.3f}**; CO if **{selected['low_cut']:.3f} ≤ score ≤ {selected['high_cut']:.3f}**; NOCOD if score > **{selected['high_cut']:.3f}**.", "",
             f"Calibration three-way accuracy: **{selected['three_way_accuracy']:.3f}**; macro-F1: **{selected['macro_f1']:.3f}**. This is a training tuning score, not an unbiased estimate.", "",
             "Confusion matrix (true rows, predicted columns; order BG, CO, NOCOD):", "",
             "| True \\ Predicted | BG | CO | NOCOD |", "|---|---:|---:|---:|"]
    for i, label in enumerate(LABELS):
        lines.append(f"| {label} | {matrix[i,0]} | {matrix[i,1]} | {matrix[i,2]} |")
    lines += ["", "Source-family counts are provided in the parent `sample_selection.json`; source and class are still associated in USC12K. Full cut-pair sweep is in `two_cut_sweep.csv`. This rule is frozen before evaluating the independent holdout sample.", ""]
    (out / "BAND_CALIBRATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Selected band BG < {selected['low_cut']:.3f}; CO through {selected['high_cut']:.3f}; NOCOD > {selected['high_cut']:.3f}", flush=True)
    print(f"Calibration accuracy={selected['three_way_accuracy']:.3f}; macro-F1={selected['macro_f1']:.3f}; confusion order={LABELS}; matrix={matrix.tolist()}", flush=True)


if __name__ == "__main__":
    main()
