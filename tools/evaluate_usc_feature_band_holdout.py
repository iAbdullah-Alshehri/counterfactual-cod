"""Apply a frozen USC12K feature band to an untouched training-derived holdout."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/experiments/usc12k_feature_band/heldout_manifest.csv"))
    parser.add_argument("--rule", type=Path, default=Path("data/experiments/usc12k_feature_band/calibration/selected_band_rule.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/usc12k_feature_band/heldout"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/experiments/usc12k_feature_band/residual_cache"))
    parser.add_argument("--backend", choices=("lama", "opencv"), default="lama")
    parser.add_argument("--inpainting-model", type=Path, default=Path("data/downloads/inpainting_lama_2025jan.onnx"))
    parser.add_argument("--feature-weights", type=Path, default=Path("data/downloads/resnet18-f37072fd.pth"))
    args = parser.parse_args()
    rule = json.loads(args.rule.read_text(encoding="utf-8"))
    low, high = float(rule["low_cut"]), float(rule["high_cut"])
    if not low < high:
        parser.error("Rule low_cut must be less than high_cut")
    out = args.output_dir.resolve()
    evaluate(args.manifest, out / "feature_score_pass", threshold=0.5,
             weights={"appearance": 0.0, "feature": 1.0, "structural": 0.0},
             backend=args.backend, ablation="feature", inpainting_model=args.inpainting_model,
             feature_weights=args.feature_weights, cache_dir=args.cache_dir,
             decision_rule="binary_threshold")
    with (out / "feature_score_pass" / "per_image.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    matrix = np.zeros((3, 3), dtype=int)
    idx = {label: i for i, label in enumerate(LABELS)}
    result_rows = []
    for row in rows:
        predicted = classify(float(row["residual_feature"]), low, high)
        matrix[idx[row["label"]], idx[predicted]] += 1
        result_rows.append({**row, "band_prediction": predicted,
                            "band_correct": predicted == row["label"]})
    n = int(matrix.sum())
    accuracy = float(np.trace(matrix) / n)
    recalls = {label: (float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else None)
               for i, label in enumerate(LABELS)}
    precisions = {label: (float(matrix[i, i] / matrix[:, i].sum()) if matrix[:, i].sum() else None)
                  for i, label in enumerate(LABELS)}
    f1s = {label: (2 * precisions[label] * recalls[label] / (precisions[label] + recalls[label])
                   if precisions[label] is not None and recalls[label] is not None and
                   precisions[label] + recalls[label] > 0 else 0.0) for label in LABELS}
    macro_f1 = float(np.mean(list(f1s.values())))
    summary = {"frozen_rule": rule, "n": n,
               "n_by_class": {label: int(matrix[i].sum()) for i, label in enumerate(LABELS)},
               "accuracy": accuracy, "macro_f1": macro_f1, "recall_by_class": recalls,
               "precision_by_class": precisions, "f1_by_class": f1s,
               "confusion_order": list(LABELS),
               "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
               "official_usc_test_used": False,
               "note": "Independent sample from USC12K training partition; rule cut points were frozen using the separate calibration sample. Not official test performance."}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "heldout_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    (out / "heldout_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Frozen two-cut rule: independent USC12K holdout", "",
             f"Applied the preselected rule with no changes: BG if score < {low:.3f}; CO if {low:.3f} ≤ score ≤ {high:.3f}; NOCOD if score > {high:.3f}.", "",
             "The 45 examples (15 per class) are from the USC12K training partition and are disjoint from previous experiment manifests and the calibration sample. This is a held-out training-derived check, not the official USC12K test split.", "",
             f"Three-way accuracy: **{accuracy:.3f}**; macro-F1: **{macro_f1:.3f}**.", "",
             "Confusion matrix (true rows, predicted columns; order BG, CO, NOCOD):", "",
             "| True \\ Predicted | BG | CO | NOCOD |", "|---|---:|---:|---:|"]
    for i, label in enumerate(LABELS):
        lines.append(f"| {label} | {matrix[i,0]} | {matrix[i,1]} | {matrix[i,2]} |")
    lines += ["", "| Class | Recall | Precision | F1 |", "|---|---:|---:|---:|"]
    for label in LABELS:
        lines.append(f"| {label} | {recalls[label]:.3f} | {precisions[label]:.3f} | {f1s[label]:.3f} |")
    lines += ["", "The holdout was used only after selecting and saving the calibration cut points. Source family and class remain associated in this USC12K construction; source composition is recorded in the parent `sample_selection.json`. Do not tune on this result.", ""]
    (out / "USC_BAND_HOLDOUT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Frozen band [{low:.3f}, {high:.3f}]; holdout accuracy={accuracy:.3f}; macro-F1={macro_f1:.3f}", flush=True)
    print(f"Confusion order={LABELS}; matrix={matrix.tolist()}; recalls={recalls}", flush=True)


if __name__ == "__main__":
    main()
