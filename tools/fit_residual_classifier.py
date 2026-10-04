"""Fit a fixed, regularized 3-class logistic model on cached residual scalars.

The model is trained only from the saved 40-per-class calibration output. It
reports results on the previously used 15-per-class USC holdout and the
previously inspected official split as exploratory comparisons; neither is
used for fitting, scaling, or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


LABELS = ("BG", "CO", "NOCOD")
FEATURES = ("residual_appearance", "residual_feature", "residual_structural")


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No per-image rows in {path}")
    required = {"image_id", "label", *FEATURES}
    if not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate image_id in {path}")
    for row in rows:
        if row["label"].upper() not in LABELS:
            raise ValueError(f"Unknown class label {row['label']!r} in {path}")
        for name in FEATURES:
            value = float(row[name])
            if not np.isfinite(value):
                raise ValueError(f"Non-finite {name} in {path}")
    return rows


def matrix(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row[name]) for name in FEATURES] for row in rows], dtype=np.float64)
    y = np.asarray([LABELS.index(row["label"].upper()) for row in rows], dtype=np.int64)
    if set(y.tolist()) != {0, 1, 2}:
        raise ValueError("Each data split must contain BG, CO, and NOCOD")
    return x, y


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, learning_rate: float,
                 max_iterations: int, tolerance: float) -> tuple[np.ndarray, np.ndarray, dict]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    n, d = z.shape
    one_hot = np.eye(len(LABELS), dtype=np.float64)[y]
    weights = np.zeros((d, len(LABELS)), dtype=np.float64)
    bias = np.zeros(len(LABELS), dtype=np.float64)
    previous_loss = float("inf")
    for iteration in range(1, max_iterations + 1):
        probabilities = softmax(z @ weights + bias)
        loss = float(-np.log(np.maximum(probabilities[np.arange(n), y], 1e-15)).mean()
                     + 0.5 * l2 * np.square(weights).sum())
        error = probabilities - one_hot
        gradient_w = z.T @ error / n + l2 * weights
        gradient_b = error.mean(axis=0)
        weights -= learning_rate * gradient_w
        bias -= learning_rate * gradient_b
        if abs(previous_loss - loss) < tolerance:
            break
        previous_loss = loss
    final_probabilities = softmax(z @ weights + bias)
    final_loss = float(-np.log(np.maximum(final_probabilities[np.arange(n), y], 1e-15)).mean()
                       + 0.5 * l2 * np.square(weights).sum())
    metadata = {"iterations": iteration, "training_objective": final_loss,
                "l2_penalty": l2, "learning_rate": learning_rate,
                "max_iterations": max_iterations, "tolerance": tolerance}
    return mean, scale, {"weights": weights, "bias": bias, **metadata}


def predict(x: np.ndarray, mean: np.ndarray, scale: np.ndarray,
            weights: np.ndarray, bias: np.ndarray) -> tuple[list[str], np.ndarray]:
    probabilities = softmax(((x - mean) / scale) @ weights + bias)
    indices = probabilities.argmax(axis=1)
    return [LABELS[int(index)] for index in indices], probabilities


def band_predictions(rows: list[dict[str, str]], low: float, high: float) -> list[str]:
    predictions = []
    for row in rows:
        score = float(row["residual_feature"])
        predictions.append("BG" if score < low else "NOCOD" if score > high else "CO")
    return predictions


def metrics(rows: list[dict[str, str]], predictions: list[str]) -> tuple[np.ndarray, dict]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for row, pred in zip(rows, predictions):
        confusion[LABELS.index(row["label"].upper()), LABELS.index(pred)] += 1
    tp = np.diag(confusion).astype(float)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(3), where=2 * tp + fp + fn > 0)
    recall = np.divide(tp, confusion.sum(axis=1), out=np.zeros(3), where=confusion.sum(axis=1) > 0)
    negative_total = int(confusion[0].sum() + confusion[2].sum())
    negative_false_accept = int(confusion[0, 1] + confusion[2, 1])
    return confusion, {
        "n": int(confusion.sum()),
        "three_way_accuracy": float(tp.sum() / max(1, confusion.sum())),
        "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
        "recall_by_class": {label: float(recall[i]) for i, label in enumerate(LABELS)},
        "f1_by_class": {label: float(f1[i]) for i, label in enumerate(LABELS)},
        "negative_false_accept_rate": negative_false_accept / max(1, negative_total),
        "negative_false_accept_rate_by_class": {
            "BG": float(confusion[0, 1] / max(1, confusion[0].sum())),
            "NOCOD": float(confusion[2, 1] / max(1, confusion[2].sum()))},
        "co_acceptance_rate": float(confusion[1, 1] / max(1, confusion[1].sum())),
        "confusion_order": list(LABELS),
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
    }


def bootstrap_ci(rows: list[dict[str, str]], predictions: list[str],
                 replicates: int, seed: int) -> dict[str, dict[str, float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["label"].upper()].append(i)
    if set(groups) != set(LABELS):
        raise ValueError("Bootstrap requires all three classes")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        ids = np.concatenate([rng.choice(groups[label], size=len(groups[label]), replace=True)
                              for label in LABELS])
        sampled_rows = [rows[int(i)] for i in ids]
        sampled_predictions = [predictions[int(i)] for i in ids]
        _, result = metrics(sampled_rows, sampled_predictions)
        for name in ("three_way_accuracy", "macro_f1", "balanced_accuracy",
                     "negative_false_accept_rate", "co_acceptance_rate"):
            samples[name].append(float(result[name]))
        for label in LABELS:
            samples[f"recall_{label}"].append(float(result["recall_by_class"][label]))
        for label in ("BG", "NOCOD"):
            samples[f"negative_false_accept_rate_{label}"].append(
                float(result["negative_false_accept_rate_by_class"][label]))
    return {name: {"lower_95": float(np.quantile(values, 0.025)),
                   "upper_95": float(np.quantile(values, 0.975))}
            for name, values in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path,
                        default=Path("data/experiments/usc12k_feature_band/calibration/feature_score_pass/per_image.csv"))
    parser.add_argument("--heldout", type=Path,
                        default=Path("data/experiments/usc12k_feature_band/heldout/feature_score_pass/per_image.csv"))
    parser.add_argument("--official", type=Path,
                        default=Path("data/experiments/usc12k/full/feature_band_lama_full/feature_band_verify/per_image.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/experiments/usc12k/full/feature_band_lama_full/residual_classifier"))
    parser.add_argument("--low-cut", type=float, default=0.185)
    parser.add_argument("--high-cut", type=float, default=0.285)
    parser.add_argument("--l2", type=float, default=0.01,
                        help="Fixed L2 penalty; do not tune this against the official split")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if not 0 <= args.low_cut < args.high_cut <= 1:
        parser.error("Cuts must satisfy 0 <= low-cut < high-cut <= 1")
    if args.l2 < 0 or args.learning_rate <= 0 or args.max_iterations < 1:
        parser.error("Invalid optimization settings")
    if args.bootstrap_replicates < 100:
        parser.error("Use at least 100 bootstrap replicates")

    train_rows = load_rows(args.calibration)
    holdout_rows = load_rows(args.heldout)
    official_rows = load_rows(args.official)
    train_ids = {row["image_id"] for row in train_rows}
    holdout_ids = {row["image_id"] for row in holdout_rows}
    official_ids = {row["image_id"] for row in official_rows}
    if train_ids & holdout_ids or train_ids & official_ids or holdout_ids & official_ids:
        raise ValueError("Calibration, holdout, and official image IDs must be disjoint")
    if any(sum(row["label"].upper() == label for row in train_rows) != 40 for label in LABELS):
        raise ValueError("Expected the documented 40-per-class calibration set")
    if any(sum(row["label"].upper() == label for row in holdout_rows) != 15 for label in LABELS):
        raise ValueError("Expected the documented 15-per-class holdout")

    x_train, y_train = matrix(train_rows)
    mean, scale, model = fit_logistic(x_train, y_train, args.l2, args.learning_rate,
                                      args.max_iterations, args.tolerance)
    weights, bias = model.pop("weights"), model.pop("bias")
    summary = {
        "analysis_type": "posthoc_fixed_logistic_regression_on_cached_residual_scalars",
        "feature_order": list(FEATURES), "class_order": list(LABELS),
        "training_n": len(train_rows),
        "training_ids_sha256": __import__("hashlib").sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
        "preprocessing": "Feature-wise standardization fitted on calibration rows only.",
        "model": {**model, "standardization_mean": mean.tolist(), "standardization_scale": scale.tolist(),
                  "coefficients_rows_features_columns_classes": weights.tolist(),
                  "intercepts_columns_classes": bias.tolist()},
        "comparison_band": {"low_cut": args.low_cut, "high_cut": args.high_cut,
                            "rule": "BG below low_cut; CO inclusive band; NOCOD above high_cut"},
        "evaluation": {},
        "limitations": [
            "The classifier is trained only on the 120 saved calibration rows; no threshold or hyperparameter search is performed.",
            "The 45-row holdout has already been used to assess the feature band, so this is a post-hoc exploratory comparison, not an untouched validation.",
            "The official split has already been inspected for the feature-band project; its results here are descriptive only and must not be used to select the model.",
            "USC12K class labels remain associated with data-source families; this experiment does not establish source-independent generalization.",
        ],
    }
    prediction_rows: list[dict] = []
    for name, rows in (("previously_used_training_holdout", holdout_rows),
                       ("previously_inspected_official_split", official_rows)):
        x, _ = matrix(rows)
        logistic_predictions, probabilities = predict(x, mean, scale, weights, bias)
        band = band_predictions(rows, args.low_cut, args.high_cut)
        split_metrics = {}
        for method, pred in (("logistic_regression", logistic_predictions), ("feature_band", band)):
            _, result = metrics(rows, pred)
            result["bootstrap_95_ci"] = bootstrap_ci(rows, pred, args.bootstrap_replicates,
                                                      args.seed + (0 if name.startswith("previously_used") else 1000)
                                                      + (0 if method == "logistic_regression" else 17))
            split_metrics[method] = result
        summary["evaluation"][name] = split_metrics
        for row, model_prediction, band_prediction, probability in zip(
                rows, logistic_predictions, band, probabilities):
            output_row = {"image_id": row["image_id"], "label": row["label"], "split": row.get("split", ""),
                          "logistic_prediction": model_prediction, "feature_band_prediction": band_prediction,
                          **{f"p_{label}": float(probability[i]) for i, label in enumerate(LABELS)},
                          **{feature: float(row[feature]) for feature in FEATURES}}
            output_row["evaluation_set"] = name
            prediction_rows.append(output_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary["provenance"] = {"calibration_csv": str(args.calibration.resolve()),
                             "holdout_csv": str(args.heldout.resolve()),
                             "official_csv": str(args.official.resolve())}
    (args.output_dir / "classifier_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    predictions_path = args.output_dir / "classifier_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(f"Wrote {args.output_dir / 'classifier_summary.json'}")
    print(f"Wrote {predictions_path}")
    print(json.dumps(summary["evaluation"], indent=2))


if __name__ == "__main__":
    main()
