"""Class-wise residual and stratified cross-validation diagnostics for the training-derived set."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np


SIGNALS = ("appearance", "feature", "structural")
THRESHOLDS = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)


def auc_high(positive: np.ndarray, negative: np.ndarray) -> float:
    # Mann-Whitney U with half credit for ties.
    pairwise = positive[:, None] - negative[None, :]
    return float(((pairwise > 0).sum() + 0.5 * (pairwise == 0).sum()) / pairwise.size)


def boot_ci(values_a: np.ndarray, values_b: np.ndarray, *, auc: bool,
            iterations: int, rng: np.random.Generator) -> list[float]:
    estimates = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        a = rng.choice(values_a, len(values_a), replace=True)
        b = rng.choice(values_b, len(values_b), replace=True)
        estimates[i] = auc_high(a, b) if auc else np.mean(a == b)
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def best_config(records: list[dict], weights_grid: list[dict[str, float]]) -> tuple[dict, list[float]]:
    truth = np.array([row["label"] == "CO" for row in records])
    best = None
    for weights in weights_grid:
        score = np.array([sum(float(row[f"residual_{name}"]) * weights[name] for name in SIGNALS)
                          for row in records])
        for direction in ("high", "low"):
            for threshold in THRESHOLDS:
                predicted = score >= threshold if direction == "high" else score <= threshold
                co_acceptance = float(predicted[truth].mean()) if truth.any() else 0.0
                negative_rejection = float((~predicted[~truth]).mean()) if (~truth).any() else 0.0
                accuracy = float((predicted == truth).mean())
                candidate = {"accuracy": accuracy, "co_acceptance": co_acceptance,
                             "negative_rejection": negative_rejection, "threshold": threshold,
                             "direction": direction, "weights": weights}
                key = (accuracy, co_acceptance, negative_rejection, threshold)
                if best is None or key > best[0]:
                    best = (key, candidate, predicted)
    return best[1], best[2].tolist()


def _source_family(image_id: str) -> str:
    match = re.match(r"^(CS|AWA|LUSI|DUTS|HKU|COD10K|background)", image_id, re.IGNORECASE)
    return match.group(1).upper() if match else image_id.split("-")[0].split("_")[0].upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-image", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/usc12k/VOC-USC12K"))
    parser.add_argument("--classes-dir", type=Path, default=Path("data/raw/usc12k_classes/Train_Class"))
    parser.add_argument("--cod-training-images", type=Path, default=Path("data/raw/cod_train/COD-TrainDataset/Imgs"))
    args = parser.parse_args()
    if args.folds < 2 or args.bootstrap < 100:
        parser.error("--folds must be >=2 and --bootstrap >=100")
    with args.per_image.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if {row["label"] for row in records} != {"CO", "BG", "NOCOD"}:
        parser.error("Expected CO, BG, and NOCOD validation records")
    for label in ("CO", "BG", "NOCOD"):
        if sum(row["label"] == label for row in records) < args.folds:
            parser.error(f"Need at least {args.folds} {label} records for stratified folds")
    chosen = json.loads(args.calibration_json.read_text(encoding="utf-8"))["selected"]["combined_best"]
    weights = {"appearance": chosen["appearance_weight"], "feature": chosen["feature_weight"],
               "structural": chosen["structural_weight"]}
    selected_scores = np.array([sum(float(row[f"residual_{name}"]) * weights[name] for name in SIGNALS)
                                for row in records])
    selected_direction = chosen["acceptance_direction"]
    selected_threshold = float(chosen["threshold"])

    rng = np.random.default_rng(args.seed)
    pairwise = {}
    for signal in (*SIGNALS, "selected_combined"):
        values = (selected_scores if signal == "selected_combined" else
                  np.array([float(row[f"residual_{signal}"]) for row in records]))
        for positive, negative in (("CO", "BG"), ("CO", "NOCOD"), ("NOCOD", "BG")):
            a = values[[row["label"] == positive for row in records]]
            b = values[[row["label"] == negative for row in records]]
            high_auc = auc_high(a, b)
            pairwise[f"{signal}:{positive}_vs_{negative}"] = {
                "n_positive": len(a), "n_negative": len(b), "auc_if_high_means_positive": high_auc,
                "auc_if_low_means_positive": 1 - high_auc,
                "bootstrap_95ci_high_direction": boot_ci(a, b, auc=True, iterations=args.bootstrap, rng=rng),
            }

    class_distribution = {}
    for label in ("CO", "BG", "NOCOD"):
        mask = np.array([row["label"] == label for row in records])
        class_distribution[label] = {"n": int(mask.sum())}
        for signal in SIGNALS:
            class_distribution[label][signal] = {
                "mean": float(np.mean([float(r[f"residual_{signal}"]) for r, keep in zip(records, mask) if keep])),
                "median": float(np.median([float(r[f"residual_{signal}"]) for r, keep in zip(records, mask) if keep])),
            }
        values = selected_scores[mask]
        class_distribution[label]["selected_combined"] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)), "q75": float(np.quantile(values, 0.75)),
            "accepted_at_selected_threshold": float(np.mean(values <= selected_threshold if selected_direction == "low" else values >= selected_threshold)),
        }

    # Stratified folds are generated within class; each test fold has equal class counts.
    py_rng = random.Random(args.seed)
    folds: list[list[int]] = [[] for _ in range(args.folds)]
    for label in ("CO", "BG", "NOCOD"):
        indices = [i for i, row in enumerate(records) if row["label"] == label]
        py_rng.shuffle(indices)
        for position, index in enumerate(indices):
            folds[position % args.folds].append(index)
    grid = [{name: units / 4 for name, units in zip(SIGNALS, combo)}
            for combo in __import__("itertools").product(range(5), repeat=3) if sum(combo) == 4]
    out_of_fold = [None] * len(records)
    fold_choices = []
    for fold_number, test_indices in enumerate(folds):
        train_indices = [i for i in range(len(records)) if i not in set(test_indices)]
        train_rows = [records[i] for i in train_indices]
        setting, _ = best_config(train_rows, grid)
        test_rows = [records[i] for i in test_indices]
        for i, row in zip(test_indices, test_rows):
            score = sum(float(row[f"residual_{name}"]) * setting["weights"][name] for name in SIGNALS)
            accepted = score >= setting["threshold"] if setting["direction"] == "high" else score <= setting["threshold"]
            out_of_fold[i] = bool(accepted)
        fold_choices.append({"fold": fold_number + 1, **setting})
    truth = np.array([row["label"] == "CO" for row in records])
    oof = np.array(out_of_fold, dtype=bool)
    cv_accuracy = float(np.mean(oof == truth))
    cv_by_class = {label: float(np.mean(oof[[r["label"] == label for r in records]]))
                   for label in ("CO", "BG", "NOCOD")}
    correct = oof == truth
    cv_accuracy_ci = boot_ci(correct, np.ones_like(correct, dtype=bool), auc=False,
                             iterations=args.bootstrap, rng=rng)
    direction_counts = dict(Counter(row["direction"] for row in fold_choices))
    feature_weight_counts = dict(Counter(str(row["weights"]["feature"]) for row in fold_choices))

    # Check source composition over all eligible training-derived pools, not only
    # the small random sample. Class membership and source family may be confounded.
    from openpyxl import load_workbook
    dataset_root, classes_dir = args.dataset_root.resolve(), args.classes_dir.resolve()
    train_ids = set((dataset_root / "ImageSets/Segmentation/train.txt").read_text(encoding="utf-8").split())
    scene_classes = {}
    for scene, filename in (("A", "SceneA2.xlsx"), ("B", "SceneB2.xlsx"), ("C", "SceneC2.xlsx")):
        sheet = load_workbook(classes_dir / filename, read_only=True, data_only=True).active
        scene_classes[scene] = {str(row[0]).strip() for row in sheet.iter_rows(min_row=2, values_only=True) if row[0]}
    cod_stems = {path.stem for path in args.cod_training_images.rglob("*") if path.is_file()}
    source_pools = {"NOCOD": scene_classes["A"] & train_ids,
                    "CO": (scene_classes["C"] & train_ids) - cod_stems,
                    "BG": train_ids - set.union(*scene_classes.values())}
    source_counts = {label: dict(Counter(_source_family(image_id) for image_id in ids))
                     for label, ids in source_pools.items()}
    source_sets = {label: set(counts) for label, counts in source_counts.items()}
    source_overlaps = {f"{a}_and_{b}": sorted(source_sets[a] & source_sets[b])
                       for a, b in (("CO", "BG"), ("CO", "NOCOD"), ("BG", "NOCOD"))}
    sampled_source_counts = {label: dict(Counter(_source_family(row["image_id"])
                                                   for row in records if row["label"] == label))
                             for label in ("CO", "BG", "NOCOD")}

    report = {"n": len(records), "class_counts": {label: sum(r["label"] == label for r in records)
                                                    for label in ("CO", "BG", "NOCOD")},
              "selected_full_sample": {**chosen, "score_class_distribution": class_distribution},
              "pairwise_auc": pairwise,
              "eligible_training_source_family_counts": source_counts,
              "sampled_source_family_counts": sampled_source_counts,
              "source_family_overlaps_between_classes": source_overlaps,
              "five_fold_stratified_oof": {"accuracy": cv_accuracy, "bootstrap_95ci": cv_accuracy_ci,
                                           "CO_acceptance_rate": cv_by_class["CO"],
                                           "BG_false_accept_rate": cv_by_class["BG"],
                                           "NOCOD_false_accept_rate": cv_by_class["NOCOD"],
                                           "selected_direction_counts": direction_counts,
                                           "feature_weight_counts": feature_weight_counts,
                                           "fold_settings": fold_choices},
              "caveat": "Fold-to-fold results remain exploratory because there are only 20 samples per class; folds share training data and the score grid is small. Random folds preserve each class's source mix, so they do not establish cross-source generalization. Source families may be confounded with class; no official test images were processed."}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "class_diagnostics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Validation class-structure diagnostics", "",
             f"Training-derived sample: n={len(records)}, with 20 per class. Official test images were not processed.", "",
             "## Residual score distributions", "",
             "Selected combined score accepts CO on the low side at threshold 0.10. Values below are means/medians by class; quartiles for the selected composite show overlap.", "",
             "| Class | Appearance mean | Feature mean | Structural mean | Composite mean [Q1, Q3] | Accepted at selected rule |", "|---|---:|---:|---:|---:|---:|"]
    for label, item in class_distribution.items():
        c = item["selected_combined"]
        lines.append(f"| {label} | {item['appearance']['mean']:.3f} | {item['feature']['mean']:.3f} | {item['structural']['mean']:.3f} | {c['mean']:.3f} [{c['q25']:.3f}, {c['q75']:.3f}] | {c['accepted_at_selected_threshold']:.2f} |")
    lines += ["", "## Pairwise AUROC", "", "AUROC values are listed in both orientations: whether larger or smaller scores rank the first class as positive. Intervals use class-stratified bootstrap resampling.", "",
              "| Signal | Comparison | AUC if high is positive (95% CI) | AUC if low is positive |", "|---|---|---:|---:|"]
    for key, item in pairwise.items():
        signal, comparison = key.split(":")
        lo, hi = item["bootstrap_95ci_high_direction"]
        lines.append(f"| {signal} | {comparison} | {item['auc_if_high_means_positive']:.3f} ({lo:.3f}–{hi:.3f}) | {item['auc_if_low_means_positive']:.3f} |")
    cv = report["five_fold_stratified_oof"]
    lines += ["", "## Cross-validated selection stability", "",
              f"Five-fold stratified out-of-fold accuracy was {cv['accuracy']:.3f} (sample-bootstrap 95% interval {cv_accuracy_ci[0]:.3f}–{cv_accuracy_ci[1]:.3f}). OOF CO acceptance was {cv['CO_acceptance_rate']:.3f}; BG false acceptance {cv['BG_false_accept_rate']:.3f}; NOCOD false acceptance {cv['NOCOD_false_accept_rate']:.3f}.",
              f"Directions selected across folds: `{direction_counts}`. Feature weights selected across folds: `{feature_weight_counts}`.", "",
              "This cross-validation repeats threshold/weight/direction selection inside each training fold; predictions come only from each held-out fold. However, random folds preserve each class's source mix. They do not test transfer to a new source dataset.", "",
              "## Dataset-source confounding", "",
              "| Training-derived label pool | Source-family image counts after exact SINet training-stem exclusion |", "|---|---|"]
    for label in ("CO", "BG", "NOCOD"):
        lines.append(f"| {label} | " + ", ".join(f"{source}: {count}" for source, count in sorted(source_counts[label].items())) + " |")
    lines += ["", f"Source-family overlaps between the three eligible class pools: `{source_overlaps}`.",
              "If these overlaps are empty, label and source family are fully confounded in this USC12K training construction. A detector can score well by recognizing source-dataset style instead of camouflage semantics; source-held-out evaluation across all three labels cannot be created from these pools as currently annotated. This is a major limitation to resolve before a long benchmark run.", ""]
    (output / "CLASS_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output / 'CLASS_DIAGNOSTICS.md'} and {output / 'class_diagnostics.json'}")
    print(f"selected setting: CO/BG/NOCOD acceptance={class_distribution['CO']['selected_combined']['accepted_at_selected_threshold']:.2f}/{class_distribution['BG']['selected_combined']['accepted_at_selected_threshold']:.2f}/{class_distribution['NOCOD']['selected_combined']['accepted_at_selected_threshold']:.2f}")
    print(f"5-fold OOF accuracy={cv_accuracy:.3f}; direction stability={direction_counts}; feature-weight stability={feature_weight_counts}")


if __name__ == "__main__":
    main()
