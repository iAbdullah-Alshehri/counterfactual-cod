"""Post-hoc baseline, mask, and bootstrap analysis from saved USC12K outputs.

This script does not run SINet-V2, LaMa, or feature extraction. It reads the
saved per-image predictions, original candidate masks/ground truth, and the
full-precision residual cache used by the completed evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cod_recon.data import read_manifest, read_mask
from cod_recon.metrics import segmentation_metrics


LABELS = ("BG", "CO", "NOCOD")
METRICS = ("S_alpha", "weighted_F_beta", "E_phi", "MAE")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def confusion_and_metrics(rows: list[dict[str, str]], predictions: list[str]) -> tuple[np.ndarray, dict]:
    if len(rows) != len(predictions):
        raise ValueError("Prediction count does not match row count")
    index = {label: i for i, label in enumerate(LABELS)}
    matrix = np.zeros((3, 3), dtype=np.int64)
    for row, prediction in zip(rows, predictions):
        label = row["label"].upper()
        if label not in index or prediction not in index:
            raise ValueError(f"Unexpected label/prediction: {label!r}/{prediction!r}")
        matrix[index[label], index[prediction]] += 1
    tp = np.diag(matrix).astype(float)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(3), where=(2 * tp + fp + fn) > 0)
    recalls = np.divide(tp, matrix.sum(axis=1), out=np.zeros(3), where=matrix.sum(axis=1) > 0)
    negative_n = int(matrix[0].sum() + matrix[2].sum())
    negative_false_accepts = int(matrix[0, 1] + matrix[2, 1])
    return matrix, {
        "n": int(matrix.sum()),
        "three_way_accuracy": float(tp.sum() / max(1, matrix.sum())),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recalls.mean()),
        "recall_by_class": {label: float(recalls[i]) for i, label in enumerate(LABELS)},
        "f1_by_class": {label: float(f1[i]) for i, label in enumerate(LABELS)},
        "negative_false_accept_rate": negative_false_accepts / max(1, negative_n),
        "negative_false_accept_rate_by_class": {
            "BG": float(matrix[0, 1] / max(1, matrix[0].sum())),
            "NOCOD": float(matrix[2, 1] / max(1, matrix[2].sum())),
        },
        "co_acceptance_rate": float(matrix[1, 1] / max(1, matrix[1].sum())),
        "confusion_order": list(LABELS),
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
    }


def bootstrap_classification(rows: list[dict[str, str]], predictions: list[str],
                             replicates: int, seed: int) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row["label"].upper()].append(i)
    if set(grouped) != set(LABELS):
        raise ValueError("Bootstrap requires all three classes")
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        ids = np.concatenate([rng.choice(grouped[label], size=len(grouped[label]), replace=True)
                              for label in LABELS])
        sample_rows = [rows[int(i)] for i in ids]
        sample_predictions = [predictions[int(i)] for i in ids]
        _, result = confusion_and_metrics(sample_rows, sample_predictions)
        for name in ("three_way_accuracy", "macro_f1", "balanced_accuracy",
                     "negative_false_accept_rate", "co_acceptance_rate"):
            values[name].append(float(result[name]))
        for label in LABELS:
            values[f"recall_{label}"].append(float(result["recall_by_class"][label]))
        for label in ("BG", "NOCOD"):
            values[f"negative_false_accept_rate_{label}"].append(
                float(result["negative_false_accept_rate_by_class"][label]))
    return {name: {"lower_95": float(np.quantile(samples, 0.025)),
                   "upper_95": float(np.quantile(samples, 0.975))}
            for name, samples in values.items()}


def ci_mean(values: list[float], replicates: int, seed: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"n": 0, "mean": float("nan"), "lower_95": float("nan"), "upper_95": float("nan")}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[indices].mean(axis=1)
    return {"n": int(len(array)), "mean": float(array.mean()),
            "lower_95": float(np.quantile(means, 0.025)),
            "upper_95": float(np.quantile(means, 0.975))}


def cache_path_for(sample, config: dict, cache_root: Path) -> Path:
    image_stat = sample.image_path.stat()
    mask_stat = sample.mask_path.stat()
    key = "\0".join((str(sample.image_path), str(image_stat.st_size), str(image_stat.st_mtime_ns),
                      str(sample.mask_path), str(mask_stat.st_size), str(mask_stat.st_mtime_ns),
                      config["backend"], str(config["dilation"]), config["inpainting_model"],
                      config["feature_extractor"]))
    return cache_root / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".npz")


def mask_analysis(manifest_path: Path, run_root: Path, verify_rows: list[dict[str, str]],
                  refine_rows: list[dict[str, str]], replicates: int, seed: int) -> dict:
    manifest_samples = read_manifest(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        source_scene_by_id = {
            (row.get("image_id") or Path(row["image"]).stem): row.get("source_scene", "")
            for row in csv.DictReader(stream)
        }
    verify_by_id = {row["image_id"]: row for row in verify_rows}
    refine_by_id = {row["image_id"]: row for row in refine_rows}
    if len(verify_by_id) != len(verify_rows) or len(refine_by_id) != len(refine_rows):
        raise ValueError("Saved per-image CSV contains duplicate IDs")
    if set(verify_by_id) != set(refine_by_id) or set(verify_by_id) != {s.image_id for s in manifest_samples}:
        raise ValueError("Manifest and verify/refine result IDs do not match")
    config_path = run_root / "feature_band_refine" / "resume_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing cache-key configuration: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache_root = run_root / "_residual_cache"
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Residual cache not found: {cache_root}")

    groups: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in METRICS} for name in ("candidate_baseline", "verified", "refined")
    }
    accepted_groups = {name: {metric: [] for metric in METRICS} for name in groups}
    negative_mask_area = {name: {label: [] for label in ("BG", "NOCOD")}
                          for name in ("candidate_baseline", "verified", "refined")}
    row_output: list[dict] = []

    for sample in manifest_samples:
        vrow, rrow = verify_by_id[sample.image_id], refine_by_id[sample.image_id]
        image_shape = None
        # read_mask needs the source image dimensions; PIL reads only the header/pixels, not model inference.
        from PIL import Image
        with Image.open(sample.image_path) as image:
            width, height = image.size
        image_shape = (height, width)
        candidate = read_mask(sample.mask_path, image_shape)
        verify_prediction = candidate.copy() if vrow["predicted_label"] == "CO" else np.zeros_like(candidate)
        cache_file = cache_path_for(sample, config, cache_root)
        if not cache_file.is_file():
            raise FileNotFoundError(f"Missing saved residual cache for {sample.image_id}: {cache_file}")
        with np.load(cache_file, allow_pickle=False) as cached:
            if "feature" not in cached.files or cached["feature"].dtype != np.float32:
                raise ValueError(f"Expected a full-precision feature map in {cache_file}")
            feature_map = cached["feature"].astype(np.float32, copy=False)
        if feature_map.shape != candidate.shape:
            raise ValueError(f"Residual/mask shape mismatch for {sample.image_id}")
        support = candidate >= 0.5
        spatial = np.zeros_like(candidate)
        if support.any():
            values = feature_map[support]
            lo, hi = np.percentile(values, [5, 95])
            if hi - lo > 1e-8:
                spatial[support] = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
            else:
                spatial[support] = 1.0 if float(values.mean()) >= 0.5 else 0.0
        refined_prediction = candidate * spatial
        if rrow["predicted_label"] != "CO":
            refined_prediction = np.zeros_like(refined_prediction)

        predictions = {"candidate_baseline": candidate,
                       "verified": verify_prediction,
                       "refined": refined_prediction}
        if sample.label == "CO":
            if sample.ground_truth_path is None:
                raise ValueError(f"CO row lacks ground truth: {sample.image_id}")
            target = read_mask(sample.ground_truth_path, image_shape)
            for name, prediction in predictions.items():
                result = segmentation_metrics(prediction, target)
                for metric in METRICS:
                    groups[name][metric].append(float(result[metric]))
                    if vrow["predicted_label"] == "CO":
                        accepted_groups[name][metric].append(float(result[metric]))
        else:
            for name, prediction in predictions.items():
                negative_mask_area[name][sample.label].append(float(prediction.mean()))

        row_output.append({"image_id": sample.image_id, "label": sample.label,
                           "source_scene": source_scene_by_id.get(sample.image_id, ""),
                           "candidate_area_fraction": float(candidate.mean()),
                           "verified_area_fraction": float(verify_prediction.mean()),
                           "refined_area_fraction": float(refined_prediction.mean()),
                           "predicted_label": vrow["predicted_label"]})

    segmentation = {}
    for name in groups:
        segmentation[name] = {
            "all_CO": {metric: ci_mean(values, replicates, seed + i)
                       for i, (metric, values) in enumerate(groups[name].items())},
            "CO_accepted_by_verifier": {metric: ci_mean(values, replicates, seed + 100 + i)
                                        for i, (metric, values) in enumerate(accepted_groups[name].items())},
            "negative_predicted_mask_area_fraction": {
                label: ci_mean(values, replicates, seed + 200 + i)
                for i, (label, values) in enumerate(negative_mask_area[name].items())},
        }
    return {"segmentation_and_mask_area": segmentation, "per_image_mask_rows": row_output,
            "note": "Candidate baseline uses raw SINet-V2 masks. Verified masks retain the candidate only when the saved rule predicts CO. Refined masks are reconstructed from cached full-precision feature maps using the evaluator's 5th/95th percentile normalization. Segmentation metrics are internal project implementations; verify against benchmark code before publication."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path,
                        default=Path("data/experiments/usc12k/full/feature_band_lama_full"))
    parser.add_argument("--manifest", type=Path, default=Path("data/experiments/usc12k/full/val_manifest.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/experiments/usc12k/full/feature_band_lama_full/posthoc_analysis"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("Use at least 100 bootstrap replicates")
    root = args.project_root.resolve()
    run_root = (root / args.run_root).resolve() if not args.run_root.is_absolute() else args.run_root.resolve()
    manifest = (root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve()
    output = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    verify_path = run_root / "feature_band_verify" / "per_image.csv"
    refine_path = run_root / "feature_band_refine" / "per_image.csv"
    verify_rows, refine_rows = read_rows(verify_path), read_rows(refine_path)
    if not verify_rows or len(verify_rows) != len(refine_rows):
        raise ValueError("Expected complete, matching verify and refine per-image CSV files")
    verify_predictions = [row["predicted_label"] for row in verify_rows]
    verify_matrix, verify_stats = confusion_and_metrics(verify_rows, verify_predictions)
    refine_predictions = [row["predicted_label"] for row in refine_rows]
    refine_matrix, refine_stats = confusion_and_metrics(refine_rows, refine_predictions)
    if not np.array_equal(verify_matrix, refine_matrix):
        raise ValueError("Verify and refine runs have different class decisions; inspect run provenance")
    verify_stats["bootstrap_95_ci"] = bootstrap_classification(
        verify_rows, verify_predictions, args.bootstrap_replicates, args.seed)
    mask_results = mask_analysis(manifest, run_root, verify_rows, refine_rows,
                                 args.bootstrap_replicates, args.seed + 1000)
    per_image_path = output / "posthoc_per_image.csv"
    output.mkdir(parents=True, exist_ok=True)
    with per_image_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mask_results["per_image_mask_rows"][0]))
        writer.writeheader()
        writer.writerows(mask_results["per_image_mask_rows"])
    summary = {
        "analysis_type": "posthoc_descriptive_analysis_of_previously_inspected_official_split",
        "n": len(verify_rows), "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "classification": {"feature_band_verify": verify_stats, "feature_band_refine": refine_stats},
        "mask_analysis": {key: value for key, value in mask_results.items() if key != "per_image_mask_rows"},
        "provenance": {"manifest": str(manifest), "verify_per_image": str(verify_path),
                       "refine_per_image": str(refine_path), "cache_dir": str(run_root / "_residual_cache")},
        "limitation": "The official split has already been evaluated and inspected. These estimates are descriptive and must not be used to tune a new model or presented as a fresh blind test. Bootstrap intervals resample images within their observed class; they do not measure dataset/source shift.",
    }
    (output / "posthoc_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                                 encoding="utf-8")
    print(f"Wrote {output / 'posthoc_summary.json'}")
    print(f"Wrote {per_image_path}")
    print(json.dumps(verify_stats, indent=2))


if __name__ == "__main__":
    main()
