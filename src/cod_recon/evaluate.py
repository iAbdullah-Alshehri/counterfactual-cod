"""Run counterfactual residual verification over a manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .data import Sample, read_manifest, read_mask, read_rgb
from .metrics import segmentation_metrics
from .reconstruction import LamaInpainting, inpaint_opencv
from .residuals import ResNet18FeatureDistance, combine_residuals, residual_maps


def _progress_fields() -> list[str]:
    return ["image_id", "label", "split", "candidate_area_fraction", "residual_appearance",
            "residual_feature", "residual_structural", "residual_combined", "verification_score",
            "ablation", "decision_rule", "accepted", "acceptance_direction", "predicted_label",
            "low_cut", "high_cut", "predicted_mask_area_fraction", "runtime_seconds",
            "inpainting_runtime_seconds", "cache_hit", "S_alpha", "weighted_F_beta", "E_phi", "MAE"]


def evaluate(manifest: str | Path, output_dir: str | Path, *, threshold: float = 0.5,
             weights: dict[str, float] | None = None, backend: str = "opencv",
             refine: bool = False, save_previews: bool = False,
             ablation: str = "combined", inpainting_model: str | Path | None = None,
             dilation: int = 5, cache_dir: str | Path | None = None,
             feature_weights: str | Path | None = None,
             acceptance_direction: str = "high", decision_rule: str = "feature_band",
             low_cut: float = 0.185, high_cut: float = 0.285,
             resume: bool = False) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    if backend not in {"opencv", "lama"}:
        raise ValueError("backend must be 'lama' or the classical baseline 'opencv'")
    if dilation < 0:
        raise ValueError("dilation must be nonnegative")
    if acceptance_direction not in {"high", "low"}:
        raise ValueError("acceptance_direction must be 'high' or 'low'")
    if decision_rule not in {"feature_band", "binary_threshold"}:
        raise ValueError("decision_rule must be 'feature_band' or 'binary_threshold'")
    if not (0 <= low_cut < high_cut <= 1):
        raise ValueError("feature-band cuts must satisfy 0 <= low_cut < high_cut <= 1")
    if ablation not in {"baseline", "appearance", "feature", "structural", "combined"}:
        raise ValueError("ablation must be baseline, appearance, feature, structural, or combined")
    if decision_rule == "feature_band":
        feature_only = {"appearance": 0.0, "feature": 1.0, "structural": 0.0}
        if weights is not None and any(float(weights.get(name, 0.0)) != value
                                       for name, value in feature_only.items()):
            raise ValueError("feature_band uses fixed feature-only weights: appearance=0, feature=1, structural=0")
        weights = feature_only
    else:
        weights = weights or {"appearance": 1 / 3, "feature": 1 / 3, "structural": 1 / 3}
    samples = read_manifest(manifest)
    image_ids = [sample.image_id for sample in samples]
    manifest_ids = set(image_ids)
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Manifest image_id values must be unique for reliable evaluation/resume")
    if decision_rule == "feature_band" and {sample.label for sample in samples} != {"BG", "CO", "NOCOD"}:
        raise ValueError("feature_band evaluation requires BG, CO, and NOCOD samples in the manifest")
    lama = None
    if backend == "lama":
        if inpainting_model is None or not Path(inpainting_model).is_file():
            raise FileNotFoundError("LaMa ONNX model is required; run tools/download_data.py --datasets lama_inpainting")
        lama = LamaInpainting(str(Path(inpainting_model).resolve()))
    feature_extractor = ResNet18FeatureDistance(feature_weights)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    preview_dir = output / "previews"
    if save_previews:
        preview_dir.mkdir(exist_ok=True)

    records: list[dict] = []
    cache_root = Path(cache_dir).expanduser().resolve() if cache_dir else None
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)
    model_stamp = ""
    if backend == "lama" and inpainting_model is not None:
        model_stat = Path(inpainting_model).resolve().stat()
        model_stamp = f"{Path(inpainting_model).resolve()}:{model_stat.st_size}:{model_stat.st_mtime_ns}"
    cache_hits = 0
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "per_image.progress.csv"
    config_path = output / "resume_config.json"
    config_hash = hashlib.sha256(Path(manifest).expanduser().resolve().read_bytes()).hexdigest()
    input_digest = hashlib.sha256()
    for sample in samples:
        for path in (sample.image_path, sample.mask_path, sample.ground_truth_path):
            if path is None:
                input_digest.update(b"<none>\0")
                continue
            stat = path.stat()
            input_digest.update(f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode("utf-8"))
    run_config = {"manifest_path": str(Path(manifest).expanduser().resolve()),
                  "manifest_sha256": config_hash, "inputs_sha256": input_digest.hexdigest(),
                  "n": len(samples), "backend": backend, "inpainting_model": model_stamp,
                  "feature_extractor": feature_extractor.cache_signature, "dilation": dilation,
                  "decision_rule": decision_rule, "low_cut": low_cut, "high_cut": high_cut,
                  "threshold": threshold, "weights": weights, "acceptance_direction": acceptance_direction,
                  "ablation": ablation, "refine": refine, "save_previews": save_previews}
    existing_progress = progress_path.is_file() or config_path.is_file()
    if resume and existing_progress:
        if not config_path.is_file():
            raise ValueError("Cannot resume: run configuration sidecar is missing")
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config != run_config:
            raise ValueError("Cannot resume: manifest, inputs, model, or evaluation settings differ from the saved run")
        if progress_path.is_file():
            with progress_path.open(newline="", encoding="utf-8") as stream:
                loaded_rows = list(csv.DictReader(stream))
        else:
            loaded_rows = []
            with progress_path.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=_progress_fields()).writeheader()
        float_fields = {"candidate_area_fraction", "residual_appearance", "residual_feature", "residual_structural",
                        "residual_combined", "verification_score", "predicted_mask_area_fraction", "runtime_seconds",
                        "inpainting_runtime_seconds", "low_cut", "high_cut", "S_alpha", "weighted_F_beta", "E_phi", "MAE"}
        bool_fields = {"accepted", "cache_hit"}
        existing_records: dict[str, dict] = {}
        for loaded in loaded_rows:
            row = dict(loaded)
            for key in float_fields:
                if row.get(key, "") != "":
                    row[key] = float(row[key])
                else:
                    row.pop(key, None)
            for key in bool_fields:
                if key in row:
                    row[key] = row[key].strip().lower() == "true"
            image_id = row.get("image_id", "")
            if image_id not in manifest_ids or image_id in existing_records:
                raise ValueError("Cannot resume: progress file contains an unknown or duplicate image_id")
            existing_records[image_id] = row
        records = list(existing_records.values())
    elif existing_progress:
        raise ValueError(f"Progress already exists at {progress_path}; pass --resume to continue it")
    else:
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
        records = []
        with progress_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=_progress_fields())
            writer.writeheader()
            stream.flush()
    resumed_ids = {str(row["image_id"]) for row in records}
    progress_stream = progress_path.open("a", newline="", encoding="utf-8")
    progress_writer = csv.DictWriter(progress_stream, fieldnames=_progress_fields())
    start_all = time.perf_counter()
    try:
        for sample in samples:
            if sample.image_id in resumed_ids:
                continue
            start = time.perf_counter()
            image = read_rgb(sample.image_path)
            candidate = read_mask(sample.mask_path, image.shape[:2])
            cache_file = None
            if cache_root:
                image_stat, mask_stat = sample.image_path.stat(), sample.mask_path.stat()
                key = "\0".join((str(sample.image_path), str(image_stat.st_size), str(image_stat.st_mtime_ns),
                                  str(sample.mask_path), str(mask_stat.st_size), str(mask_stat.st_mtime_ns),
                                  backend, str(dilation), model_stamp, feature_extractor.cache_signature))
                cache_file = cache_root / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".npz")
            reconstruction_seconds = 0.0
            cache_hit = False
            residual_cache_hit = False
            maps = None
            if cache_file is not None and cache_file.is_file():
                with np.load(cache_file, allow_pickle=False) as cached:
                    background = cached["background"]
                    reconstruction_seconds = float(cached["reconstruction_seconds"])
                    if all(name in cached.files for name in ("appearance", "feature", "structural")) and all(
                            cached[name].dtype == np.float32 for name in ("appearance", "feature", "structural")):
                        maps = {name: cached[name].astype(np.float32, copy=False)
                                for name in ("appearance", "feature", "structural")}
                        residual_cache_hit = True
                cache_hits += 1
                cache_hit = True
            else:
                if backend == "lama" and lama is None:
                    lama = LamaInpainting(str(Path(inpainting_model).resolve()))
                reconstruction_start = time.perf_counter()
                background = (lama(image, candidate, dilation=dilation) if lama is not None
                              else inpaint_opencv(image, candidate, dilation=dilation))
                reconstruction_seconds = time.perf_counter() - reconstruction_start
            # Reuse only full-precision maps from a cache keyed by the feature
            # extractor; older half-precision entries are recomputed for accuracy.
            if maps is None:
                maps = residual_maps(image, background, candidate, feature_extractor)
            if cache_file is not None and not residual_cache_hit:
                np.savez_compressed(cache_file, background=background,
                                    reconstruction_seconds=np.float64(reconstruction_seconds),
                                    **{name: value.astype(np.float32) for name, value in maps.items()})
            # The primary feature-band mode must bypass the abandoned combined-score
            # threshold path; its decision signal is exactly the learned feature map.
            combined = (maps["feature"] if decision_rule == "feature_band"
                        else combine_residuals(maps, weights))
            support = candidate >= 0.5
            score_map = (maps["feature"] if decision_rule == "feature_band"
                         else (combined if ablation == "combined" else maps.get(ablation, combined)))
            evidence = float(score_map[support].mean()) if support.any() else 0.0
            if decision_rule == "feature_band":
                predicted_label = ("BG" if evidence < low_cut else
                                   "NOCOD" if evidence > high_cut else "CO")
                # An empty candidate has no foreground support and cannot be accepted as CO.
                if not support.any():
                    predicted_label = "BG"
                accepted = bool(support.any() and predicted_label == "CO")
            else:
                passes = evidence >= threshold if acceptance_direction == "high" else evidence <= threshold
                accepted = bool(support.any() and (ablation == "baseline" or passes))
                predicted_label = "CO" if accepted else "NEGATIVE"
            if refine and support.any():
                values = score_map[support]
                lo, hi = np.percentile(values, [5, 95])
                spatial = np.zeros_like(combined)
                if hi - lo > 1e-8:
                    spatial_values = np.clip((values - lo) / (hi - lo), 0, 1)
                    if acceptance_direction == "low":
                        spatial_values = 1.0 - spatial_values
                    spatial[support] = spatial_values
                else:
                    mean_passes = (float(values.mean()) >= threshold if acceptance_direction == "high"
                                   else float(values.mean()) <= threshold)
                    spatial[support] = 1.0 if mean_passes else 0.0
                prediction = candidate * spatial
            else:
                prediction = candidate.copy()
            if not accepted:
                prediction = np.zeros_like(prediction)
            row: dict[str, str | float | int | bool] = {
                "image_id": sample.image_id, "label": sample.label, "split": sample.split,
                "candidate_area_fraction": float(candidate.mean()), "residual_appearance": float(maps["appearance"][support].mean()) if support.any() else 0.0,
                "residual_feature": float(maps["feature"][support].mean()) if support.any() else 0.0,
                "residual_structural": float(maps["structural"][support].mean()) if support.any() else 0.0,
                "residual_combined": float(combined[support].mean()) if support.any() else 0.0,
                "verification_score": evidence,
                "ablation": "feature" if decision_rule == "feature_band" else ablation,
                "decision_rule": decision_rule,
                "accepted": accepted,
                "acceptance_direction": acceptance_direction if decision_rule == "binary_threshold" else "band",
                "predicted_label": predicted_label,
                "predicted_mask_area_fraction": float(prediction.mean()),
                "runtime_seconds": time.perf_counter() - start,
                "inpainting_runtime_seconds": reconstruction_seconds,
                "cache_hit": cache_hit,
            }
            if decision_rule == "feature_band":
                row.update({"low_cut": low_cut, "high_cut": high_cut,
                            "predicted_label": predicted_label})
            if sample.ground_truth_path and sample.label == "CO":
                gt = read_mask(sample.ground_truth_path, image.shape[:2])
                row.update(segmentation_metrics(prediction, gt))
            records.append(row)
            if save_previews:
                stem = Path(sample.image_id).name
                Image.fromarray(image).save(preview_dir / f"{stem}_original.png")
                Image.fromarray(np.uint8(np.clip(candidate, 0, 1) * 255)).save(preview_dir / f"{stem}_candidate.png")
                Image.fromarray(background).save(preview_dir / f"{stem}_background.png")
                Image.fromarray(np.uint8(np.clip(combined, 0, 1) * 255)).save(preview_dir / f"{stem}_residual.png")
                Image.fromarray(np.uint8(np.clip(prediction, 0, 1) * 255)).save(preview_dir / f"{stem}_final.png")
                if sample.ground_truth_path and sample.label == "CO":
                    Image.fromarray(np.uint8(np.clip(gt, 0, 1) * 255)).save(preview_dir / f"{stem}_ground_truth.png")
            progress_writer.writerow(row)
            progress_stream.flush()
    finally:
        progress_stream.close()

    records_by_id = {str(row["image_id"]): row for row in records}
    records = [records_by_id[sample.image_id] for sample in samples]

    effective_ablation = "feature_band" if decision_rule == "feature_band" else ablation
    summary = _summarize(records, time.perf_counter() - start_all, weights, threshold,
                         refine, cache_hits, effective_ablation, backend, dilation, acceptance_direction,
                         decision_rule, low_cut, high_cut)
    summary["resumed_images"] = len(resumed_ids)
    summary["progress_checkpoint"] = str(progress_path)
    _write_csv(output / "per_image.csv", records)
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def _summarize(records: list[dict], elapsed: float, weights: dict[str, float], threshold: float,
               refine: bool, cache_hits: int, ablation: str, backend: str, dilation: int,
               acceptance_direction: str = "high", decision_rule: str = "binary_threshold",
               low_cut: float = 0.185, high_cut: float = 0.285) -> dict:
    by_label: dict[str, list[dict]] = defaultdict(list)
    by_split: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_label[str(row["label"])].append(row)
        by_split[str(row["split"])].append(row)

    def means(rows: list[dict], keys: tuple[str, ...]) -> dict[str, float | int]:
        result: dict[str, float | int] = {"n": len(rows)}
        for key in keys:
            vals = [float(row[key]) for row in rows if key in row]
            if vals:
                result[key] = float(np.mean(vals))
        return result

    labels = [str(row["label"]) for row in records]
    predicted = [str(row["predicted_label"]) for row in records]
    binary_truth = ["CO" if label == "CO" else "NEGATIVE" for label in labels]
    binary_predicted = ["CO" if label == "CO" else "NEGATIVE" for label in predicted]
    binary_accuracy = float(np.mean([a == b for a, b in zip(binary_truth, binary_predicted)]))
    confusion_order = ("BG", "CO", "NOCOD")
    confusion_matrix = None
    three_way_accuracy = None
    macro_f1 = None
    if decision_rule == "feature_band":
        indices = {label: index for index, label in enumerate(confusion_order)}
        confusion_matrix = np.zeros((3, 3), dtype=int)
        for row in records:
            if row["label"] in indices and row["predicted_label"] in indices:
                confusion_matrix[indices[row["label"]], indices[row["predicted_label"]]] += 1
        three_way_accuracy = float(np.trace(confusion_matrix) / confusion_matrix.sum())
        tp = np.diag(confusion_matrix).astype(float)
        fp = confusion_matrix.sum(axis=0) - tp
        fn = confusion_matrix.sum(axis=1) - tp
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(3), where=(2 * tp + fp + fn) > 0)
        macro_f1 = float(f1.mean())
    negatives = [row for row in records if row["label"] in {"BG", "NOCOD"}]
    co = [row for row in records if row["label"] == "CO"]
    fp_scene_rate = float(np.mean([bool(r["accepted"]) for r in negatives])) if negatives else None
    per_label = {
        label: means(rows, ("accepted", "predicted_mask_area_fraction", "verification_score", "residual_combined", "S_alpha", "weighted_F_beta", "E_phi", "MAE"))
        for label, rows in sorted(by_label.items())
    }
    per_split = {split: means(rows, ("accepted", "predicted_mask_area_fraction", "verification_score", "residual_combined", "S_alpha", "weighted_F_beta", "E_phi", "MAE"))
                 for split, rows in sorted(by_split.items())}
    return {
        "n": len(records), "co_vs_negative_accuracy": binary_accuracy,
        "three_way_accuracy": three_way_accuracy, "macro_f1": macro_f1,
        "confusion_order": list(confusion_order) if confusion_matrix is not None else None,
        "confusion_matrix_true_rows_predicted_columns": confusion_matrix.tolist() if confusion_matrix is not None else None,
        "negative_scene_false_accept_rate": fp_scene_rate,
        "negative_scene_rejection_accuracy": 1 - fp_scene_rate if fp_scene_rate is not None else None,
        "co_acceptance_rate": float(np.mean([bool(r["accepted"]) for r in co])) if co else None,
        "mean_runtime_seconds_per_image": float(np.mean([r["runtime_seconds"] for r in records])),
        "mean_inpainting_runtime_seconds_per_image": float(np.mean([r["inpainting_runtime_seconds"] for r in records])),
        "cache_hits": cache_hits,
        "total_runtime_seconds": float(elapsed),
        "threshold": threshold if decision_rule == "binary_threshold" else None,
        "weights": weights,
        "acceptance_direction": acceptance_direction if decision_rule == "binary_threshold" else None,
        "decision_rule": decision_rule,
        "low_cut": low_cut if decision_rule == "feature_band" else None,
        "high_cut": high_cut if decision_rule == "feature_band" else None,
        "mask_refinement_enabled": refine, "ablation": ablation, "inpainting_backend": backend,
        "inpainting_dilation_pixels": dilation, "metrics_by_label": per_label, "metrics_by_split": per_split,
        "note": ("Feature-only three-way band: BG below low_cut, CO within the inclusive band, NOCOD above high_cut."
                 if decision_rule == "feature_band" else
                 "Legacy binary CO-vs-negative threshold mode. It does not distinguish BG from NOCOD. Evidence scores are not calibrated probabilities."),
    }


def _write_csv(path: Path, records: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
