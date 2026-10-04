"""Cross-check full USC12K CO segmentation metrics with PySODMetrics.

Reconstructs verify/refine masks from the saved candidate masks and full precision
feature residual cache. It does not run LaMa, SINet-V2, or feature extraction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from py_sod_metrics import Emeasure, MAE, Smeasure, WeightedFmeasure

from cod_recon.data import read_manifest, read_mask


def rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def cache_path_for(sample, config: dict, cache_root: Path) -> Path:
    image_stat, mask_stat = sample.image_path.stat(), sample.mask_path.stat()
    key = "\0".join((str(sample.image_path), str(image_stat.st_size), str(image_stat.st_mtime_ns),
                      str(sample.mask_path), str(mask_stat.st_size), str(mask_stat.st_mtime_ns),
                      config["backend"], str(config["dilation"]), config["inpainting_model"],
                      config["feature_extractor"]))
    return cache_root / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".npz")


def create_metrics():
    return {"S_alpha": Smeasure(), "weighted_F_beta": WeightedFmeasure(),
            "MAE": MAE(), "E_phi_adaptive_values": []}


def step_pysod(metric_objects: dict, prediction: np.ndarray, target: np.ndarray) -> None:
    # The library's non-normalizing path accepts [0,1] float predictions and bool GT.
    binary_gt = target >= 0.5
    pred = prediction.astype(np.float32, copy=False)
    metric_objects["S_alpha"].step(pred, binary_gt, normalize=False)
    metric_objects["weighted_F_beta"].step(pred, binary_gt, normalize=False)
    metric_objects["MAE"].step(pred, binary_gt, normalize=False)
    # The project calls its continuous soft enhanced-alignment score E_phi.
    # Compare against the standard PySODMetrics adaptive E scalar, without also
    # calculating and storing the unused 256-threshold E curve for every image.
    e_metric = Emeasure()
    e_metric.gt_fg_numel = int(np.count_nonzero(binary_gt))
    e_metric.gt_size = int(binary_gt.size)
    metric_objects["E_phi_adaptive_values"].append(float(e_metric.cal_adaptive_em(pred, binary_gt)))


def pysod_means(objects: dict) -> dict[str, float]:
    results = {"S_alpha": float(objects["S_alpha"].get_results()["sm"]),
               "weighted_F_beta": float(objects["weighted_F_beta"].get_results()["wfm"]),
               # PySODMetrics' standard scalar E-measure is adaptive E (adp).
               "E_m_adaptive": float(np.mean(objects["E_phi_adaptive_values"])),
               "MAE": float(objects["MAE"].get_results()["mae"])}
    return results


def process_chunk(samples, row_maps, config, cache_root):
    """Evaluate a chunk in a worker process to use several CPU cores."""
    internal_values = {name: {key: [] for key in ("S_alpha", "weighted_F_beta", "E_phi", "MAE")}
                       for name in row_maps}
    pysod = {name: create_metrics() for name in row_maps}
    for sample in samples:
        if sample.ground_truth_path is None:
            raise ValueError(f"CO row lacks ground truth: {sample.image_id}")
        with Image.open(sample.image_path) as image:
            shape = (image.height, image.width)
        candidate = read_mask(sample.mask_path, shape)
        support = candidate >= 0.5
        target = read_mask(sample.ground_truth_path, shape)
        cache_file = cache_path_for(sample, config, cache_root)
        if not cache_file.is_file():
            raise FileNotFoundError(f"Missing residual cache for {sample.image_id}: {cache_file}")
        with np.load(cache_file, allow_pickle=False) as cached:
            if "feature" not in cached.files or cached["feature"].dtype != np.float32:
                raise ValueError(f"Missing full-precision feature residual for {sample.image_id}")
            feature_map = cached["feature"].astype(np.float32, copy=False)
        if feature_map.shape != candidate.shape:
            raise ValueError(f"Feature/mask shape mismatch for {sample.image_id}")

        accepted = row_maps["verify"][sample.image_id]["predicted_label"] == "CO"
        if accepted != (row_maps["refine"][sample.image_id]["predicted_label"] == "CO"):
            raise ValueError(f"Verify/refine decision differs for {sample.image_id}")
        verify_prediction = candidate.copy() if accepted else np.zeros_like(candidate)
        spatial = np.zeros_like(candidate)
        if support.any():
            values = feature_map[support]
            lo, hi = np.percentile(values, [5, 95])
            if hi - lo > 1e-8:
                spatial[support] = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
            else:
                spatial[support] = 1.0 if float(values.mean()) >= 0.5 else 0.0
        refine_prediction = candidate * spatial
        if not accepted:
            refine_prediction = np.zeros_like(refine_prediction)

        for name, prediction in (("verify", verify_prediction), ("refine", refine_prediction)):
            row = row_maps[name][sample.image_id]
            if abs(float(row["predicted_mask_area_fraction"]) - float(prediction.mean())) > 1e-7:
                raise ValueError(f"Reconstructed {name} mask area differs from saved run for {sample.image_id}")
            for metric in internal_values[name]:
                internal_values[name][metric].append(float(row[metric]))
            step_pysod(pysod[name], prediction, target)
    return {"n": len(samples),
            "internal": {name: {metric: float(np.mean(values))
                                 for metric, values in metrics.items()}
                         for name, metrics in internal_values.items()},
            "pysodmetrics": {name: pysod_means(metrics) for name, metrics in pysod.items()}}


def process_raw_chunk(samples):
    """Evaluate raw SINet-V2 candidate masks, with no verifier or cache needed."""
    metrics = create_metrics()
    for sample in samples:
        if sample.ground_truth_path is None:
            raise ValueError(f"CO row lacks ground truth: {sample.image_id}")
        with Image.open(sample.image_path) as image:
            shape = (image.height, image.width)
        prediction = read_mask(sample.mask_path, shape)
        target = read_mask(sample.ground_truth_path, shape)
        step_pysod(metrics, prediction, target)
    return {"n": len(samples), "pysodmetrics": pysod_means(metrics)}


def run_raw_baseline(co_samples, output_dir: Path, workers: int, n_chunks: int) -> None:
    chunks = [list(part) for part in np.array_split(co_samples, min(n_chunks, len(co_samples)))]
    print(f"Evaluating raw candidate masks for {len(co_samples)} CO images in "
          f"{len(chunks)} chunks across {workers} workers...", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_raw_chunk, chunk) for chunk in chunks]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            print(f"Completed raw-baseline chunk {completed}/{len(chunks)}", flush=True)
    n = sum(item["n"] for item in results)
    if n != len(co_samples):
        raise ValueError(f"Raw baseline count mismatch: expected {len(co_samples)}, received {n}")
    metrics = {key: sum(item["pysodmetrics"][key] * item["n"] for item in results) / n
               for key in results[0]["pysodmetrics"]}
    raw = {"n_CO": n, "condition": "raw_baseline", "library": "py_sod_metrics",
           "library_version": version("pysodmetrics"), "prediction": "saved raw candidate masks",
           "pysodmetrics": metrics}
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "pysodmetrics_raw_baseline.json"
    raw_path.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    validation_path = output_dir / "pysodmetrics_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["conditions"]["raw_baseline"] = {
        "n_CO": n, "pysodmetrics": metrics,
        "prediction": "saved raw candidate masks; verification disabled"}
    validation["conditions"] = {
        key: validation["conditions"][key]
        for key in ("raw_baseline", "verify", "refine")
    }
    validation["raw_baseline_provenance"] = "Candidate masks + ground-truth masks from the same saved official manifest; no inference or reconstruction."
    validation_path.write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    order = ("raw_baseline", "verify", "refine")
    table_rows = []
    for condition in order:
        values = validation["conditions"][condition]["pysodmetrics"]
        table_rows.append({"condition": condition, "S_alpha": values["S_alpha"],
                           "weighted_F_beta": values["weighted_F_beta"],
                           "E_m_adaptive": values["E_m_adaptive"], "MAE": values["MAE"]})
    table_path = output_dir / "pysodmetrics_three_condition_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    md_path = output_dir / "PYSODMETRICS_VALIDATION.md"
    md_text = [
        "# PySODMetrics validation of official-split CO segmentation scores", "",
        "All three conditions use PySODMetrics 1.6.2 on the same 1,800 CO images and saved ground-truth masks.",
        "Raw uses saved candidate masks. Verify/refine are reconstructed from candidates and cached feature maps; their areas were checked against saved per-image outputs. No inference ran.",
        "E is sample-based adaptive E-measure (`E_m_adaptive`); it is distinct from the project's continuous `E_phi`.", "",
        "| Condition | S-alpha | Weighted F-beta | Adaptive E-m | MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        md_text.append(f"| {row['condition']} | {row['S_alpha']:.6f} | {row['weighted_F_beta']:.6f} | "
                       f"{row['E_m_adaptive']:.6f} | {row['MAE']:.6f} |")
    md_text += ["", "For S-measure and weighted F-measure, use the library values in thesis tables: the project's internal implementations are not canonical (see the prior implementation comparison in this folder). PySODMetrics documents verification against Fan's Matlab CODToolbox: <https://github.com/lartpang/PySODMetrics>.", ""]
    md_path.write_text("\n".join(md_text), encoding="utf-8")
    print(f"Wrote {raw_path}")
    print(f"Wrote {table_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-root", type=Path,
                        default=Path("data/experiments/usc12k/full/feature_band_lama_full"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/experiments/usc12k/full/val_manifest.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/experiments/usc12k/full/feature_band_lama_full/posthoc_analysis"))
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel CPU workers for PySODMetrics (default: 4)")
    parser.add_argument("--chunks", type=int, default=8,
                        help="Number of progress-reporting work chunks (default: 8)")
    parser.add_argument("--raw-baseline-only", action="store_true",
                        help="Evaluate saved raw candidate masks and assemble the three-condition PySODMetrics table")
    args = parser.parse_args()
    root = args.project_root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    run_root, manifest_path, output_dir = map(resolve, (args.run_root, args.manifest, args.output_dir))
    manifest = read_manifest(manifest_path)
    run_names = {"verify": "feature_band_verify", "refine": "feature_band_refine"}
    saved_rows = {name: rows_from_csv(run_root / folder / "per_image.csv")
                  for name, folder in run_names.items()}
    row_maps = {name: {row["image_id"]: row for row in rows}
                for name, rows in saved_rows.items()}
    ids = [sample.image_id for sample in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest has duplicate image IDs")
    for name, mapping in row_maps.items():
        if set(mapping) != set(ids):
            raise ValueError(f"{name} result IDs do not exactly match the manifest")

    config_path = run_root / "feature_band_verify" / "resume_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache_root = run_root / "_residual_cache"
    co_samples = [sample for sample in manifest if sample.label == "CO"]
    other_count = len(manifest) - len(co_samples)
    if len(co_samples) != 1800:
        raise ValueError(f"Expected 1800 official CO images, found {len(co_samples)}")
    if args.workers < 1 or args.chunks < args.workers:
        raise ValueError("workers must be >=1 and chunks must be >= workers")
    if args.raw_baseline_only:
        run_raw_baseline(co_samples, output_dir, args.workers, args.chunks)
        return
    n_chunks = min(args.chunks, len(co_samples))
    chunks = [list(part) for part in np.array_split(co_samples, n_chunks)]
    work_items = []
    for chunk in chunks:
        chunk_ids = {sample.image_id for sample in chunk}
        chunk_maps = {name: {image_id: row for image_id, row in mapping.items() if image_id in chunk_ids}
                      for name, mapping in row_maps.items()}
        work_items.append((chunk, chunk_maps, config, cache_root))
    results = []
    print(f"Evaluating {len(co_samples)} CO masks in {n_chunks} chunks across {args.workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_chunk, *item) for item in work_items]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            print(f"Completed chunk {completed}/{n_chunks}", flush=True)
    counts_co = sum(item["n"] for item in results)
    if counts_co != len(co_samples):
        raise ValueError(f"Worker count mismatch: expected {len(co_samples)}, received {counts_co}")
    internal_means, pysod_mean = {}, {}
    for name in run_names:
        internal_means[name] = {
            metric: sum(item["internal"][name][metric] * item["n"] for item in results) / counts_co
            for metric in results[0]["internal"][name]}
        pysod_mean[name] = {
            metric: sum(item["pysodmetrics"][name][metric] * item["n"] for item in results) / counts_co
            for metric in results[0]["pysodmetrics"][name]}

    output_rows = []
    summary = {"n_CO": counts_co, "n_non_CO_in_manifest": other_count,
               "library": "py_sod_metrics", "library_version": version("pysodmetrics"),
               "scope": "CO segmentation metrics, official full split",
               "mask_source": "candidate masks + cached full-precision feature residuals; no inference",
               "E_measure_protocol": "PySODMetrics sample-based adaptive E (em.adp); project E_phi is a distinct continuous enhanced-alignment implementation.",
               "conditions": {}}
    for name in run_names:
        summary["conditions"][name] = {"n_CO": counts_co, "internal": internal_means[name],
                                       "pysodmetrics": pysod_mean[name]}
        metric_pairs = (("S_alpha", "S_alpha"), ("weighted_F_beta", "weighted_F_beta"),
                        ("E_phi", "E_m_adaptive"), ("MAE", "MAE"))
        for internal_metric, library_metric in metric_pairs:
            diff = pysod_mean[name][library_metric] - internal_means[name][internal_metric]
            shown_metric = ("E_phi vs adaptive E_m" if internal_metric == "E_phi" else internal_metric)
            output_rows.append({"condition": f"feature_band_{name}", "metric": shown_metric,
                                "internal": internal_means[name][internal_metric],
                                "pysodmetrics": pysod_mean[name][library_metric],
                                "pysodmetrics_minus_internal": diff,
                                "absolute_difference": abs(diff)})

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pysodmetrics_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    json_path = output_dir / "pysodmetrics_validation.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"CO images evaluated: {counts_co}; other manifest images skipped: {other_count}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print("condition,metric,internal,pysodmetrics,delta")
    for row in output_rows:
        print(f"{row['condition']},{row['metric']},{row['internal']:.9f},"
              f"{row['pysodmetrics']:.9f},{row['pysodmetrics_minus_internal']:+.9f}")
    print(summary["E_measure_protocol"])


if __name__ == "__main__":
    main()
