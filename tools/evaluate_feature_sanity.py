"""Evaluate a frozen feature-only rule on held-out USC12K training-source samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cod_recon.evaluate import evaluate


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def boot_ci(pos: np.ndarray, neg: np.ndarray, *, seed: int = 20260927, n: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for i in range(n):
        vals[i] = auc(rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True))
    return [float(x) for x in np.quantile(vals, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/experiments/cod10k_feature_only/usc_sanity_manifest.csv"))
    parser.add_argument("--rule", type=Path, default=Path("data/experiments/cod10k_feature_only/calibration/selected_feature_rule.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/cod10k_feature_only/usc_sanity"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/experiments/cod10k_feature_only/residual_cache"))
    parser.add_argument("--backend", choices=("lama", "opencv"), default="lama")
    parser.add_argument("--inpainting-model", type=Path, default=Path("data/downloads/inpainting_lama_2025jan.onnx"))
    parser.add_argument("--feature-weights", type=Path, default=Path("data/downloads/resnet18-f37072fd.pth"))
    args = parser.parse_args()
    rule = json.loads(args.rule.read_text(encoding="utf-8"))
    out = args.output_dir.resolve()
    summary = evaluate(args.manifest, out / "frozen_feature_rule", threshold=rule["threshold"],
                       weights={"appearance": 0.0, "feature": 1.0, "structural": 0.0},
                       backend=args.backend, ablation="feature", inpainting_model=args.inpainting_model,
                       feature_weights=args.feature_weights, cache_dir=args.cache_dir,
                       acceptance_direction=rule["acceptance_direction"], decision_rule="binary_threshold")
    with (out / "frozen_feature_rule" / "per_image.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    values = {label: np.array([float(r["residual_feature"]) for r in rows if r["label"] == label])
              for label in ("CO", "BG", "NOCOD")}
    if any(not len(v) for v in values.values()):
        parser.error("Sanity manifest must contain CO, BG, and NOCOD")
    direction = rule["acceptance_direction"]
    oriented = {k: (v if direction == "high" else -v) for k, v in values.items()}
    neg = np.concatenate([oriented["BG"], oriented["NOCOD"]])
    comparisons = {"CO_vs_BG": (oriented["CO"], oriented["BG"]),
                   "CO_vs_NOCOD": (oriented["CO"], oriented["NOCOD"]),
                   "CO_vs_BG_plus_NOCOD": (oriented["CO"], neg)}
    aucs = {name: {"auroc_CO_positive": auc(pos, negative),
                   "bootstrap_95ci": boot_ci(pos, negative, seed=20260927 + i)}
            for i, (name, (pos, negative)) in enumerate(comparisons.items())}
    threshold = float(rule["threshold"])
    accept = {label: (v >= threshold if direction == "high" else v <= threshold)
              for label, v in values.items()}
    rates = {label: float(mask.mean()) for label, mask in accept.items()}
    accuracy = (float(accept["CO"].sum()) + float((~accept["BG"]).sum()) + float((~accept["NOCOD"]).sum())) / sum(len(v) for v in values.values())
    result = {"frozen_rule": rule, "sample_sizes": {k: len(v) for k, v in values.items()},
              "source_subsets": {"CO": "CS", "BG": "BACKGROUND", "NOCOD": "HKU"},
              "score_means": {k: float(v.mean()) for k, v in values.items()},
              "score_medians": {k: float(np.median(v)) for k, v in values.items()},
              "acceptance_rates": rates, "co_vs_negative_accuracy": accuracy,
              "auroc": aucs, "evaluate_summary": summary,
              "official_usc_test_used": False,
              "caveat": "Small, training-derived transfer sanity sample, not official USC12K test performance. It samples one source family per class; each class remains source-associated and this cannot establish causal separation of semantic camouflage from all domain effects."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "feature_sanity_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Frozen feature-only rule: USC-source sanity check", "",
             f"Applied the COD10K-development-selected rule without any threshold or weight changes: accept CO when the feature score is {direction} {threshold:.3f}.", "",
             "This held-out sanity sample uses USC12K training-derived images that were excluded from the development sample and earlier manifests: CO from CS, BG from BACKGROUND, and NOCOD from HKU. The official USC12K test split was not used.", "",
             "| USC source class | n | Mean feature score | Median | CO acceptance rate under frozen rule |", "|---|---:|---:|---:|---:|"]
    for label, source in (("CO", "CS"), ("BG", "BACKGROUND"), ("NOCOD", "HKU")):
        lines.append(f"| {label} ({source}) | {len(values[label])} | {values[label].mean():.3f} | {np.median(values[label]):.3f} | {rates[label]:.3f} |")
    lines += ["", "## AUROC", "", "Scores are oriented according to the frozen feature rule; higher oriented score means more CO-like.", "",
              "| Comparison | AUROC (CO positive) | Bootstrap 95% CI |", "|---|---:|---:|"]
    for name, val in aucs.items():
        lines.append(f"| {name.replace('_', ' ')} | {val['auroc_CO_positive']:.3f} | {val['bootstrap_95ci'][0]:.3f}–{val['bootstrap_95ci'][1]:.3f} |")
    lines += ["", f"Frozen threshold accuracy over all three classes treated as CO vs negative: {accuracy:.3f}. This differs from AUROC because threshold performance depends on the selected operating point.", "",
              "This is a small training-derived transfer sanity check, not an official USC12K evaluation. Only one source family per class is represented, and label and source remain associated, so the result cannot prove source-independent camouflage understanding. Do not launch the full official-test run based on this sample alone.", ""]
    (out / "FEATURE_SANITY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Frozen rule {direction} {threshold:.3f}; accuracy={accuracy:.3f}; AUROC CO-vs-all-negative={aucs['CO_vs_BG_plus_NOCOD']['auroc_CO_positive']:.3f}", flush=True)


if __name__ == "__main__":
    main()
