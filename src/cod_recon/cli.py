"""Command-line interface."""

from __future__ import annotations

import argparse
import json

from .evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(prog="cod-recon", description="Evaluate counterfactual reconstruction residuals for realistic COD")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("evaluate", help="evaluate candidate masks from a CSV manifest")
    run.add_argument("--manifest", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--backend", choices=("lama", "opencv"), default="lama")
    run.add_argument("--inpainting-model", default="data/downloads/inpainting_lama_2025jan.onnx")
    run.add_argument("--dilation", type=int, default=5, help="candidate-mask expansion radius in pixels")
    run.add_argument("--cache-dir", help="optional reconstruction/residual cache for repeat ablations")
    run.add_argument("--resume", action="store_true",
                     help="checkpoint each completed image and safely skip it on a matching rerun")
    run.add_argument("--decision-rule", choices=("feature_band", "binary_threshold"), default="feature_band",
                     help="primary three-way ResNet feature band (default) or legacy binary threshold")
    run.add_argument("--low-cut", type=float, default=0.185, help="BG/CO cut for feature_band")
    run.add_argument("--high-cut", type=float, default=0.285, help="CO/NOCOD cut for feature_band")
    run.add_argument("--threshold", type=float, default=0.5, help="single cutoff for --decision-rule binary_threshold")
    run.add_argument("--acceptance-direction", choices=("high", "low"), default="high",
                     help="whether CO candidates are accepted above or below the residual threshold")
    run.add_argument("--appearance-weight", type=float)
    run.add_argument("--feature-weight", type=float)
    run.add_argument("--structural-weight", type=float)
    run.add_argument("--feature-weights", help="local pretrained ResNet-18 state dictionary; downloads official ImageNet weights if omitted")
    run.add_argument("--refine", action="store_true", help="attenuate candidate mask by combined residual")
    run.add_argument("--ablation", choices=("baseline", "appearance", "feature", "structural", "combined"), default="combined")
    run.add_argument("--save-previews", action="store_true")
    args = parser.parse_args()
    weight_values = {"appearance": args.appearance_weight, "feature": args.feature_weight,
                     "structural": args.structural_weight}
    weights = None if all(value is None for value in weight_values.values()) else {
        name: (value if value is not None else 1 / 3) for name, value in weight_values.items()}
    summary = evaluate(args.manifest, args.output_dir, threshold=args.threshold, weights=weights,
                       backend=args.backend, refine=args.refine, save_previews=args.save_previews,
                       ablation=args.ablation, inpainting_model=args.inpainting_model,
                       dilation=args.dilation, cache_dir=args.cache_dir,
                       feature_weights=args.feature_weights,
                       acceptance_direction=args.acceptance_direction,
                       decision_rule=args.decision_rule, low_cut=args.low_cut, high_cut=args.high_cut,
                       resume=args.resume)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
