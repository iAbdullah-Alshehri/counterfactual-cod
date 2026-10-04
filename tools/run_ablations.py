"""Run proposal-aligned residual ablations on a fixed candidate-mask manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cod_recon.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decision-rule", choices=("feature_band", "binary_threshold"), default="feature_band",
                        help="primary three-way feature band by default; opt into legacy binary ablations explicitly")
    parser.add_argument("--low-cut", type=float, default=0.185)
    parser.add_argument("--high-cut", type=float, default=0.285)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--backend", choices=("lama", "opencv"), default="lama")
    parser.add_argument("--inpainting-model", type=Path, default=Path("data/downloads/inpainting_lama_2025jan.onnx"))
    parser.add_argument("--dilation", type=int, default=5)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="resume per-condition evaluation checkpoints")
    parser.add_argument("--feature-weights", type=Path)
    parser.add_argument("--acceptance-direction", choices=("high", "low"), default="high")
    parser.add_argument("--save-previews", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    modes = ([ ("feature_band", False), ("feature_band", True) ] if args.decision_rule == "feature_band"
             else [("baseline", False), ("appearance", False), ("feature", False),
                   ("structural", False), ("combined", False), ("combined", True)])
    rows = []
    for signal, refine in modes:
        name = f"{signal}_{'refine' if refine else 'verify'}"
        summary = evaluate(args.manifest, out / name, threshold=args.threshold,
                           ablation="feature" if args.decision_rule == "feature_band" else signal,
                           refine=refine, save_previews=args.save_previews,
                           backend=args.backend, inpainting_model=args.inpainting_model,
                           dilation=args.dilation, feature_weights=args.feature_weights,
                           acceptance_direction=args.acceptance_direction,
                           decision_rule=args.decision_rule, low_cut=args.low_cut, high_cut=args.high_cut,
                           resume=args.resume,
                           cache_dir=args.cache_dir or out / "_residual_cache")
        rows.append({"condition": name, "decision_rule": args.decision_rule,
                     "low_cut": args.low_cut if args.decision_rule == "feature_band" else None,
                     "high_cut": args.high_cut if args.decision_rule == "feature_band" else None,
                     "threshold": args.threshold if args.decision_rule == "binary_threshold" else None,
                     "acceptance_direction": args.acceptance_direction,
                     "mask_refinement": refine, "co_vs_negative_accuracy": summary["co_vs_negative_accuracy"],
                     "three_way_accuracy": summary["three_way_accuracy"], "macro_f1": summary["macro_f1"],
                     "confusion_matrix_true_rows_predicted_columns": summary["confusion_matrix_true_rows_predicted_columns"],
                     "negative_false_accept_rate": summary["negative_scene_false_accept_rate"],
                     "negative_rejection_accuracy": summary["negative_scene_rejection_accuracy"],
                     "co_acceptance_rate": summary["co_acceptance_rate"],
                     "mean_runtime_seconds_per_image": summary["mean_runtime_seconds_per_image"],
                     **{f"CO_{key}": value for key, value in summary["metrics_by_label"].get("CO", {}).items() if key != "n"}})
        print(name, summary["three_way_accuracy"] if args.decision_rule == "feature_band" else summary["co_vs_negative_accuracy"],
              summary["confusion_matrix_true_rows_predicted_columns"], flush=True)
    with (out / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote ablation summary: {out / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
