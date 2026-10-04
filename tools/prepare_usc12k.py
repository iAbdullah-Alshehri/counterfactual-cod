"""Build a stratified USC12K evaluation manifest and generate SINet-V2 candidates."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

from cod_recon.sinet import predict_images


SCENE_LABELS = {"Scene-A": "NOCOD", "Scene-B": "CO", "Scene-C": "CO", "Scene-D": "BG"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/usc12k/VOC-USC12K"))
    parser.add_argument("--split", choices=("val",), default="val", help="official USC12K evaluation split")
    parser.add_argument("--sample-per-scene", type=int, default=10, help="balanced pilot sample per scene; 0 uses the full split")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/downloads/SINet_V2_Net_epoch_best.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/usc12k/pilot"))
    parser.add_argument("--size", type=int, default=352)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    sets = root / "ImageSets" / "Segmentation"
    image_root, label_root = root / "JPEGImages", root / "SegmentationClass"
    for needed in (sets, image_root, label_root):
        if not needed.is_dir():
            parser.error(f"USC12K component not found: {needed}; download and extract usc12k first")
    scene_ids = {scene: set((sets / f"{scene}.txt").read_text(encoding="utf-8").split()) for scene in SCENE_LABELS}
    val_ids = set((sets / "val.txt").read_text(encoding="utf-8").split())
    if set().union(*scene_ids.values()) != val_ids:
        parser.error("Official scene lists do not exactly cover val.txt; refusing to infer labels")
    if sum(len(v) for v in scene_ids.values()) != len(val_ids):
        parser.error("Scene lists overlap; refusing to infer three-way labels")

    rng = random.Random(args.seed)
    chosen: dict[str, str] = {}
    for scene, ids in scene_ids.items():
        candidates = sorted(ids)
        if args.sample_per_scene:
            if args.sample_per_scene < 0:
                parser.error("--sample-per-scene must be nonnegative")
            rng.shuffle(candidates)
            candidates = candidates[:args.sample_per_scene]
        chosen.update({image_id: scene for image_id in candidates})
    images = [image_root / f"{image_id}.jpg" for image_id in sorted(chosen)]
    missing = [path for path in images if not path.is_file()]
    if missing:
        parser.error(f"Missing image file: {missing[0]}")

    out = args.output_dir.resolve()
    prediction_dir = out / "candidate_masks"
    index_path = predict_images(images, image_root, prediction_dir, args.checkpoint.resolve(), size=args.size)
    mask_rows = {row["image"]: row["mask"] for row in csv.DictReader(index_path.open(encoding="utf-8"))}
    manifest = out / f"{args.split}_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "label", "mask", "ground_truth", "split", "image_id", "source_scene"])
        writer.writeheader()
        for image_path in images:
            image_id = image_path.stem
            scene = chosen[image_id]
            gt_path = label_root / f"{image_id}.png"
            if not gt_path.is_file():
                parser.error(f"Missing segmentation label: {gt_path}")
            if SCENE_LABELS[scene] == "CO":
                # USC palette index 2 marks camouflage; index 1 marks salient objects.
                with Image.open(gt_path) as label_image:
                    gt_path_out = out / "camouflage_ground_truth" / f"{image_id}.png"
                    gt_path_out.parent.mkdir(parents=True, exist_ok=True)
                    mask = label_image.convert("P")
                    values = np.asarray(mask)
                    binary = np.where(values == 2, 255, 0).astype("uint8")
                    Image.fromarray(binary, mode="L").save(gt_path_out)
            else:
                # BG and NOCOD are negative for camouflage; their target mask is empty.
                gt_path_out = out / "camouflage_ground_truth" / f"{image_id}.png"
                gt_path_out.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(gt_path) as label_image:
                    Image.new("L", label_image.size, color=0).save(gt_path_out)
            writer.writerow({"image": str(image_path), "label": SCENE_LABELS[scene],
                             "mask": mask_rows[str(image_path.resolve())], "ground_truth": str(gt_path_out),
                             "split": "usc12k_official_val", "image_id": image_id, "source_scene": scene})
    print(f"Wrote {len(images)} stratified samples to {manifest}")
    for scene in SCENE_LABELS:
        print(f"{scene} ({SCENE_LABELS[scene]}): {sum(value == scene for value in chosen.values())}")


if __name__ == "__main__":
    main()
