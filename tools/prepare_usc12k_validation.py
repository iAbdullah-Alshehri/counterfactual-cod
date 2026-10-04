"""Build a held-out USC12K training-derived validation manifest (never the official test split)."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from PIL import Image

from cod_recon.sinet import predict_images


def _class_ids(workbook: Path) -> set[str]:
    sheet = load_workbook(workbook, read_only=True, data_only=True).active
    return {str(row[0]).strip() for row in sheet.iter_rows(min_row=2, values_only=True) if row[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/usc12k/VOC-USC12K"))
    parser.add_argument("--classes-dir", type=Path, default=Path("data/raw/usc12k_classes/Train_Class"))
    parser.add_argument("--cod-training-images", type=Path, default=Path("data/raw/cod_train/COD-TrainDataset/Imgs"),
                        help="exclude exact image stems previously used to train SINet-V2")
    parser.add_argument("--sample-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/downloads/SINet_V2_Net_epoch_best.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/usc12k/validation"))
    args = parser.parse_args()
    if args.sample_per_class < 1:
        parser.error("--sample-per-class must be at least 1")
    root, classes_dir = args.dataset_root.resolve(), args.classes_dir.resolve()
    train_file = root / "ImageSets/Segmentation/train.txt"
    test_file = root / "ImageSets/Segmentation/val.txt"
    train_ids = set(train_file.read_text(encoding="utf-8").split())
    test_ids = set(test_file.read_text(encoding="utf-8").split())
    if train_ids & test_ids:
        parser.error("USC12K train.txt overlaps val.txt; refusing to build validation data")
    classes = {scene: _class_ids(classes_dir / filename) for scene, filename in
               (("A", "SceneA2.xlsx"), ("B", "SceneB2.xlsx"), ("C", "SceneC2.xlsx"))}
    for scene, ids in classes.items():
        if not ids <= train_ids:
            parser.error(f"Scene {scene} annotation contains IDs outside official train.txt")
    scene_a, scene_b, scene_c = (classes[key] for key in "ABC")
    scene_d = train_ids - scene_a - scene_b - scene_c
    candidates = {"NOCOD": scene_a, "CO": scene_c, "BG": scene_d}
    if any(not ids for ids in candidates.values()):
        parser.error("Could not derive all three labels from the official training annotations")

    cod_stems = {path.stem for path in args.cod_training_images.rglob("*") if path.is_file()}
    # Scene B is known to overlap the SINet-V2 COD training archive; use held-out
    # Scene-C CO examples that are absent by exact stem instead.
    candidates["CO"] = candidates["CO"] - cod_stems
    if len(candidates["CO"]) < args.sample_per_class:
        parser.error(f"Only {len(candidates['CO'])} training CO images remain after exact SINet train-overlap exclusion")
    chosen: dict[str, str] = {}
    rng = random.Random(args.seed)
    for label in ("CO", "BG", "NOCOD"):
        pool = sorted(candidates[label])
        rng.shuffle(pool)
        chosen.update({image_id: label for image_id in pool[:args.sample_per_class]})
    if set(chosen) & test_ids:
        parser.error("Validation samples overlap official USC12K val.txt; refusing to proceed")

    image_root, label_root = root / "JPEGImages", root / "SegmentationClass"
    images = [image_root / f"{image_id}.jpg" for image_id in sorted(chosen)]
    missing = [path for path in images if not path.is_file()]
    if missing:
        parser.error(f"Missing image: {missing[0]}")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pred_index = predict_images(images, image_root, out / "candidate_masks", args.checkpoint.resolve())
    prediction_rows = {row["image_id"].split("/")[-1]: row["mask"]
                       for row in csv.DictReader(pred_index.open(encoding="utf-8"))}
    manifest = out / "validation_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        fields = ("image", "label", "mask", "ground_truth", "split", "image_id", "source_scene")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image_path in images:
            image_id, label = image_path.stem, chosen[image_path.stem]
            target = out / "ground_truth" / f"{image_id}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            if label == "CO":
                with Image.open(label_root / f"{image_id}.png") as indexed:
                    mask = np.asarray(indexed.convert("P"))
                Image.fromarray(np.where(mask == 2, 255, 0).astype(np.uint8), mode="L").save(target)
                scene = "Scene-C"
            elif label == "NOCOD":
                with Image.open(label_root / f"{image_id}.png") as indexed:
                    Image.new("L", indexed.size, 0).save(target)
                scene = "Scene-A"
            else:
                with Image.open(label_root / f"{image_id}.png") as indexed:
                    Image.new("L", indexed.size, 0).save(target)
                scene = "Scene-D (train complement)"
            writer.writerow({"image": str(image_path.resolve()), "label": label,
                             "mask": prediction_rows[image_id], "ground_truth": str(target.resolve()),
                             "split": "usc12k_train_heldout_validation", "image_id": image_id,
                             "source_scene": scene})
    print(f"Wrote held-out train validation manifest: {manifest}")
    for label in ("CO", "BG", "NOCOD"):
        print(f"{label}: {args.sample_per_class}")
    print(f"Disjoint from official val.txt: {not bool(set(chosen) & test_ids)}")
    print(f"Exact-stem overlap with SINet COD training archives excluded: {len(cod_stems)} stems")


if __name__ == "__main__":
    main()
