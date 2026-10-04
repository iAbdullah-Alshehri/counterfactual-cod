"""Manifest loading and image/mask I/O."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


LABELS = {"CO", "BG", "NOCOD"}


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: Path
    label: str
    mask_path: Path
    ground_truth_path: Path | None
    split: str


def _resolve(base: Path, raw: str | None, field: str, row_number: int) -> Path | None:
    if raw is None or not raw.strip():
        return None
    path = Path(raw.strip())
    path = path if path.is_absolute() else (base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest row {row_number}: {field} file not found: {path}")
    return path


def read_manifest(path: str | Path) -> list[Sample]:
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    samples: list[Sample] = []
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"image", "label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Manifest must contain at least the columns: image,label")
        for row_number, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip().upper()
            if label not in LABELS:
                raise ValueError(f"Manifest row {row_number}: label must be CO, BG, or NOCOD")
            image = _resolve(manifest.parent, row.get("image"), "image", row_number)
            mask = _resolve(manifest.parent, row.get("mask"), "mask", row_number)
            gt = _resolve(manifest.parent, row.get("ground_truth"), "ground_truth", row_number)
            if image is None:
                raise ValueError(f"Manifest row {row_number}: image is required")
            if mask is None:
                raise ValueError(f"Manifest row {row_number}: candidate mask is required (use an all-zero mask for abstention)")
            if label == "CO" and gt is None:
                raise ValueError(f"Manifest row {row_number}: CO rows require ground_truth for segmentation metrics")
            image_id = (row.get("image_id") or "").strip() or image.stem
            samples.append(Sample(image_id, image, label, mask, gt, (row.get("split") or "unspecified").strip()))
    if not samples:
        raise ValueError("Manifest contains no samples")
    return samples


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return mask.astype(np.float32) / 255.0
