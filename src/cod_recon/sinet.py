"""CPU-capable adapter for the official SINet-V2 pretrained COD baseline."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def _load_model(checkpoint: Path, device: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install CPU PyTorch first: python -m pip install -e .[model]") from exc
    repo = Path(__file__).resolve().parents[2] / "third_party" / "SINet-V2"
    model_file = repo / "lib" / "Network_Res2Net_GRA_NCD.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"Official SINet-V2 source is missing: {model_file}")
    sys.path.insert(0, str(repo))
    from lib.Network_Res2Net_GRA_NCD import Network

    network = Network(imagenet_pretrained=False)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # support older torch releases that lack weights_only
        state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and state and next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    network.load_state_dict(state, strict=True)
    network.to(device).eval()
    return torch, network


def predict_images(image_paths: list[Path], image_root: Path, output_dir: Path,
                   checkpoint: Path, *, size: int = 352, device: str = "cpu",
                   limit: int | None = None) -> Path:
    """Create deterministic soft candidate masks and a CSV linking them to images."""
    if device != "cpu":
        raise ValueError("This project currently supports CPU inference only")
    if size < 32:
        raise ValueError("Inference size must be at least 32")
    if limit is not None:
        image_paths = image_paths[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, model = _load_model(checkpoint, device)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    records = []
    started = time.perf_counter()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    for index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            resized = image.resize((size, size), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(((array - mean) / std).transpose(2, 0, 1).copy()).unsqueeze(0)
        with torch.inference_mode():
            prediction = torch.sigmoid(model(tensor.to(device))[-1])
            prediction = torch.nn.functional.interpolate(prediction, size=(height, width), mode="bilinear", align_corners=False)
        confidence = prediction[0, 0].cpu().numpy()
        # Match the official SINet-V2 test script's per-image output normalization.
        low, high = float(confidence.min()), float(confidence.max())
        confidence = (confidence - low) / (high - low + 1e-8)
        relative = image_path.resolve().relative_to(image_root.resolve())
        mask_path = (output_dir / relative.parent / f"{relative.stem}.png").resolve()
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(np.clip(confidence, 0, 1) * 255)).save(mask_path)
        records.append({"image_id": relative.with_suffix("").as_posix(),
                        "image": str(image_path.resolve()), "mask": str(mask_path),
                        "candidate_area_fraction": float((confidence >= 0.5).mean())})
        if index == 1 or index % 25 == 0 or index == len(image_paths):
            elapsed = time.perf_counter() - started
            print(f"SINet-V2 {index}/{len(image_paths)}; {elapsed / index:.2f} sec/image", flush=True)

    index_path = output_dir / "candidate_masks.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "image", "mask", "candidate_area_fraction"])
        writer.writeheader()
        writer.writerows(records)
    return index_path
