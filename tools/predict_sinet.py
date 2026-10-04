"""Generate baseline COD masks for every image below --image-root."""

from __future__ import annotations

import argparse
from pathlib import Path

from cod_recon.sinet import predict_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/downloads/SINet_V2_Net_epoch_best.pth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=352)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    image_root = args.image_root.resolve()
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    if not images:
        parser.error(f"No image files found under {image_root}")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}; run tools/download_data.py --datasets sinet_v2_weights")
    output = args.output_dir.resolve()
    index = predict_images(images, image_root, output, checkpoint, size=args.size, limit=args.limit)
    print(f"Wrote candidate index: {index}")


if __name__ == "__main__":
    main()
