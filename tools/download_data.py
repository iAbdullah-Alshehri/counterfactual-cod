"""Download author-linked research datasets with resume and safe extraction.

This script only fetches original archive files into data/downloads. It does not
redistribute them or alter official files/splits. Extracted files stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


DATASETS = {
    "usc12k": {
        "file_id": "1MIVCH7sLOzFwrzEDjKs2PSba7UpzLG7I",
        "filename": "VOC-USC12K.zip",
        "bytes": 2_310_013_741,
        "source": "https://github.com/ssecv/USCNet",
    },
    "usc12k_classes": {
        "file_id": "10cQDz1b910D2JNNQfv--WbxOZXavQTZE",
        "filename": "Scenen_Class_Excel_Refine.zip",
        "bytes": 288_843,
        "source": "https://github.com/ssecv/USCNet",
    },
    "cod_test": {
        "file_id": "1V0iSEdYJrT0Y_DHZfVGMg6TySFRNTy4o",
        "filename": "COD-TestDataset.zip",
        "bytes": 1_310_372_023,
        "source": "https://github.com/GewelsJI/SINet-V2",
    },
    "cod_train": {
        "file_id": "1M8-Ivd33KslvyehLK9_IUBGJ_Kf52bWG",
        "filename": "COD-TrainDataset.zip",
        "bytes": 1_089_873_536,
        "source": "https://github.com/GewelsJI/SINet-V2",
    },
    "rcod_d": {
        "file_id": "1IN12EqKfIYF2NqdCKF8vsl6rDuqvhl4z",
        "filename": "COD-D.zip",
        "bytes": 3_138_477_774,
        "source": "https://github.com/zhimengXin/RCOD",
    },
    "rcod_d_md5": {
        "file_id": "12JW8mTgrgo_bclct3rjDp4sKAWsP-S_Q",
        "filename": "COD-D.md5",
        "bytes": 44,
        "source": "https://github.com/zhimengXin/RCOD",
    },
    "sinet_v2_weights": {
        "file_id": "1D3RKQ8Nzd0ArV_c47StVKEuaoYTwnclR",
        "filename": "SINet_V2_Net_epoch_best.pth",
        "bytes": 108_330_636,
        "source": "https://github.com/GewelsJI/SINet-V2",
    },
    "lama_inpainting": {
        "url": "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
        "filename": "inpainting_lama_2025jan.onnx",
        "bytes": 92_591_623,
        "source": "https://huggingface.co/opencv/inpainting_lama",
    },
    "resnet18_weights": {
        "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "filename": "resnet18-f37072fd.pth",
        "bytes": 46_830_571,
        "source": "https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18",
    },
}


def _request(url: str, start: int = 0) -> urllib.request.urlopen:
    headers = {"User-Agent": "counterfactual-cod-research/0.1", "Accept-Encoding": "identity"}
    if start:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=90)


def download(name: str, root: Path) -> Path:
    item = DATASETS[name]
    root.mkdir(parents=True, exist_ok=True)
    target = root / item["filename"]
    partial = target.with_suffix(target.suffix + ".part")
    if target.is_file() and target.stat().st_size == item["bytes"]:
        print(f"Already present: {target} ({target.stat().st_size:,} bytes)", flush=True)
        return target

    url = item.get("url") or (
        "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
            {"id": item["file_id"], "export": "download", "confirm": "t"}
        )
    )
    attempts = 0
    while attempts < 8:
        attempts += 1
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with _request(url, offset) as response:
                if response.status not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP status {response.status}")
                if offset and response.status == 200:
                    offset = 0
                mode = "ab" if offset and response.status == 206 else "wb"
                written = offset
                started = time.monotonic()
                last_report = started
                with partial.open(mode) as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        written += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 5:
                            elapsed = max(now - started, 0.1)
                            percent = written * 100 / item["bytes"]
                            print(f"{name}: {percent:5.1f}% ({written:,}/{item['bytes']:,} bytes; {written / elapsed / 1e6:.1f} MB/s)", flush=True)
                            last_report = now
                if written != item["bytes"]:
                    raise IOError(f"Expected {item['bytes']:,} bytes, got {written:,}")
                partial.replace(target)
                print(f"Downloaded: {target}", flush=True)
                return target
        except (OSError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            print(f"Attempt {attempts}/8 for {name} stopped: {exc}", file=sys.stderr, flush=True)
            if attempts >= 8:
                raise
            time.sleep(min(2**attempts, 30))
    raise RuntimeError(f"Could not download {name}")


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise zipfile.BadZipFile(f"CRC error in archive member: {bad}")
        for member in zf.infolist():
            name = member.filename.replace("\\", "/")
            if name.startswith("__MACOSX/") or Path(name).name.startswith("._"):
                continue
            if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
                raise ValueError(f"Unsafe absolute archive member: {member.filename}")
            target = (base / name).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if member.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError(f"Refusing symbolic link in dataset archive: {member.filename}")
        members = [info for info in zf.infolist()
                   if not info.filename.replace("\\", "/").startswith("__MACOSX/")
                   and not Path(info.filename).name.startswith("._")]
        zf.extractall(destination, members=members)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--extract", action="store_true", help="validate and safely extract each downloaded ZIP")
    parser.add_argument("--workers", type=int, default=3, help="parallel downloads (default: 3)")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")
    downloads = args.data_dir / "downloads"
    paths: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, len(args.datasets))) as pool:
        futures = {pool.submit(download, name, downloads): name for name in args.datasets}
        for future in as_completed(futures):
            paths[futures[future]] = future.result()
    records = []
    for name in args.datasets:
        path = paths[name]
        record = {"name": name, "filename": path.name, "bytes": path.stat().st_size,
                  "sha256": sha256(path), "source": DATASETS[name]["source"]}
        if args.extract and path.suffix.lower() == ".zip":
            destination = args.data_dir / "raw" / name
            safe_extract(path, destination)
            record["extracted_to"] = str(destination.resolve())
        records.append(record)
    registry_path = args.data_dir / "download_manifest.json"
    previous = {}
    if registry_path.exists():
        previous = {x["name"]: x for x in json.loads(registry_path.read_text(encoding="utf-8"))}
    previous.update({x["name"]: x for x in records})
    registry_path.write_text(json.dumps(list(previous.values()), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote provenance and checksums: {registry_path}")


if __name__ == "__main__":
    main()
