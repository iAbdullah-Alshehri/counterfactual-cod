# Dataset acquisition and status

## Source policy

Download archives only from links published by the benchmark/model authors. Keep original archives and split files unchanged under `data/`; the downloader records SHA-256 digests in `data/download_manifest.json`. Dataset files are ignored by Git and remain on this computer. Follow each dataset's research and redistribution terms.

## Proposal datasets

| Dataset | Role | Author release found | Status |
|---|---|---|---|
| OPC16K | Main CO/BG/NOCOD realistic benchmark | [OPCOD repository](https://github.com/2231122/OPCOD) currently says release is forthcoming | Pending official release |
| USC12K | Realistic/unconstrained benchmark | [USCNet repository](https://github.com/ssecv/USCNet) provides image/annotation archive and class file | Downloaded and extracted |
| RCOD | Realistic camouflaged object detection | [RCOD repository](https://github.com/zhimengXin/RCOD) provides a Google Drive folder for detection datasets/annotations | Downloaded and extracted; inspect annotations before use |
| CAMO, COD10K, NC4K | Target-present segmentation benchmarks | [SINet-V2 repository](https://github.com/GewelsJI/SINet-V2) provides combined training/validation and testing archives | Downloaded and extracted |

The downloader covers USC12K, its separate class annotation file, the combined COD train/test archives, the official RCOD-D archive plus its published MD5 file, the SINet-V2 checkpoint, the LaMa ONNX inpainting model, and official Torchvision ImageNet ResNet-18 weights. RCOD is a box-detection benchmark and its published labels are not equivalent to the pixel masks required by this project's core residual/mask-refinement experiments. OPC16K is the preferred primary dataset, but there is no official public archive at the time this project was initialized. No unofficial substitute is treated as an equivalent dataset.

## Download

From the project root:

```powershell
python tools/download_data.py --datasets usc12k usc12k_classes cod_train cod_test rcod_d rcod_d_md5 sinet_v2_weights lama_inpainting resnet18_weights --extract
```

The downloader resumes partial transfers, verifies archive byte counts and ZIP CRCs, rejects unsafe ZIP paths/symlinks, and records file hashes. The author-published USC12K archive is approximately 2.31 GB; the COD bundles are approximately 1.09 GB and 1.31 GB; RCOD-D is approximately 3.14 GB; model checkpoints add about 0.24 GB, including the 46.8 MB ResNet-18 weights. Check free disk space before extraction because uncompressed images and masks need several times the archive space.

## Citation and access

- [USCNet / USC12K](https://github.com/ssecv/USCNet)
- [SINet-V2 dataset bundle](https://github.com/GewelsJI/SINet-V2)
- [RCOD](https://github.com/zhimengXin/RCOD)
- [OPCOD / OPC16K](https://github.com/2231122/OPCOD)
- [LaMa ONNX model, OpenCV implementation, and Apache-2.0 license](https://huggingface.co/opencv/inpainting_lama)

Check the current author pages before each download; access links and release status can change.
