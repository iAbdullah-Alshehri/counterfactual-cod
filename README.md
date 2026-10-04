# Counterfactual Background Reconstruction for Realistic COD

Research code scaffold based on the supplied master's project proposal. The experiment asks whether reconstructing the scene after masking a candidate region produces useful appearance, feature, and structural evidence for distinguishing camouflaged objects (CO), background-only scenes (BG), and non-camouflaged objects (NOCOD).

## Current scope

The project now includes author-source dataset acquisition, a balanced USC12K manifest builder, CPU inference with SINet-V2, pretrained LaMa background inpainting, three residual signals, candidate verification, optional mask refinement, and proposal-aligned ablations. OPC16K is still pending its official public release, so the available realistic benchmark is USC12K; the limitation is documented in `DATASETS.md`.

The primary inpainting backend is the LaMa ONNX model published by OpenCV and run with ONNX Runtime on CPU. OpenCV Telea is available as a classical baseline. Feature residuals use a frozen pretrained ResNet-18 layer2 embedding. The primary evaluation path uses its calibrated three-way band (BG/CO/NOCOD); the legacy binary threshold remains available only when explicitly selected. SINet-V2 code and weights are included for research and education use with attribution.

## Environment

Python 3.10+; OpenCV, NumPy, Pillow, and ONNX Runtime are installed by the package. SINet-V2 and frozen ImageNet ResNet-18 embeddings use PyTorch and Torchvision. This computer has no CUDA GPU, so inference runs on CPU.

On Windows PowerShell, set up the environment from the project root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install --index-url https://download.pytorch.org/whl/cpu ".[model]"
```

## Data manifest

Create a UTF-8 CSV with columns `image`, `label`, and optionally `mask`, `ground_truth`, `split`, and `image_id`. Labels must be `CO`, `BG`, or `NOCOD`. Paths may be absolute or relative to the manifest file. `mask` is the initial candidate mask produced by the baseline COD method; `ground_truth` is the target mask used only for target-present segmentation evaluation. BG and NOCOD rows should not have a ground-truth camouflage mask. Keep train/validation/test partitions and benchmark provenance in the manifest; never tune thresholds on the test split.

Example:

```csv
image,label,mask,ground_truth,split,image_id
images/a.jpg,CO,predictions/a.png,masks/a.png,test,a
images/b.jpg,BG,predictions/b.png,,,test,b
images/c.jpg,NOCOD,predictions/c.png,,,test,c
```

Masks can be binary or grayscale confidence maps. Predictions with no candidate should be represented as an all-zero mask, not omitted. The evaluator reports missing candidate masks as an input error.

## Run

```bash
python -m cod_recon.cli evaluate --manifest data/manifest.csv --output-dir runs/usc12k --backend lama --decision-rule feature_band --low-cut 0.185 --high-cut 0.285 --cache-dir runs/usc12k/cache --resume
```

This creates `per_image.csv`, `summary.json`, a flushed per-image checkpoint (`per_image.progress.csv`), and a configuration sidecar. With `--resume`, the first run starts normally and a matching rerun skips checkpointed images; it rejects changed manifests, inputs, models, or decision settings. It also caches reconstructions and full-precision residual maps, so an image interrupted before checkpointing can reuse completed computation. Keep the same input files and command arguments when resuming. Optional previews require `--save-previews`.

Download author-linked archives and the official baseline checkpoint:

```powershell
python tools/download_data.py --datasets usc12k usc12k_classes cod_train cod_test rcod_d rcod_d_md5 sinet_v2_weights lama_inpainting resnet18_weights --extract
```

Create a balanced pilot, generate candidate masks with SINet-V2 on CPU, and write an evaluation manifest. The completed 20-image smoke experiment is at `data/experiments/usc12k/pilot20`; regenerate it with:

```powershell
python tools/prepare_usc12k.py --sample-per-scene 5 --output-dir data/experiments/usc12k/pilot20
```

Scene-A maps to NOCOD (salient-only), Scene-B to CO, Scene-C to CO (camouflage is present alongside salient objects), and Scene-D to BG. The builder extracts camouflage pixels from palette index 2 and refuses to continue if the official scene lists do not cover `val.txt` exactly. The 40-image pilot checks the pipeline; it is not a benchmark result. Use `--sample-per-scene 0` only when ready to process the full 3,600-image evaluation split.

Run proposal-aligned residual and mask-refinement ablations on the pilot:

```powershell
python tools/run_ablations.py --manifest data/experiments/usc12k/pilot/val_manifest.csv --output-dir data/experiments/usc12k/pilot/ablations
```

By default, this runner uses the primary three-way ResNet feature-band decision: BG below 0.185, CO from 0.185 through 0.285, and NOCOD above 0.285. It reports three-way accuracy, macro-F1, and a confusion matrix. The optional refinement run uses that same decision rule. To deliberately run the legacy binary residual ablations, pass `--decision-rule binary_threshold`; they are not the default evaluation path. Reconstructions are cached for repeat conditions. In the first pilot, CPU LaMa averaged about 9.2 seconds per reconstruction. Its initial threshold rejected all candidates, a pipeline diagnostic rather than evidence of useful verification.

## Validation and calibration

Create an independent 20-per-class validation slice from USC12K's training partition. The builder checks that it is disjoint from the official `val.txt` split and excludes exact filename overlaps with the SINet-V2 COD training image archive:

```powershell
python tools/prepare_usc12k_validation.py --sample-per-class 20
```

The feature residual is a frozen ImageNet ResNet-18 `layer2` embedding cosine distance between each original image and its LaMa reconstruction. Full frames are resized to 256×256 and normalized with ImageNet statistics. The official Torchvision checkpoint is downloaded with the data command above. Calibrate thresholds and 15 weight configurations in both score directions using validation data only:

```powershell
python tools/calibrate_validation.py --manifest data/experiments/usc12k/validation/validation_manifest.csv --output-dir data/experiments/usc12k/validation/calibration
```

This is a historical binary CO-vs-negative calibration utility. It tests both score directions and residual weights, and writes a complete sweep and selected settings. Its output is not the primary three-way rule; use the USC12K feature-band calibration below for the default evaluation pipeline. These are training-derived results, not benchmark test results; do not tune on the official evaluation split.

The historical 60-image binary calibration selected `low` direction, threshold `0.10`, and weights `appearance=0.25, feature=0.00, structural=0.75` (tuning-set accuracy 0.817). It is preserved for audit and is not used by the primary evaluation default. The primary evaluation now uses the independently calibrated feature-only three-way band documented below.

Review [CLASS_DIAGNOSTICS.md](data/experiments/usc12k/validation/calibration/class_structure/CLASS_DIAGNOSTICS.md) alongside the three-way calibration and holdout reports below. USC12K's labels remain associated with source families, so neither random folds nor the small source-stratified holdout establishes source-independent transfer.

A source-matched diagnostic is recorded in [MATCHED_DIAGNOSTICS.md](data/experiments/cod10k_matched/analysis/MATCHED_DIAGNOSTICS.md). It compared 25 COD10K-CAM positives from the COD test archive with 25 COD10K-NonCAM negatives from USC12K's training-derived pool, using the frozen selected composite. The low-score composite fell to AUROC 0.424 (95% bootstrap CI 0.270–0.587) and 0.46 accuracy at its unchanged 0.10 threshold. The standalone ResNet feature, in the high-score direction, had AUROC 0.902 (0.810–0.971) on that pairwise comparison. This supports investigating feature-based multiclass or hierarchical fusion, but remains a small exploratory sample and does not justify tuning or launching the full run.

### Fresh feature-only development and transfer sanity check

Build disjoint samples and masks, calibrate on COD10K source-matched data only, then apply the frozen rule to the USC training-derived sanity sample:

```powershell
python tools/prepare_feature_only_samples.py
python tools/calibrate_feature_only.py
python tools/evaluate_feature_sanity.py
```

The sample builder excludes IDs from prior experiment manifests and exact-stem overlaps with SINet-V2 COD training images. It checks the USC training-derived sanity IDs against official `val.txt` IDs without evaluating or opening official test images/masks. The COD10K development set is separate from the earlier 50-image matched diagnostic. The threshold sweep uses only the ResNet-18 feature residual (`appearance=0`, `feature=1`, `structural=0`) and both score directions; it selects maximum accuracy on 25 COD10K-CAM and 25 COD10K-NonCAM rows. The frozen rule and tuning procedure are documented in `data/experiments/cod10k_feature_only/calibration/FEATURE_RULE.md` and `threshold_sweep.csv`.

In the fresh sweep, the selected rule was high-score CO at 0.175 (0.800 tuning accuracy; not an unbiased estimate). On a separate 15-per-class USC training-derived sample (CO=CS, BG=BACKGROUND, NOCOD=HKU), the frozen rule reached 0.467 threshold accuracy and CO-vs-(BG+NOCOD) AUROC 0.471 (bootstrap 95% CI 0.307–0.636). Pairwise AUROC was 0.849 for CO vs BG but 0.093 for CO vs NOCOD. Full details are in `data/experiments/cod10k_feature_only/usc_sanity/FEATURE_SANITY.md`. This is a small transfer diagnostic, not official USC12K test performance; the class/source association remains, and the failure against NOCOD means the feature-only binary rule is not ready for the full run.

### Three-way feature-band calibration on USC12K training data

The follow-up experiment uses two new, disjoint USC12K training-derived samples: 40 images per class for calibration and 15 per class for a frozen-rule check. Every ID from existing experiment manifests and every exact-stem overlap with SINet-V2 COD training were excluded. No official USC12K test image or mask was evaluated. Source representation was stratified across available groups (calibration CO: 35 CS, 3 AWA, 2 LUSI; BG: 20 COD10K, 20 BACKGROUND; NOCOD: 20 DUTS, 20 HKU).

Reproduce the samples and two-cut search:

```powershell
python tools/prepare_usc_feature_band_samples.py
python tools/calibrate_usc_feature_band.py
python tools/evaluate_usc_feature_band_holdout.py
```

The calibrated feature-only band labels scores `<0.185` as BG, scores from `0.185` through `0.285` as CO, and scores `>0.285` as NOCOD. Calibration accuracy was 0.708 (macro-F1 0.709), with confusion matrix (true rows, predicted columns; BG/CO/NOCOD) `[[32,7,1],[4,29,7],[3,13,24]]`. On the untouched 15-per-class holdout, accuracy was 0.667 (macro-F1 0.672), with matrix `[[11,4,0],[1,10,4],[1,5,9]]`. These are small training-derived results, not official test performance. Class labels remain strongly associated with dataset source families; the holdout supports further development but is not a source-independent validation. See `data/experiments/usc12k_feature_band/calibration/BAND_CALIBRATION.md`, `data/experiments/usc12k_feature_band/heldout/USC_BAND_HOLDOUT.md`, and `sample_selection.json` for complete cut sweep, class metrics, and selected source counts.

Explore residual separability without selecting a final threshold:

```powershell
python tools/analyze_residuals.py --per-image data/experiments/usc12k/pilot20/ablations/baseline_verify/per_image.csv --output-dir data/experiments/usc12k/pilot20/analysis
```

The supplied segmentation metric implementations are lightweight internal diagnostics; validate them against the cited benchmark evaluation code before comparing scores with published papers.

Use only training-derived validation data to select decision thresholds and weights. Keep USC12K's official evaluation split untouched for final reporting.

For a quick legacy binary plumbing check on one row, pass `--decision-rule binary_threshold`; the default three-way band requires all three labels in the manifest. A one-row run checks file loading and output generation only; it is not evidence for the research hypothesis.

## Method and outputs

For image `I` and baseline candidate `M0`, the backend fills the dilated candidate region to form `I_bg`. Appearance discrepancy is mean absolute RGB difference in the original candidate. Feature discrepancy is cosine distance between frozen ImageNet-pretrained ResNet-18 layer2 embeddings of the original and reconstructed image. Structural discrepancy is the difference between Canny edge maps. Component maps retain their common `[0,1]` input scale to preserve cross-image magnitude; the weighted combination defaults to equal weights. Calibration across a validation set is required before scientific comparison because the combined residual is not a calibrated probability.

The primary verifier uses mean frozen ResNet-18 feature residual inside the candidate mask and applies the calibrated two-cut rule: BG below the low cut, CO within the inclusive band, and NOCOD above the high cut. Predicted BG/NOCOD rows suppress the candidate mask; predicted CO rows retain it. The evaluator records the three-way confusion matrix and class metrics. The legacy binary threshold remains selectable for historical comparisons and calibration scripts, but is not the default evaluation behavior.

The summary reports CO-vs-negative accuracy, negative-scene false acceptance/rejection, false-positive mask area on BG/NOCOD, and target-present MAE, S-measure, weighted F-measure, and E-measure when ground-truth masks are supplied. It also records mean per-image runtime and per-class/per-split results. Results are not comparable to paper tables unless dataset versions/splits, candidate model/checkpoint, reconstruction backend/checkpoint, preprocessing, threshold calibration, and compute hardware are documented.

### Post-hoc analysis of the completed full run

The scripts in [POSTHOC_ANALYSIS.md](POSTHOC_ANALYSIS.md) compare the raw candidate masks with saved verify/refine results, calculate stratified bootstrap intervals, and fit a fixed regularized logistic regression on the three cached residual scalars. They do not rerun LaMa. The official USC12K split and the 15-per-class holdout have already been inspected, so all new comparisons on those rows are explicitly exploratory; do not select a model or tune settings from their results.

## Planned work

1. Build a source-aware validation design that tests class semantics independently of dataset-family shortcuts.
2. Compare binary and hierarchical or three-class decision rules, including the learned feature signal.
3. Assess LaMa reconstruction quality and compare it with the classical OpenCV baseline.
4. After the validation design is fixed, run frozen residual ablations, negative-scene rejection, refinement, sensitivity analyses, and timing comparisons.
5. Add existence-aware baselines and complete the empirical write-up.

## Proposal and related primary sources

- Proposal: `project_proposal_draft_4.pdf` (provided by the project author).
- OPC16K / OPCNet: [arXiv:2608.11135](https://arxiv.org/abs/2608.11135); authors' repository: [2231122/OPCOD](https://github.com/2231122/OPCOD).
- USC12K / USCNet: [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_Rethinking_Detecting_Salient_and_Camouflaged_Objects_in_Unconstrained_Scenes_ICCV_2025_paper.pdf).
- RCOD: [arXiv:2501.07297](https://arxiv.org/abs/2501.07297); authors' repository: [zhimengXin/RCOD](https://github.com/zhimengXin/RCOD).
- SINet-V2 baseline and official checkpoint: [authors' repository](https://github.com/GewelsJI/SINet-V2). Its README requests research/education use and citation.
- LaMa ONNX model and Apache-2.0 license: [OpenCV model repository](https://huggingface.co/opencv/inpainting_lama).

Check dataset terms, access conditions, and official split definitions before downloading or redistributing any data.
