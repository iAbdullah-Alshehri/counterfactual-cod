# Post-hoc analysis scripts

These scripts use the completed USC12K artifacts. They do not run SINet-V2,
LaMa, or a feature backbone. They are **descriptive follow-up analyses**: the
official split has already been evaluated and inspected, and must not be used
to select a model or tune hyperparameters.

Run from the project root in PowerShell after activating the project
environment:

```powershell
.\.venv\Scripts\Activate.ps1
python tools\analyze_saved_run.py --bootstrap-replicates 2000 --seed 20261001
python tools\fit_residual_classifier.py --bootstrap-replicates 2000 --seed 20261001
```

## `analyze_saved_run.py`

Reads the official manifest, verify/refine per-image CSVs, original SINet-V2
candidate masks, ground-truth masks, run configuration, and saved full-precision
residual maps. It reports:

- Three-way classification accuracy, macro-F1, balanced accuracy, class recall,
  negative false acceptance, and stratified image-bootstrap 95% intervals.
- A raw candidate-mask baseline versus verified and refined mask summaries.
- CO segmentation metrics both over all CO examples and over the subset accepted
  as CO, plus negative predicted-mask area.
- `posthoc_per_image.csv` and `posthoc_summary.json` under
  `data/experiments/usc12k/full/feature_band_lama_full/posthoc_analysis`.

It reconstructs refined masks from the cached feature maps using the evaluator's
5th/95th percentile normalization. It fails if required full-precision cache
files are missing; it does not silently recompute residuals. Segmentation metrics
are the project's internal implementations and should be checked against the
benchmark's official evaluator before publication.

## `fit_residual_classifier.py`

Fits one regularized multinomial logistic regression using only the saved
40-per-class USC12K training-derived calibration CSV. It standardizes the three
residual features using calibration rows only and uses a fixed L2 setting; there
is no hyperparameter sweep. It compares the model with the already-frozen
0.185/0.285 feature band on the existing 15-per-class holdout and official-split
per-image CSVs. It also writes bootstrap intervals and per-image class
probabilities to `classifier_summary.json` and `classifier_predictions.csv` in
`data/experiments/usc12k/full/feature_band_lama_full/residual_classifier`.

The old 15-per-class holdout has already been used to assess the band rule, and
the official split has already been inspected. Results from both are therefore
post-hoc comparisons, not untouched confirmatory estimates. The classifier must
not be selected based on the official-split results. Source-family confounding
also remains unresolved by these saved-output analyses.
