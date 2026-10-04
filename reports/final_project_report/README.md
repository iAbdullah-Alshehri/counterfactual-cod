# Revised advisor report

`report.pdf` is the compiled 10-page report. Source: `report.tex`, `references.bib`, `tables/`, and `figures/`. All four figure PNGs are 300 DPI. Full-precision results remain in the CSV/JSON files in `tables/` and the original project outputs identified in Appendix A.

## Compile

From this report directory, with MiKTeX/TeX Live available:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

Alternatively run `pdflatex report.tex`, `bibtex report`, and `pdflatex report.tex` twice. The PDF was successfully compiled with pdflatex/BibTeX and visually inspected. The app's standalone compiler reported an environment error (`Unable to find standard directories for platform`); use the supplied PDF or the command above for this multi-file project.

## Six-example supplementary export

Only the six IDs in `tables/qualitative_cases.csv` were selected. The evaluation command below was run from the project root. All six entries were cache hits: no SINet-V2 or LaMa inference was repeated. The recorded evaluation-loop time was 2.5894793999905232 seconds, excluding initialization. Cached historical inpainting times are not new execution times.

```powershell
.\.venv\Scripts\python.exe -m cod_recon.cli evaluate `
  --manifest reports/final_project_report/supplementary/manifest.csv `
  --output-dir reports/final_project_report/supplementary/refine `
  --backend lama `
  --feature-weights data/downloads/resnet18-f37072fd.pth `
  --decision-rule feature_band --low-cut 0.185 --high-cut 0.285 `
  --ablation feature --refine --save-previews `
  --cache-dir data/experiments/usc12k/full/feature_band_lama_full/_residual_cache
```

`supplementary/consistency_check.json` records the comparison with the original full run. All six predicted classes match; scores and final mask areas match to an absolute tolerance of 1e-10. The supplementary sample was deliberately selected by outcome and is illustrative, not a new performance estimate.

Figure 4 uses actual exported original, candidate, background, feature residual, and refined mask stages. Orange outlines identify CO ground truth. The residual heatmap uses a fixed 0–1 scale, and images are letterboxed without distortion.

To regenerate the figures from existing saved files (no inference), run from the project root:

```powershell
.\.venv\Scripts\python.exe reports/final_project_report/build_report_assets.py
```

This uses Matplotlib, Pillow and NumPy from the project environment. The build script reads project artifacts outside this report folder; those datasets and full-run caches are not included in the portable report bundle.
