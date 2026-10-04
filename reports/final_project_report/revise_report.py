from pathlib import Path
import re,json,csv,shutil
H=Path('reports/final_project_report'); p=H/'report.tex'; t=p.read_text(encoding='utf-8-sig')
# Centralize every existing source note before removing repetitive blocks.
source_blocks=re.findall(r'\\src\{([^}]*)\}',t)
for f in (H/'tables').glob('*.tex'):source_blocks+=re.findall(r'\\src\{([^}]*)\}',f.read_text())
(H/'supplementary/original_source_notes.json').write_text(json.dumps(source_blocks,indent=2))
t=re.sub(r'\\newcommand\{\\src\}[^\n]*\n','',t)
t=re.sub(r'\\src\{[^}]*\}','',t)
t=t.replace(r'\documentclass[11pt,a4paper]',r'\documentclass[10pt,a4paper]').replace('margin=25mm','margin=20mm')
t=t.replace(r'\usepackage{longtable}',r'''\usepackage{longtable}
\usepackage{multicol}
\usepackage{amsmath}
\usepackage{placeins}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.08}
\setlength{\textfloatsep}{9pt}
\setlength{\intextsep}{9pt}
\setlist{nosep,topsep=4pt}
\setlength{\emergencystretch}{2em}''')
# setlist must follow enumitem load
line=r'\setlist{nosep,topsep=4pt}'
t=t.replace(line+'\n','').replace(r'\usepackage{enumitem}',r'\usepackage{enumitem}'+'\n'+line)
t=t.replace('0.55em','0.3em').replace(r'\captionsetup{font=small,labelfont=bf}',r'\captionsetup{font=small,labelfont=bf,skip=4pt}')
t=t.replace(r'\maketitle',r'''\begin{titlepage}
\centering
{\Large\bfseries Counterfactual Background Reconstruction\\for Realistic Camouflaged Object Detection\par}
\vspace{0.5em}
{\large Executive summary for thesis-advisor review\par}
{\small 3 October 2026\par}
\vspace{1.4em}
\raggedright
\textbf{Research question.} Can reconstructing a plausible background after removing a frozen COD model's candidate provide useful evidence for distinguishing camouflage, empty background, and salient non-camouflaged objects? The project tests this through SINet-V2 candidates, LaMa reconstruction, and appearance, learned-feature, and structural residuals, followed by verification and optional refinement.

\medskip
\textbf{Headline findings on the saved official evaluation split.}
\begin{itemize}[leftmargin=1.4em,itemsep=0.65em]
\item \textbf{Three-way accuracy: 0.7122}; macro-F1: \textbf{0.7018}, on 3,600 USC12K images. The saved split is named \texttt{usc12k\_official\_val}.
\item \textbf{Negative false acceptance: 1.0000 $\rightarrow$ 0.2883}. The baseline comparison defines any nonempty raw candidate as accepted; verification rejects many negative scenes but also rejects some true CO images.
\item \textbf{Segmentation tradeoff:} raw SINet-V2 masks remain best on PySODMetrics S-alpha, weighted F-beta, and adaptive E-measure. Verification has only a small MAE advantage; refinement does not provide an overall CO segmentation improvement.
\end{itemize}
\medskip
\textbf{What the calibration revealed.} The initial appearance/structural rule failed the source-matched diagnostic; a COD10K-calibrated feature cutoff inverted on USC12K salient-object negatives. A feature-only band, calibrated on USC12K training-derived samples, was frozen before the full evaluation, but source/class confounding remains unresolved.

\medskip
\textbf{Thesis claim.} The verifier succeeds as a candidate rejection mechanism on this evaluated split, while the evidence does not show that it improves CO segmentation or generalizes independently of dataset source.

\medskip
\textbf{Assessment.} This supports a substantive master's thesis as an audited empirical study of counterfactual verification, dataset shortcuts, and the rejection--segmentation tradeoff. A claim of state-of-the-art performance, source-independent camouflage recognition, or improved segmentation is not supported.

\vfill
{\small All headline results are supported by Tables~\ref{tab:classification}--\ref{tab:pysod} and the exact saved artifacts mapped in Appendix~\ref{app:provenance}. The six supplementary examples are illustrations selected by outcome, not a new performance estimate.}
\end{titlepage}''')
t=t.replace('All project-result numbers in this report are tied to the exact saved artifact paths printed in tables, captions, or adjacent source notes. No SINet-V2 or LaMa inference was run to prepare this report. Figures were assembled from saved metrics, saved images, and saved masks.','All project-result numbers are traceable through Appendix~\\ref{app:provenance}; full precision is retained in the accompanying CSV/JSON files. A supplementary evaluation exported previews for six previously selected examples, reusing all six saved reconstruction/residual caches without new model inference.')
t=t.replace(r'\tableofcontents\n\clearpage','') if False else t
t=t.replace('\\tableofcontents\n\\clearpage',r'''\begingroup\small\setcounter{tocdepth}{1}
\begin{multicols}{2}\tableofcontents\end{multicols}\endgroup''')
t=t.replace(r'\input{tables/confusion.tex}','')
t=t.replace(r'Tables~\ref{tab:classification} and~\ref{tab:confusion}',r'Table~\ref{tab:classification} and Appendix~\ref{app:counts}')
a=t.index(r'\subsection{Calibration journey and pitfalls}');b=t.index(r'\input{tables/calibration_journey.tex}',a)
t=t[:a]+r'''\subsection{Calibration journey and pitfalls}
Table~\ref{tab:calibration} preserves the complete progression from the initial composite through source-matched diagnostics, the feature-only transfer failure, and the final USC12K band. The apparent composite success did not survive source matching, and the CO--NOCOD inversion rules out treating one binary score direction as universal. The final train-derived band and disjoint holdout support further investigation, but their small size and persistent source/class association do not establish source-independent generalization (Appendix~\ref{app:provenance}).

'''+t[b:]
a=t.index(r'\subsection{Realistic and open-world COD}');b=t.index(r'\begin{table}[H]',a)
t=t[:a]+r'''\subsection{Realistic and open-world COD}
Table~\ref{tab:literature} compares OPC16K/OPCNet, USC12K/USCNet, and RCOD with our frozen, post-hoc verifier. Their tasks, outputs, training, and evaluation protocols differ, and no matched head-to-head experiment was conducted; the project therefore supports no superiority claim over these methods. RCOD's box annotations were downloaded but were not used as pixel-mask evaluation targets. \citep{opc16k,usc12k,rcod}

\subsection{Counterfactual and reconstruction-based methods}
CFCamo's paired detect-or-abstain training, Visual Jenga's object-dependency counterfactuals, and RUN's learned background extraction share aspects of reconstruction reasoning while answering distinct questions, as detailed in Table~\ref{tab:literature}. Their reported metrics cannot be ranked against this project's three-way USC12K accuracy without a harmonized protocol. The surveys reinforce why target-present segmentation alone cannot establish open-world presence decisions or source-independent transfer. \citep{cfcamo,visualjenga,run,xiaoSurvey,zhaoSurvey}

'''+t[b:]
t=t.replace('Directly aligned in open-world intent; substantially more explicit counterfactual supervision and policy training.','CF-COD Pair Accuracy 80.0--90.8\\%; a different paired benchmark/metric, with explicit counterfactual supervision and policy training.')
t=t.replace('Same open-world motivation; our work uses USC12K and a frozen baseline, without an OPC16K head-to-head.','Hierarchical existence reasoning; our accuracy 0.7122 is on USC12K, not OPC16K; its release was unavailable at acquisition.')
t=t.replace('Same dataset context; USCNet is end-to-end, our band is post-hoc and feature-only.','End-to-end inter/intra-sample prompt queries; our feature-only band does not implement its aspect-relation module.')
t=t.replace('Shares use of inpainting counterfactuals, but addresses scene dependency rather than camouflage presence.','Progressive object removal probes dependency and scene coherence, rather than camouflage presence.')
t=t.replace('Reconstruction is integrated into a trained segmentation model rather than a post-hoc residual decision.','Reversible RGB/mask unfolding integrates reconstruction-oriented background extraction into a trained segmenter.')
# Compact captions and consistent figure sizes.
caps={
'fig:pipeline':'Counterfactual reconstruction and feature-band verification pipeline.',
'fig:confusion':'Full-split confusion matrix with true rows and predicted columns.',
'fig:pysod':'PySODMetrics comparison for raw, verified, and refined CO masks.',
'fig:qualitative':'Six real examples traced through the reconstruction and refinement pipeline.',
 'tab:literature':'Research scope and comparability of recent related work.'}
for label,cap in caps.items():
    t=re.sub(r'\\caption\{[^\n]*\}\n\\label\{'+re.escape(label)+r'\}',lambda m:'\\caption{'+cap+'}\n\\label{'+label+'}',t)
t=t.replace(r'width=0.78\textwidth',r'width=0.53\textwidth')
t=t.replace(r'width=0.92\textwidth',r'width=0.94\textwidth')
a=t.index('Figure~\\ref{fig:qualitative} shows');b=t.index(r'\begin{figure}',a)
t=t[:a]+r'''Figure~\ref{fig:qualitative} retains the same six previously selected examples: one correct and one incorrect class decision per class. A supplementary evaluation with \texttt{--save-previews} reused all six full-run caches; its recorded evaluation-loop time was 2.589 seconds, excluding startup. The displayed stages are original, candidate overlay, LaMa reconstruction, feature-residual heatmap (fixed scale), and the refined final mask; orange contours show CO ground truth. Images are letterboxed consistently without cropping or geometric distortion. These outcome-selected illustrations are not representative performance estimates; their decisions, scores, and mask areas were checked against the saved full run (Appendix~\ref{app:provenance}).

'''+t[b:]
old='The requested qualitative grid was possible because the saved manifest, raw candidate masks, and source images exist; no saved full-run preview files were present, so the figure was assembled from those real files.'
t=t.replace(old,'The full run did not save previews. The authorized six-image supplement now exports original, candidate, reconstructed-background, residual, final-mask, and CO ground-truth stages using the original full-precision cache; no full-split evaluation was repeated.')
# Preserve tables; remove source blocks; apply numeric formatting and headers.
for f in (H/'tables').glob('*.tex'):
    s=f.read_text();s=re.sub(r'\\src\{[^}]*\}','',s)
    s=re.sub(r'(?<![A-Za-z0-9])0\.\d{5,}',lambda m:format(float(m[0]),'.4g'),s)
    s=s.replace('1.0000000','1.000').replace('22,164.16736839991',r'$2.216\times10^4$')
    s=s.replace(r'\begin{tabularx}{\textwidth}{p{3.6cm}p{2.5cm}X}',r'\begin{tabularx}{\textwidth}{L{3.6cm}L{2.2cm}Y}')
    s=s.replace('Proposal question or assumption & Assessment & Evidence-based answer',r'\textbf{Proposal question} & \textbf{Assessment} & \textbf{Evidence-based answer}')
    s=s.replace('Measure & Estimate & Bootstrap 95\\% CI',r'\textbf{Measure} & \textbf{Estimate} & \textbf{95\% CI}')
    s=s.replace('Condition & Class & False acceptance rate & Mean mask area fraction',r'\textbf{Condition} & \textbf{Class} & \textbf{False acceptance $\downarrow$} & \textbf{Area fraction $\downarrow$}')
    s=s.replace('Condition & S-alpha & Weighted F-beta & Adaptive E-measure & MAE',r'\textbf{Condition} & $\boldsymbol{S_\alpha\uparrow}$ & $\boldsymbol{F_\beta^w\uparrow}$ & $\boldsymbol{E_{adp}\uparrow}$ & \textbf{MAE $\downarrow$}')
    s=s.replace('True $\\backslash$ predicted & BG & CO & NOCOD',r'\textbf{True / predicted} & \textbf{BG} & \textbf{CO} & \textbf{NOCOD}')
    capnotes={
    'classification.tex':('Feature-band classification on the official evaluation split.','Class-stratified 95\\% intervals; unavailable intervals are marked ``Not reported.'' Higher is better except false acceptance. Full precision: Appendix~\\ref{app:provenance}.'),
    'pysodmetrics.tex':('CO segmentation evaluated consistently with PySODMetrics.','PySODMetrics 1.6.2; identical CO ground truths for all conditions. $E_{adp}$ is sample-based adaptive E-measure.'),
    'negative_mask_area.tex':('Negative acceptance and predicted-mask area by condition.','Raw acceptance means any nonempty candidate. Area is the mean per-image mask fraction; classification decisions are identical for verify and refine.'),
    'confusion.tex':('Exact full-split confusion counts.','True rows; predicted columns. Source mapping: Appendix~\\ref{app:provenance}.'),
    'proposal_questions.tex':('Evidence-based answers to the original proposal.','Assessment refers to this evaluated split and does not establish source-independent transfer.')}
    if f.name in capnotes:
        cap,note=capnotes[f.name];s=re.sub(r'\\caption\{[^\n]*\}',lambda m:'\\caption{'+cap+'}',s)
        s=s.replace(r'\end{table}',r'\par\smallskip{\footnotesize '+note+'}\n\\end{table}')
    f.write_text(s)
# Readable mixed text table and concise footnote.
t=t.replace(r'{p{2.8cm} p{4.0cm} X}',r'{L{2.5cm} L{4.1cm} Y}')
t=t.replace('Work & Main method/question & Relation to this project',r'\textbf{Work} & \textbf{Method/question} & \textbf{Relation to this project}')
t=t.replace(r'\caption{Research scope and comparability of recent related work.}',r'\small\caption{Research scope and comparability of recent related work.}')
# Appendices follow bibliography, counts at the very end.
t=t.replace(r'\bibliographystyle{plainnat}',r'\FloatBarrier\begingroup\small\setlength{\bibsep}{2pt}\bibliographystyle{plainnat}')
t=t.replace(r'\bibliography{references}',r'''\bibliography{references}\endgroup
\appendix
\section{Data Provenance}\label{app:provenance}
\input{tables/provenance.tex}
\section{Exact confusion counts}\label{app:counts}
\input{tables/confusion.tex}''')
p.write_text(t,encoding='utf-8')
# Full-precision metrics are local to the report package, with explicit upstream sources.
F=Path('data/experiments/usc12k/full/feature_band_lama_full')
post=json.loads((F/'posthoc_analysis/posthoc_summary.json').read_text())
verify=json.loads((F/'feature_band_verify/summary.json').read_text())
fp={'source_root':str(F),'classification':post['classification'],'negative_mask_area':{k:v['negative_predicted_mask_area_fraction'] for k,v in post['mask_analysis']['segmentation_and_mask_area'].items()},'binary_accuracy':verify['co_vs_negative_accuracy'],'negative_rejection_accuracy':verify['negative_scene_rejection_accuracy'],'runtime':{k:v for k,v in verify.items() if 'runtime' in k},'bootstrap_replicates':post['bootstrap_replicates'],'seed':post['seed']}
(H/'tables/full_precision_results.json').write_text(json.dumps(fp,indent=2))
shutil.copyfile(F/'posthoc_analysis/pysodmetrics_three_condition_table.csv',H/'tables/pysodmetrics_full_precision.csv')
print('Revised report and tables; full precision preserved locally.')
