# OAC feature-selection benchmark (v2)

A ground-truth benchmark for feature selection in disease transcriptomics, accompanying
*"Classification accuracy is not selection quality: a ground-truth benchmark shows that
disease-gene selection does not transfer across transcriptomic cohorts"* (Chen & Lee,
submitted to GigaScience).

**Findings.** (1) Case–control classification **AUC saturates** — random 100-gene panels
classify as well as supervised panels across 3 tissues / 3 diseases, so AUC cannot validate
a selection. (2) **Disease-attributable variance (DAV)** discriminates where AUC does not and,
crucially, can be evaluated out-of-sample. (3) Unsupervised deep-autoencoder selection is
**not disease-aware** (recovers true genes only marginally above a random-ranking null on known
ground truth). (4) Gene selection **does not transfer across cohorts** — DAV collapses in
independent cohorts while AUC stays high — and a **within-cohort split-half control** shows the
collapse is cohort heterogeneity, not small-sample winner's curse; in leave-one-cohort-out
evaluation **no method (simple or deep) exceeds a random baseline**.

## Install
```bash
pip install -r requirements.txt        # numpy scipy pandas scikit-learn torch matplotlib openpyxl
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
CUDA is used automatically if present; otherwise everything runs on CPU (slower).

## Data
Place the six public GEO datasets in `data/` (exact names in [`data/README.md`](data/README.md)).
Six datasets → eight case–control contrasts. Data are not redistributed here.

## Run
```bash
python run_all.py --fast                     # quick CPU smoke test
python run_all.py --steps repro,f2,f3,f4,splithalf,confounder,f5,figs \
                  --seeds 50 --epochs 500 --synthG 0 --device cuda   # submission-scale
```
Steps: `repro` (3-cohort validation) · `f2` (semi-synthetic ground truth) ·
`f3` (cross-cohort transfer) · `f4` (leave-one-cohort-out **with Random & HighVar baselines** +
significance test) · `splithalf` (within-cohort winner's-curse control) ·
`confounder` (real-data sex-confounder capture) · `f5` (multi-disease panel) · `figs`.

## Outputs (`results/`)
`repro.json` · `f2ci_A.json`,`f2ci_B.json` → `Figure1_ground_truth.png` ·
`f5_panel.json` → `Figure2_universality.png` ·
`f3_cartilage.json`,`f3_synovium.json`,`splithalf.json` → `Figure3_transfer.png` ·
`f4_xc.json` → `Figure4_consensus.png` · `confounder_real.json`.

## Code
`pipeline.py` — loaders, moderated-t DE, DAV, selectors, cross-validated AUC.
`methods.py` — DFS-AE, VAE, PERSIST-style, CADA-AE, XC-CADA-AE (GPU-ready).
`run_all.py` — one-command orchestrator (incl. the review-driven analyses).
`figures.py` — regenerates Figures 1–4.

## Reproducibility / DOME-ML
Fixed seeds; `repro` checks the three OA cohorts against internal reference values before any
downstream analysis. Architectures, hyperparameters and the DOME-ML reporting summary are in the
manuscript Methods; compute is CPU for the reproduction/real-data metrics and GPU-ready (PyTorch)
for the deep selectors. All intervals are 95% CIs over the stated seeds/folds.

## License & citation
MIT (`LICENSE`; OSI-approved). Cite the article and the archived software (`CITATION.cff`;
`.zenodo.json` controls the Zenodo record — fill ORCID/GitHub before release). Zenodo DOI to be
assigned on GitHub release.

## AI-assisted development
Parts of the code and manuscript were drafted with an AI assistant under author supervision;
all results were verified by the authors.
