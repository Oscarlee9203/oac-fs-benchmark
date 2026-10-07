# OAC feature-selection benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22188862.svg)](https://doi.org/10.5281/zenodo.22188862)

Code for *"A ground-truth benchmark and an out-of-sample metric for evaluating gene selection
in transcriptomics of osteoarthritis and other diseases"* (Chen & Lee, submitted).

The benchmark scores gene-selection methods in two ways: against a semi-synthetic ground truth
in which the true disease genes and a confounder programme are known by construction, and on
eight real case–control contrasts (six public GEO datasets; three tissues, three diseases). The
evaluation metric is **disease-attributable variance (DAV)** — the median squared point-biserial
correlation of the selected genes with disease status — which, unlike classification accuracy,
can be measured on samples that were not used for selection.

## Install

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the article (Python 3.13). Results that do not
involve an autoencoder (supervised DE, high-variance, PCA, random, consensus, split-half, DAV,
AUC) are deterministic. Autoencoder-based results (DFS-AE, PERSIST, CADA-AE, XC-CADA-AE) are
seeded but change slightly with the PyTorch version and hardware; use the pinned versions to
reproduce the article's numbers. CUDA is used automatically if present.

## Data

Create a `data/` folder next to the scripts and place these files in it, with these exact names
(all are downloadable from the corresponding NCBI GEO records; data are not redistributed here):

| File | GEO record |
|---|---|
| `GSE114007_OA_normalized.counts.txt.gz` | GSE114007 (supplementary file) |
| `GSE114007_normal_normalized.counts.txt.gz` | GSE114007 (supplementary file) |
| `GSE117999_normalized_matrix.xlsx` | GSE117999 (supplementary file) |
| `GSE57218_series_matrix.txt.gz` | GSE57218 (series matrix) |
| `GSE57218_RAW.tar` | GSE57218 (supplementary file; provides the probe annotation) |
| `GSE55235_series_matrix.txt.gz` | GSE55235 (series matrix) |
| `GSE55457_series_matrix.txt.gz` | GSE55457 (series matrix) |
| `GSE5281_series_matrix.txt.gz` | GSE5281 (series matrix) |

## Run

```bash
python run_all.py            # reproduces every analysis and figure in the article
python run_all.py --fast     # quick smoke test (reduced settings; not the article's numbers)
```

The default settings are the ones used in the article:

| Analysis | Setting |
|---|---|
| Semi-synthetic benchmark (`f2`) | random 3,000-gene subset of the cartilage background (`--synthG 3000`), 10 seeds (`--seeds 10`), deep selectors trained for 250 epochs (`--synth-epochs 250`) |
| Real-data deep selectors (`f3`, `f4`, `confounder`) | 500 epochs (`--epochs 500`); the first XC-CADA-AE version uses half of that |
| Split-half control (`splithalf`) | 40 random stratified splits per cohort |
| Random baselines (`f3`, `f4`) | 50 random panels; in `f4` the same panels are scored in every held-out cohort |
| Sex-confounder analysis (`confounder`) | 5 seeds for the stochastic selectors (`--conf-seeds 5`) |

On a 2-core CPU the full run takes about 40 minutes. It includes the training-length sensitivity
analysis (`f2sens`), which repeats the semi-synthetic benchmark with the deep selectors trained for
80 and 500 epochs.

Steps (`--steps`, comma-separated): `repro` (data-loading check; stops if the three cartilage cohorts
do not match stored reference values) · `f2` (semi-synthetic ground truth) · `f2sens` (its sensitivity to training length) ·
`f3` (cross-cohort transfer) · `f4`
(leave-one-cohort-out with random baseline) · `splithalf` (within-cohort control) · `confounder`
(sex-confounder capture on real data) · `f5` (eight-contrast panel) · `figs`.

## Outputs (`results/`)

| File | Content |
|---|---|
| `repro.json` | data-loading check |
| `f2ci_A.json`, `f2ci_B.json`, `f2_sensitivity.json` → `Figure1.png/.pdf` | ground-truth benchmark and training-length sensitivity |
| `f5_panel.json` → `Figure2.png/.pdf` | accuracy vs DAV on eight contrasts |
| `f3_cartilage.json`, `f3_synovium.json`, `f3_random.json`, `splithalf.json`, `splithalf_random.json` → `Figure3.png/.pdf` | cross-cohort transfer, split-half control and their random-panel floors |
| `f4_xc.json` → `Figure4.png/.pdf` | leave-one-cohort-out against matched random panels |
| `confounder_real.json` | sex-confounder capture |

## Code

`pipeline.py` — loaders, moderated-t differential expression, DAV, classical selectors,
cross-validated AUC. `methods.py` — DFS-AE, PERSIST-style selector, CADA-AE, XC-CADA-AE.
`run_all.py` — orchestrator. `figures.py` — Figures 1–4.

## License and citation

MIT (see `LICENSE`). Please cite the article and the archived software (see `CITATION.cff`).
The release used for the article is v1.1.0
([10.5281/zenodo.23219920](https://doi.org/10.5281/zenodo.23219920)); the Zenodo concept DOI,
which always resolves to the latest version, is
[10.5281/zenodo.22188862](https://doi.org/10.5281/zenodo.22188862).

## AI-assisted development

Parts of the code were written with an AI assistant (Anthropic Claude) under author supervision;
the authors reviewed the code and verified the results.
