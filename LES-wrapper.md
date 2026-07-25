# LES-wrapper (ppiBTEP): Learning Efficiency Score Evaluation

## Overview

The **LES-wrapper** automates evaluation of model trainability across ppiBTEP
(SiameseBTPE) training checkpoints. It runs inference on PRS (Positive Reference
Set) and RRS (Random Reference Set) sequence-pair files at each per-epoch
checkpoint, computes ROC metrics, and derives integrated learning-efficiency
scores.

This is the ppiBTEP port of the LES-wrapper family
([ppiGPLM](https://github.com/kouroshSA/ppiGPLM),
[ppiDCE](https://github.com/kouroshSA/ppiDCE),
[ppiYYD](https://github.com/kouroshSA/ppiYYD)). The evaluation logic (PRS/RRS →
`Probability_Friends` → ROC → AUC/Best-F1 → LES) is identical; only the
model-specific glue differs. **Its outputs match ppiGPLM's `LES-wrapper_v2.py`**
— see [Differences from the other wrappers](#differences-from-the-other-wrappers).

## What is LES?

LES (Learning Efficiency Score) is the **area under the metric-vs-epoch curve**.
Unlike metrics that capture only final performance, LES summarizes the entire
learning trajectory:

- **LES-AUC**: Area under the AUC trajectory curve
- **LES-F1**: Area under the Best-F1 trajectory curve

Epochs are normalized to `[0, 1]` before integration, so LES values are
comparable across runs of different length. Higher LES indicates faster, more
consistent learning across training.

> **v2 change (adopted here):** the optimal-F1 **threshold** metric is *not*
> reported. For non-discriminating controls the best-F1 threshold collapses
> toward 0 ("predict everything positive"), so it added noise. Dropped
> throughout: `trajectory_Threshold`, `LES-Threshold`, the `Best_F1_Threshold`
> summary column, the manifest `Threshold` entry, and the threshold panel of the
> combined figure.

## Workflow

For each checkpoint the wrapper:

1. Runs `inference_ppiBTPE_2GPU.py` on the PRS and RRS files
2. Extracts the positive-class probability (`Probability_Friends`) for every pair
3. Combines PRS and RRS probabilities into a single file for ROC analysis
4. Draws a per-checkpoint probability-distribution plot (PRS vs RRS violins)
5. Computes AUC and Best-F1 and renders the ROC curve
6. Aggregates per-checkpoint results into a summary table
7. Plots metric trajectories and probability-distribution summaries across epochs
8. Computes LES for AUC and Best-F1

## Installation

Use the same `esm` environment as ppiDCE — ppiBTEP's `requirements.txt` is
identical:

```bash
conda activate esm
pip install -r requirements.txt   # numpy, scikit-learn, matplotlib, pandas, torch, transformers
```

## Basic Usage

```bash
python LES-wrapper.py \
    --checkpoint_dir ROC_Checkpoints \
    --prs_file MED4_PRS_100.csv \
    --rrs_file MED4_RRS_100.csv \
    --output_dir LES_results_MED4 \
    --num_layers 12 \
    --include_final
```

`--num_layers` **must match the value used at training time** — the inference
script rebuilds the SiameseBTPE config with that many transformer layers before
loading the checkpoint, so a mismatch causes a state-dict load error. Default is
`12` (the README training configuration).

`--include_final` additionally evaluates `ppiBTPE_final.pth` (plotted after the
last numbered epoch; it is excluded from the LES integral so it does not distort
the area).

## Input File Format

PRS and RRS files are CSVs read by `inference_ppiBTPE_2GPU.py` — only the first
two columns (`seq1`, `seq2`) are used; any third `label` column is ignored. The
wrapper assigns labels itself: every PRS pair is positive (1), every RRS pair is
negative (0).

> **Header note:** `inference_ppiBTPE_2GPU.py` reads input with pandas' default
> behavior, which treats the **first row as a header**. Give each PRS/RRS file a
> `seq1,seq2` header row; otherwise the first sequence pair is silently consumed
> as the header and dropped. The shipped `MED4_PRS_100.csv` / `MED4_RRS_100.csv`
> are headerless — add a header line if you need all 100 pairs scored.

## Common Patterns

### Multi-GPU inference

ppiBTEP's inference script supports DataParallel across GPUs:

```bash
--device cuda:0,1     # run inference on GPUs 0 and 1
```

### Selecting specific checkpoints

```bash
# Only epochs 5, 10, 15, 20
--checkpoint_pattern "ppiBTPE_epoch_[51]*0.pth"

# Every epoch (default)
--checkpoint_pattern "ppiBTPE_epoch_*.pth"
```

### Re-computing metrics without re-running inference

```bash
python LES-wrapper.py ... --num_layers 12 --skip_inference
```

reuses the existing `*_probabilities.csv` files. Add `--no_plots` to skip the
trajectory / distribution figures when you only need the summary CSV.

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_dir` | *(required)* | Directory containing `ppiBTPE_epoch_*.pth` |
| `--prs_file` | *(required)* | Positive Reference Set CSV (`seq1,seq2[,label]`) |
| `--rrs_file` | *(required)* | Random Reference Set CSV (`seq1,seq2[,label]`) |
| `--output_dir` | `LES_results` | Directory for all output files |
| `--checkpoint_pattern` | `ppiBTPE_epoch_*.pth` | Glob to select per-epoch checkpoints |
| `--include_final` | False | Also evaluate `ppiBTPE_final.pth` |
| `--inference_script` | *(auto)* | Path to `inference_ppiBTPE_2GPU.py` (defaults alongside this wrapper) |
| `--model_config` | `facebook/esm1b_t33_650M_UR50S` | ESM tokenizer/config source |
| `--num_layers` | `12` | Transformer layers used at training time (config override) |
| `--batch_size` | `4` | Inference batch size |
| `--max_length` | `1024` | Max token length per sequence |
| `--device` | `cuda` | `cpu`, `cuda`, or multi-GPU `cuda:0,1` |
| `--skip_inference` | False | Reuse existing probability CSVs |
| `--no_plots` | False | Skip trajectory / distribution plots |
| `--color_threshold` | False | Color the ROC curve by decision threshold and add a colorbar |

## Figures

All PNGs are written at **publication quality** (600 dpi, tight bounding box,
enlarged fonts, heavier axis lines), and every PNG now has a companion vector
**`.pdf`** at the same path. Individual ROC plots annotate **AUC and Best F1**;
by default the ROC curve is a single color with no threshold colorbar — pass
`--color_threshold` to render the threshold-colored curve.

The probability-distribution figures show `Probability_Friends` for PRS (blue,
positives) vs RRS (red, negatives) as violins + jittered points, y-axis fixed to
`[0, 1]`. A discriminating model keeps PRS high and RRS low.

## Output Structure

```
LES_results_MED4/
├── epoch_1/
│   ├── PRS_epoch1_probabilities.csv        # full inference_ppiBTPE_2GPU.py output
│   ├── RRS_epoch1_probabilities.csv
│   ├── combined_probabilities_epoch1.csv   # PRS,RRS positive-prob columns for ROC
│   ├── prob_dist_epoch1.png / .pdf         # PRS-vs-RRS distribution
│   └── ROC_epoch1.png / .pdf
├── epoch_2/ ...
├── epoch_final/ ...                        # only with --include_final
├── trajectory_AUC.png / .pdf
├── trajectory_F1.png / .pdf
├── trajectory_combined.png / .pdf          # 1x2 AUC + Best-F1 (no threshold panel)
├── summary_prob_distributions.png / .pdf           # one panel per checkpoint
├── summary_prob_distributions_combined.png / .pdf  # all PRS then all RRS, one axes
├── summary_table.csv
├── manifest.json
└── README.md                               # legend for the analysis-level plots
```

`summary_table.csv` has columns `checkpoint, epoch, AUC, Best_F1, PRS_samples,
RRS_samples` plus a final LES row. `manifest.json` records full run metadata
(timestamp, inputs, model config, num_layers, per-checkpoint results, and LES
scores).

> **Single-checkpoint runs.** With only one matching checkpoint the wrapper does
> the per-checkpoint analysis (probabilities, ROC, distribution plot) but skips
> LES, the trajectory plots, and the distribution summaries — these need ≥ 2
> checkpoints.

## Differences from the other wrappers

| Aspect | ppiGPLM `v2` | ppiDCE | ppiBTEP (this wrapper) |
|--------|--------------|--------|------------------------|
| Checkpoints | `ckpt_*.pt` (iterations) | `ppiDCE_epoch*.pth` | `ppiBTPE_epoch_*.pth` (underscore) + `ppiBTPE_final.pth` |
| Trajectory x-axis | iteration | epoch | **epoch** |
| Inference engine | `sample_fasta…3f.py` | `inference_ppiDCE.py` | `inference_ppiBTPE_2GPU.py` (**requires `--num_layers`**; supports `cuda:0,1`) |
| Positive-class prob | `Probability_of_1` → `[-2]` | `prob_1` (last col) | `Probability_Friends` (**2nd-to-last**; `Probability_Enemies` last) |
| Output shape | v2 (no threshold, PDFs, prob-dist plots, README) | v2 | **v2** (matches ppiGPLM) |

Two model-specific notes for ppiBTEP:

- **Score meaning.** `Probability_Friends` is a genuine 2-class softmax over
  `{enemies, friends}`, so `Probability_Friends + Probability_Enemies = 1` exactly
  — unlike ppiGPLM's whole-vocabulary language-model softmax where `P(1)` and
  `P(0)` need not sum to 1. The `README.md` written into each output dir explains
  this.
- **Layer count is load-critical.** The wrapper threads the required `--num_layers`
  through to the inference script, which rebuilds the SiameseBTPE config with that
  many transformer layers before loading the checkpoint.
