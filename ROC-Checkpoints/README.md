# ppiBTEP ROC / LES trajectory checkpoints

**Per-epoch training checkpoints used to trace the model's classification
performance over training — the inputs to the Learning Efficiency Score (LES)
and ROC analysis produced by `LES-wrapper.py`.**

Unlike [`checkpoints/`](../checkpoints/), which holds the single best checkpoint
([`ppiBTPE_epoch_4.pth`](../checkpoints/ppiBTPE_epoch_4.md)) used for *screening*
the MED4 interactome, this set captures the **training trajectory**
(epochs 1–30) so that ROC-AUC, Best-F1, and the optimal-F1 threshold can be
computed at every epoch and integrated into a single LES per metric (the area
under the metric-vs-epoch curve).

## Provenance

| | |
|---|---|
| Model | ppiBTEP (Siamese / twin-branch ESM-1b-inspired transformer, trained from scratch) |
| Architecture | 12 transformer layers |
| Training run | `out_june17_MED4_l12` (learning rate 1e-5) |
| Epochs | 1–30 (one checkpoint per epoch) |
| File size | ~913 MB each (`ppiBTPE_epoch_{N}.pth`) |
| Training set | `train_MED4_ppiBTEPM-pseudo_Int_combo1-2-3.csv` (≈13,008 pairs, pre-clean — see note below) |
| Validation set | `val_MED4_100_Y2H-RND_ppiBRTPM.csv` |

## Download (Hugging Face)

These checkpoints are large (~28 GB total, 30 × ~913 MB) and are hosted on
Hugging Face rather than in this Git repository. They live in the
`ROC-Checkpoints/` folder of the model repo
[kouroshSA/ppiBTEP](https://huggingface.co/kouroshSA/ppiBTEP/tree/main/ROC-Checkpoints).

```bash
# Download just the ROC-Checkpoints folder into ./ROC-Checkpoints
hf download kouroshSA/ppiBTEP --repo-type model \
    --include "ROC-Checkpoints/*" --local-dir .
```

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="kouroshSA/ppiBTEP",
    repo_type="model",
    allow_patterns="ROC-Checkpoints/*",
    local_dir=".",
)
```

## Intended use

Reproducing the LES / ROC analysis for ppiBTEP. Point `LES-wrapper.py` at this
folder and supply a PRS (Positive Reference Set) and RRS (Random Reference Set):

```bash
python LES-wrapper.py \
    --checkpoint_dir ROC-Checkpoints \
    --prs_file MED4_PRS-RRS/PRS-V3-1.csv \
    --rrs_file MED4_PRS-RRS/RRS-V3-1.csv \
    --num_layers 12 \
    --output_dir LES_results --include_final
```

The wrapper runs `inference_ppiBTPE_2GPU.py` at each checkpoint (requires
`--num_layers 12` to rebuild the SiameseBTPE config; `--model_config
facebook/esm1b_t33_650M_UR50S` is config-only — weights come from these
checkpoints, not the HF ESM-1b release), computes ROC-AUC / Best-F1 / optimal
threshold per epoch, and integrates them into LES values.

## Notes

- These checkpoints were produced **before** the PRS/RRS de-overlapping pass on
  `train.csv`. Approximately 608 of the 13,008 training rows (4.67 %) overlap
  with the PRS+RRS evaluation pairs in either orientation. Treat metrics on
  those pairs accordingly.
