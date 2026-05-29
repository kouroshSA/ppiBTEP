# ppiBTPE_epoch_4.pth

**Checkpoint used for screening low-confidence Y2H pairs in the *Prochlorococcus marinus* MED4 interactome.**

## Provenance

| | |
|---|---|
| Model | ppiBTEP (Siamese / twin-branch ESM-1b-inspired transformer, trained from scratch) |
| Architecture | 12 transformer layers |
| Epoch | 4 |
| File size | ~913 MB |
| Training run | `out_june17_MED4_l12` (June 17, 2025) |
| Training set | `train_MED4_ppiBTEPM-pseudo_Int_combo1-2-3.csv` (≈13,008 pairs, pre-clean — see note below) |
| Validation set | `val_MED4_100_Y2H-RND_ppiBRTPM.csv` |

## Intended use

Inference / screening of candidate MED4 protein–protein interactions that
were originally flagged as **low-confidence Y2H hits**. The model is run on
each candidate pair and its softmax probability is used (in concert with the
other tri-model components, ppiDCE and ppiGPLM) to retain or discard the pair.

## Notes

- This checkpoint was produced **before** the PRS/RRS de-overlapping pass on
  `train.csv` (see [`MED4-PPIs-low-confidence_ppiTEPM_prompts.csv`](../MED4-PPIs-low-confidence_ppiTEPM_prompts.csv) and the cleaned
  `train.clean.csv` / `train.clean2x.csv` companions). Approximately 608 of
  the 13,008 training rows (4.67 %) overlap with the PRS+RRS evaluation pairs
  in either orientation. Treat metrics on those pairs accordingly.
- Loading: use `train_ppiBTPE3b.py` / `inference_ppiBTPE_2GPU.py` from the
  parent repo with `--num_layers 12` and `--model_config
  facebook/esm1b_t33_650M_UR50S` (config-only; weights are loaded from this
  checkpoint, not from the HF ESM-1b release).
