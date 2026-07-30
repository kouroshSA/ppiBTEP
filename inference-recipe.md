# ppiBTEP V3 inference / LES recipe

How to score the trained ppiBTEP V3 checkpoints on the PRS/RRS reference sets and
produce the Learning-Efficiency-Score (LES) analysis.

## Match the training config (required)

The V3 checkpoints are **6-layer** and were trained at **max_length 1024**. Every
inference / LES run must use the same, or it breaks:

- `--num_layers 6` — the inference script rebuilds the Siamese config with this
  many layers; a mismatch (e.g. the old default 12) raises a contact-head size
  error and cannot load the checkpoint.
- `--max_length 1024` — a from-scratch model only trained position embeddings up to
  its training length; evaluating longer feeds it untrained positions.

`LES-wrapper.py` now **defaults to `--num_layers 6 --max_length 1024`**, so the
plain command is correct for V3; pass explicit values only for other models.

## Run LES on a checkpoint directory

```bash
PY=/home/ksa/anaconda3/envs/esm/bin/python
$PY LES-wrapper.py \
    --checkpoint_dir results_V3-1/model \
    --prs_file V3_PRS-RRS/PRS-RRS/PRS-V3-1.csv \
    --rrs_file V3_PRS-RRS/PRS-RRS/RRS-V3-1.csv \
    --output_dir LES_Results_V3/V3-1/LES_regular \
    --num_layers 6 --max_length 1024 --include_final
```

Per checkpoint it writes `epoch_<N>/` with the probability CSVs, a PRS-vs-RRS
probability-distribution **violin** (`prob_dist_epoch<N>.png`) and (for real
positive sets) `ROC_epoch<N>.png`; across checkpoints it writes the AUC / Best-F1
trajectories, the `summary_prob_distributions*` violins, `summary_table.csv`
(per-epoch AUC/Best-F1 + an LES row), and `manifest.json`.

## Reference-set conditions

`V3_PRS-RRS/` provides, per replicate:

| condition | PRS / RRS files |
|---|---|
| regular | `PRS-RRS/{PRS,RRS}-V3-k.csv` |
| homodimer-depleted | `PRS-RRS_no_homodimers/{PRS,RRS}-V3-k.csv` |
| homodimers-only | `PRS-RRS_homodimers_only/{PRS,RRS}-V3-k.csv` |
| ps1 / ps2 / ps1-ps2 random | `random_controls/{PRS,RRS}-V3-k_{ps1,ps2,ps1-ps2}_random.csv` |

`run_les_all_conditions.sh` runs all six conditions for the given replicates
(using the in-repo `V3_PRS-RRS/` sets; override with `REF=...`), or use
`eval_per_epoch.sh` to score each epoch as it lands (see `campaign.md` §2b).
`run_composites_all_conditions.sh` then builds the cross-replicate composites
(`make_composite_les.py` trajectories + violins, `composite_roc_btep.py` ROC for
the real-positive conditions).

## Random controls: AUC-ROC and Best-F1 are excluded

For the random-substituted controls (`ps1_random`, `ps2_random`,
`ps1-ps2_random`) **neither** reference file contains true positives — both the
"PRS" and the "RRS" are random pairs — so ROC-AUC and Best-F1, which measure
positive-vs-negative ranking, are **not meaningful** and are **not reported**.

`LES-wrapper.py` handles this automatically:

- It **auto-enables `--no_metrics`** when the PRS/RRS filenames contain `random`
  (or pass `--no_metrics` explicitly).
- In that mode it **skips ROC-AUC, Best-F1, the LES integral, the ROC plots, and
  the AUC/F1 trajectory plots**; the `summary_table.csv` AUC/Best_F1 columns are
  left blank and there is no LES row / manifest LES entry.
- It **still produces** the per-checkpoint and summary **probability-distribution
  violins** and the raw probability CSVs — which is the meaningful read-out for a
  control: a model that relies on genuine pair information should show the two
  random distributions collapse together (no separation).

Note: the inference script itself (`inference_ppiBTPE_2GPU.py`) only writes
per-pair probabilities and needs no change — the metric exclusion lives in the
LES-wrapper, which is where AUC/F1/LES are computed.
