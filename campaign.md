# ppiBTEP V3 campaign — LES, composite & ensemble analyses

End-to-end recipe for the ppiBTEP V3 evaluation campaign: train the replicates,
run LES against every reference-set condition, and build the cross-replicate
composite / ensemble figures. See `train-recipe.md` and `inference-recipe.md` for
the per-run details this ties together.

Config throughout: **from-scratch, `--num_layers 6`, `--max_length 1024`** (the V3
checkpoints; the LES-wrapper defaults to both).

## 0. Reference sets

All PRS/RRS conditions are in `V3_PRS-RRS/` (2-column `SEQ1,SEQ2`):
`PRS-RRS/` (regular), `PRS-RRS_no_homodimers/`, `PRS-RRS_homodimers_only/`,
`random_controls/`. Training sets are not committed.

## 1. Train

Per replicate, from scratch (see `train-recipe.md`); checkpoints land in
`results_V3-k/model/` as `ppiBTPE_epoch_{1..10}.pth` + `ppiBTPE_final.pth`.

## 2. LES across all five conditions

```bash
./run_les_all_conditions.sh            # replicates 1 2 (or pass a list)
```

For each replicate it runs the LES-wrapper on all five conditions — `regular`,
`no_homodimers`, `homodimers_only`, `ps1_random`, `ps2_random`,
`ps1-ps2_random` — writing `LES_Results_V3/V3-k/LES_<condition>/`. The three
random-control conditions **auto-skip ROC-AUC / Best-F1 / LES** (their filenames
contain `random`) and emit only the probability-distribution **violins** + raw
CSVs, since those sets have no true positives (see `inference-recipe.md`).

## 3. Composite / ensemble across replicates

```bash
./run_composites_all_conditions.sh     # same replicate list
```

Per condition it stages the replicates' `LES_<condition>` dirs and runs:

- **`make_composite_les.py`** — across-replicate mean AUC/Best-F1 **trajectories**
  (± SD), ensemble LES, pooled PRS-vs-RRS **violins**;
- **`composite_roc_btep.py`** — vertically-averaged **composite ROC** (per-epoch,
  all-epoch overlay with SD bands, AUC-vs-epoch).

Output: `LES_Results_V3/ensemble/<condition>/` (+ `ROC/`). For the random controls
only the violins are meaningful (no AUC/ROC).

## Summary of outputs

| step | script(s) | output |
|---|---|---|
| train | `train_ppiBTPE3b.py` (see `train-recipe.md`) | `results_V3-k/model/ppiBTPE_epoch_*.pth` |
| LES (5 conditions) | `run_les_all_conditions.sh` | `LES_Results_V3/V3-k/LES_<condition>/` |
| composite/ensemble | `run_composites_all_conditions.sh` → `make_composite_les.py`, `composite_roc_btep.py` | `LES_Results_V3/ensemble/<condition>/` |

Envs: LES/inference use the `esm` conda env (GPU); the composite scripts use the
`gpt` env (numpy/matplotlib/sklearn).
