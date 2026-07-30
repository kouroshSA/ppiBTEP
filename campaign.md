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

## 2. LES across all six conditions

```bash
./run_les_all_conditions.sh            # replicates 1 2 (or pass a list)
```

For each replicate it runs the LES-wrapper on all six conditions — `regular`,
`no_homodimers`, `homodimers_only`, `ps1_random`, `ps2_random`,
`ps1-ps2_random` — writing `LES_Results_V3/V3-k/LES_<condition>/`, using the
in-repo `V3_PRS-RRS/` sets (override the reference root with `REF=...`). The three
real-positive conditions (`regular`, `no_homodimers`, `homodimers_only`) report
ROC-AUC / Best-F1 / LES; the three random-control conditions **auto-skip
ROC-AUC / Best-F1 / LES** (their filenames contain `random`) and emit only the
probability-distribution **violins** + raw CSVs, since those sets have no true
positives (see `inference-recipe.md`).

### 2b. Per-epoch evaluation (optional)

`run_les_all_conditions.sh` scores a run's checkpoints in one batch after training
finishes. To get per-epoch results *as training progresses* instead, use:

```bash
./eval_per_epoch.sh 1 2 3              # watch replicates; score each epoch as it lands
ONCE=1 ./eval_per_epoch.sh 1 2 3      # single pass over existing checkpoints, then exit
```

It scores each epoch checkpoint **once** as it appears (all six conditions) and
assembles the summaries/trajectories with LES-wrapper `--skip_inference` (no
re-inference). Output layout is identical to step 2, so step 3 (composites)
consumes it unchanged. Paths are configurable via env (`REF`, `RESULTS`, `OUT`,
`PY`, …); see the header.

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
| LES (6 conditions) | `run_les_all_conditions.sh` (batch, after a run) or `eval_per_epoch.sh` (per epoch, live) | `LES_Results_V3/V3-k/LES_<condition>/` |
| composite/ensemble | `run_composites_all_conditions.sh` → `make_composite_les.py`, `composite_roc_btep.py` | `LES_Results_V3/ensemble/<condition>/` |

Envs: LES/inference use the `esm` conda env (GPU); the composite scripts use the
`gpt` env (numpy/matplotlib/sklearn).
