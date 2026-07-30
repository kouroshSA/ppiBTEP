# ppiBTEP V3 training recipe

The recipe used to train the V3 (MCCV replicate) ppiBTEP models — a Siamese
ESM-1b-architecture PPI classifier trained **from scratch** (shared encoder,
`concat([CLS_A, CLS_B]) -> head`).

## Config

| Setting | Value |
|---|---|
| Encoder | ESM-1b architecture, **from scratch** (not pretrained weights) |
| Layers | `--num_layers 6` (6 transformer layers; **not** the full 33) |
| Freeze | `--freeze_layers 0` (train everything) |
| Max length | `--max_length 512` |
| LR schedule | `--lr_schedule warmup_cosine` — peak `--learning_rate 2e-5`, floor `--min_lr 2e-6`, `--warmup_ratio 0.1` |
| Epochs / batch | `--epochs 10`, `--batch_size 4` |
| Seed | `--seed 42` |
| Model config | `--model_config facebook/esm1b_t33_650M_UR50S` (tokenizer/config only; weights random) |

> **Two values that matter downstream:** the checkpoints are **6-layer** and were
> trained at **max_length 512**. Any inference / LES run must use the same
> (`--num_layers 6 --max_length 512`) or loading fails (layer-count mismatch) /
> results drift (position embeddings the from-scratch model never trained on). The
> LES-wrapper now defaults to both.

## Command (per replicate k)

```bash
PY=/home/ksa/anaconda3/envs/esm/bin/python
$PY train_ppiBTPE3b.py \
    --train_file <depleted_training_set-V3-k.csv> --val_file <val.csv> \
    --num_layers 6 --freeze_layers 0 \
    --epochs 10 --batch_size 4 --max_length 512 \
    --lr_schedule warmup_cosine --learning_rate 2e-5 --min_lr 2e-6 --warmup_ratio 0.1 \
    --seed 42 --output_dir results_V3-k/model
```

Training set format: `SEQ1,SEQ2,label` (3 columns, label `0`/`1`), headerless —
the shared ppiDCE/ppiBTEP format. See `V3_PRS-RRS/` for the matched evaluation
reference sets (training sets are not committed here).

## Outputs

Per epoch: `ppiBTPE_epoch_{1..10}.pth` plus `ppiBTPE_final.pth` (11 checkpoints).
These are what `LES-wrapper.py` scores — see `inference-recipe.md`.
