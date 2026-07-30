#!/usr/bin/env python3
"""
train_ppiBTPE3b.py — Train or fine-tune the ppiBTEP / SiameseBTPE model for
protein-protein interaction (PPI) classification.

Architecture
------------
SiameseBTPE: two branches with shared weights, each an ESM-1b encoder
(facebook/esm1b_t33_650M_UR50S by default). Each branch produces a [CLS]
embedding (last_hidden_state[:, 0, :], dim=1280). The two [CLS] embeddings
are concatenated (dim=2560), passed through Dropout(0.1) and a Linear layer
to 2 logits (CrossEntropyLoss; softmax applied at inference).

Modes
-----
- From scratch:       --num_layers N --freeze_layers 0
- Fine-tuning ESM-1b: omit --num_layers, set --freeze_layers >= 1
- Resume:             --checkpoint <path/to/ppiBTPE_epoch_K.pth>

Important
---------
When training from scratch, pass --freeze_layers 0 explicitly. The default is
20, which would freeze most of the model.

Data format
-----------
CSV with columns: seq1, seq2, label
  - label = 1 or 'friends'  → interacting
  - label = 0 or 'enemies'  → non-interacting

Example
-------
    python train_ppiBTPE3b.py \\
        --train_file train.csv \\
        --val_file val.csv \\
        --model_config facebook/esm1b_t33_650M_UR50S \\
        --num_layers 12 \\
        --freeze_layers 0 \\
        --epochs 20 \\
        --batch_size 2 \\
        --learning_rate 1e-5 \\
        --max_length 1024 \\
        --output_dir ./out \\
        --device cuda

Learning-rate schedule
----------------------
Default is a constant --learning_rate. Pass --lr_schedule warmup_cosine to use
the same per-iteration warmup+cosine schedule as ppiYYD (linear warmup 0->peak,
then cosine decay peak->--min_lr), for side-by-side comparable training:

    --lr_schedule warmup_cosine --learning_rate 2e-5 --min_lr 2e-6 --warmup_ratio 0.1
"""
import argparse
import math
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import EsmModel, EsmTokenizer, EsmConfig
from tqdm import tqdm

# Command-line arguments
parser = argparse.ArgumentParser(
    description='Train or fine-tune ppiBTPE Siamese model for PPI prediction.'
)

# Input files
parser.add_argument('--train_file', type=str, required=True,
                    help='Path to the training CSV file.')
parser.add_argument('--val_file', type=str, required=True,
                    help='Path to the validation CSV file.')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='(Optional) Path to a .pth checkpoint to load before training/fine-tuning.')

# Model parameters
parser.add_argument('--model_config', type=str,
                    default='facebook/esm1b_t33_650M_UR50S',
                    help='ESM config name or path for architecture.')
parser.add_argument('--num_labels', type=int, default=2,
                    help='Number of output labels (e.g., 2 for binary classification).')
parser.add_argument('--num_layers', type=int, default=None,
                    help='Total transformer layers to initialize (scratch).')
parser.add_argument('--freeze_layers', type=int, default=20,
                    help='Number of bottom layers to freeze during fine-tuning.')

# Training hyperparameters
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='Learning rate. With --lr_schedule warmup_cosine this is the PEAK '
                         'reached at the end of warmup.')
parser.add_argument('--lr_schedule', type=str, default='constant',
                    choices=['constant', 'warmup_cosine'],
                    help="LR schedule. 'constant' (default) holds --learning_rate. "
                         "'warmup_cosine' ramps 0->peak over the warmup, then cosine-decays "
                         "peak->--min_lr over the rest (per-iteration, mirrors ppiYYD).")
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='warmup_cosine: warmup length as a fraction of total steps '
                         '(default 0.1); ignored if --warmup_steps is given.')
parser.add_argument('--warmup_steps', type=int, default=None,
                    help='warmup_cosine: explicit warmup length in iterations '
                         '(overrides --warmup_ratio).')
parser.add_argument('--min_lr', type=float, default=0.0,
                    help='warmup_cosine: floor the cosine decays to (e.g. 2e-6). Default 0.')
parser.add_argument('--max_length', type=int, default=1024,
                    help='Maximum sequence length for tokenization.')

# Misc
parser.add_argument('--output_dir', type=str, default='./',
                    help='Directory to save checkpoints and final model.')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'],
                    help='Device to run training on.')
parser.add_argument('--seed', type=int, default=None,
                    help='Random seed for reproducibility (Python/torch/CUDA, incl. the '
                         'DataLoader shuffle). Default None = unseeded. The V3 recipe uses '
                         '--seed 42.')
args = parser.parse_args()

# Device setup
if torch.cuda.is_available() and args.device.startswith('cuda'):
    device = torch.device(args.device)
    n_gpu = torch.cuda.device_count()
    print(f"GPUs available: {n_gpu}")
else:
    device = torch.device('cpu')
    n_gpu = 0
    print("Using CPU.")

# Dataset definition
class SiamesePPIDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        seq1 = self.data.iloc[idx,0]
        seq2 = self.data.iloc[idx,1]
        lbl = self.data.iloc[idx,2]
        # map labels
        if isinstance(lbl, str):
            label = 1 if lbl=='friends' else 0
        else:
            label = int(lbl)
        enc1 = self.tokenizer(seq1, truncation=True, padding='max_length',
                               max_length=self.max_length, return_tensors='pt')
        enc2 = self.tokenizer(seq2, truncation=True, padding='max_length',
                               max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids1': enc1.input_ids.squeeze(0),
            'attention_mask1': enc1.attention_mask.squeeze(0),
            'input_ids2': enc2.input_ids.squeeze(0),
            'attention_mask2': enc2.attention_mask.squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Model definition
class SiameseBTPE(nn.Module):
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.esm = EsmModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size*2, num_labels)
    def forward(self, id1, mask1, id2, mask2):
        o1 = self.esm(input_ids=id1, attention_mask=mask1)
        o2 = self.esm(input_ids=id2, attention_mask=mask2)
        p1 = o1.last_hidden_state[:,0,:]
        p2 = o2.last_hidden_state[:,0,:]
        x = torch.cat((p1,p2), dim=1)
        x = self.dropout(x)
        return self.classifier(x)

# Main training loop

def main():
    if args.seed is not None:
        import random as _random
        _random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seed set to {args.seed} (Python/torch/CUDA; DataLoader shuffle deterministic)")

    tokenizer = EsmTokenizer.from_pretrained(args.model_config)
    config = EsmConfig.from_pretrained(args.model_config)
    if args.num_layers is not None:
        config.num_hidden_layers = args.num_layers
        print(f"Using {args.num_layers} layers (override)")

    # datasets + loaders
    train_ds = SiamesePPIDataset(args.train_file, tokenizer, args.max_length)
    val_ds   = SiamesePPIDataset(args.val_file,   tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    model = SiameseBTPE(config, num_labels=args.num_labels)
    # load checkpoint if provided
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    # freeze layers
    total = len(model.esm.encoder.layer)
    to_freeze = min(args.freeze_layers, total)
    for param in model.esm.embeddings.parameters(): param.requires_grad=False
    for layer in model.esm.encoder.layer[:to_freeze]:
        for p in layer.parameters(): p.requires_grad=False
    print(f"Frozen {to_freeze}/{total} layers")

    # device
    model.to(device)
    if n_gpu>1: model = nn.DataParallel(model)

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.learning_rate)
    crit  = nn.CrossEntropyLoss()
    os.makedirs(args.output_dir, exist_ok=True)

    # per-iteration LR schedule (mirrors nanoGPT's get_lr(it) / ppiYYD): linear
    # warmup 0->peak, then cosine decay peak->min_lr. Stepped once per optimizer step.
    scheduler = None
    if args.lr_schedule == 'warmup_cosine':
        total_steps = args.epochs * len(train_loader)
        warmup = (args.warmup_steps if args.warmup_steps is not None
                  else int(round(args.warmup_ratio * total_steps)))
        peak, floor = args.learning_rate, args.min_lr

        def lr_lambda(step):                       # returns lr(step)/peak
            if step < warmup:
                return (step + 1) / (warmup + 1)
            dr = (step - warmup) / max(1, total_steps - warmup)
            dr = min(1.0, dr)
            coeff = 0.5 * (1.0 + math.cos(math.pi * dr))    # 1 -> 0
            return (floor + coeff * (peak - floor)) / peak

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
        print(f"LR schedule: warmup_cosine | peak={peak:g} min={floor:g} "
              f"warmup={warmup} steps ({warmup/len(train_loader):.2f} epoch) "
              f"total={total_steps} steps (per-iteration)")

    for e in range(args.epochs):
        cur_lr = optim.param_groups[0]['lr']
        print(f"Epoch {e+1}/{args.epochs}  (lr={cur_lr:.3e})")
        model.train()
        train_loss=0
        for b in tqdm(train_loader, desc='Train'):
            optim.zero_grad()
            logits = model(b['input_ids1'].to(device), b['attention_mask1'].to(device),
                           b['input_ids2'].to(device), b['attention_mask2'].to(device))
            loss = crit(logits, b['labels'].to(device))
            loss.backward(); optim.step()
            if scheduler is not None:
                scheduler.step()               # per-iteration LR update
            train_loss+=loss.item()
        print(f"Train loss: {train_loss/len(train_loader):.4f}")

        model.eval()
        val_loss, correct, total = 0,0,0
        for b in tqdm(val_loader, desc='Val'):
            with torch.no_grad():
                logits = model(b['input_ids1'].to(device), b['attention_mask1'].to(device),
                               b['input_ids2'].to(device), b['attention_mask2'].to(device))
                loss = crit(logits, b['labels'].to(device))
                val_loss+=loss.item()
                preds=logits.argmax(dim=1)
                correct+=(preds==b['labels'].to(device)).sum().item()
                total+=len(preds)
        print(f"Val loss: {val_loss/len(val_loader):.4f}, Acc: {correct/total:.4f}")

        # save
        path = os.path.join(args.output_dir, f"ppiBTPE_epoch_{e+1}.pth")
        torch.save(model.module.state_dict() if n_gpu>1 else model.state_dict(), path)
        print(f"Saved {path}")

    final = os.path.join(args.output_dir, 'ppiBTPE_final.pth')
    torch.save(model.module.state_dict() if n_gpu>1 else model.state_dict(), final)
    print(f"Saved final model: {final}")

if __name__=='__main__':
    main()
