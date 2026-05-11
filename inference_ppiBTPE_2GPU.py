#!/usr/bin/env python3
"""
inference_ppiBTPE_2GPU.py — Batch inference for ppiBTEP / SiameseBTPE,
supporting single-GPU, multi-GPU (DataParallel), and CPU execution.

Inputs
------
CSV with at least 2 columns: seq1, seq2 (label column, if present, is ignored).

Outputs
-------
CSV with columns: seq1, seq2, Prediction, Probability_Friends, Probability_Enemies

Example (single GPU)
--------------------
    python inference_ppiBTPE_2GPU.py \\
        --model_path out/ppiBTPE_epoch_17.pth \\
        --model_config facebook/esm1b_t33_650M_UR50S \\
        --num_layers 12 \\
        --input_file test_pairs.csv \\
        --output_file predictions.csv \\
        --batch_size 4 \\
        --max_length 1024 \\
        --device cuda

Example (multi-GPU)
-------------------
    python inference_ppiBTPE_2GPU.py \\
        --model_path out/ppiBTPE_final.pth \\
        --model_config facebook/esm1b_t33_650M_UR50S \\
        --num_layers 12 \\
        --input_file test_pairs.csv \\
        --output_file predictions.csv \\
        --device cuda:0,1
"""
import argparse
import os
import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer, EsmConfig
import pandas as pd
from tqdm import tqdm

# Command-line arguments
parser = argparse.ArgumentParser(
    description='Inference using the trained ppiBTPE Siamese model for PPI prediction.'
)
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained ppiBTPE checkpoint (.pth).')
parser.add_argument('--model_config', type=str, default='facebook/esm1b_t33_650M_UR50S',
                    help='ESM config name or path used during training.')
parser.add_argument('--num_layers', type=int, required=True,
                    help='Number of transformer layers used during training (for config override).')
parser.add_argument('--num_labels', type=int, default=2, help='Number of output labels.')
parser.add_argument('--input_file', type=str, required=True, help='CSV with protein pairs (seq1, seq2).')
parser.add_argument('--output_file', type=str, required=True, help='Path to write predictions CSV.')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference.')
parser.add_argument('--max_length', type=int, default=1024, help='Max token length.')
parser.add_argument('--device', type=str, default='cuda', help='Device: cpu or cuda or cuda:0,1')
args = parser.parse_args()

# Device setup
def get_device(device_str):
    if device_str == 'cpu':
        return torch.device('cpu'), None
    if ',' in device_str:
        devs = [d.strip() for d in device_str.split(',')]
        device = torch.device(devs[0])
        device_ids = [int(d.split(':')[-1]) for d in devs]
        return device, device_ids
    else:
        return torch.device(device_str), None

device, device_ids = get_device(args.device)

# Dataset
class PPIDatasetInference(torch.utils.data.Dataset):
    def __init__(self, csv_file, tokenizer, max_length):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq1 = self.data.iloc[idx, 0]
        seq2 = self.data.iloc[idx, 1]
        enc1 = self.tokenizer(seq1, truncation=True, padding='max_length',
                              max_length=self.max_length, return_tensors='pt')
        enc2 = self.tokenizer(seq2, truncation=True, padding='max_length',
                              max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids1': enc1.input_ids.squeeze(0),
            'attention_mask1': enc1.attention_mask.squeeze(0),
            'input_ids2': enc2.input_ids.squeeze(0),
            'attention_mask2': enc2.attention_mask.squeeze(0),
        }

# Model definition matching training
class SiameseBTPE(nn.Module):
    def __init__(self, config, num_labels=2):
        super(SiameseBTPE, self).__init__()
        self.esm = EsmModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        o1 = self.esm(input_ids=input_ids1, attention_mask=attention_mask1)
        o2 = self.esm(input_ids=input_ids2, attention_mask=attention_mask2)
        p1 = o1.last_hidden_state[:, 0, :]
        p2 = o2.last_hidden_state[:, 0, :]
        concat = torch.cat((p1, p2), dim=1)
        out = self.dropout(concat)
        logits = self.classifier(out)
        return logits


def main():
    # Tokenizer & config
    tokenizer = EsmTokenizer.from_pretrained(args.model_config)
    config = EsmConfig.from_pretrained(args.model_config)
    config.num_hidden_layers = args.num_layers
    print(f'Overriding config to {args.num_layers} transformer layers.')

    # Dataset & loader
    ds = PPIDatasetInference(args.input_file, tokenizer, args.max_length)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Model init & load
    model = SiameseBTPE(config, num_labels=args.num_labels)
    ckpt = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt)

    # DataParallel if needed
    if device_ids:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    model.eval()

    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference'):
            ids1 = batch['input_ids1'].to(device)
            mask1 = batch['attention_mask1'].to(device)
            ids2 = batch['input_ids2'].to(device)
            mask2 = batch['attention_mask2'].to(device)
            logits = model(ids1, mask1, ids2, mask2)
            probs = nn.functional.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    # Map & save
    label_map = {0: 'enemies', 1: 'friends'}
    df = pd.read_csv(args.input_file)
    df['Prediction'] = [label_map[p] for p in all_preds]
    df['Probability_Friends'] = [p[1] for p in all_probs]
    df['Probability_Enemies'] = [p[0] for p in all_probs]
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(f'Saved predictions to {args.output_file}')

if __name__ == '__main__':
    main()
