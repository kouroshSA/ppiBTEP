"""
Developed by K. Salehi-Ashtiani and ChatGPTo1

***************DEPENDENCY INSTALLATION********************************************************************
Make a new conda env with python 3.10, e.g., $conda create -n esm python=3.10, then $conda activate esm
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install pandas
pip install fair-esm #Install the ESM Library
pip install biopython
pip install transformers
conda install tqdm
********************************************************************************************************


The script defines a Siamese-style model architecture that uses a pre-trained ESM (Evolutionary Scale Modeling) transformer-based protein language model for predicting protein-protein interactions (PPIs). In more detail:

Model Type:
we reworked the Siamese PPI script into a new ppiBTPE framework that initializes the transformer from scratch and still supports freezing the first N layers for fine-tuning. Check out the updated siamese_ppi_btpe.py canvas alongside—I’ve renamed the model class to SiameseBTPE, switched to random weight initialization via the ESM config, and kept the same CLI flags (now using --model_config to point at the architecture).

The architecture allows freezing lower layers of a pretrained model and training the top ones as specified by args; it should also allow starting with a pretrained ESM model and again, freezing lower layers and training the top one ()

Approach:

Pre-trained Foundation Model: The code employs a pre-trained ESM model architecture (to train from scratch, or from a pretrained ESM model) from the Hugging Face Transformers library. ESM models are large language models trained on protein sequences.
Feature Extraction: Each input sequence is tokenized and fed into the ESM model to produce embedding representations. The [CLS] (or first token) embeddings from both sequences are extracted.
Siamese Architecture: The outputs from the two branches (each branch is the same model, i.e., user or ESM model, applied to a different input sequence) are concatenated and passed through a linear classification layer to predict whether the two proteins interact.
Fine-Tuning with Layer Freezing: The code partially freezes the lower layers of the model, allowing only the top layers and the classification head to be fine-tuned.
Classification Task: The final output is a classification (e.g., binary classification of "friends" vs. not interacting), trained using a cross-entropy loss.
In summary, the script implements a Siamese neural network for protein-protein interaction prediction, allowing training from scratch, or using a pre-trained transformer model with partial fine-tuning.




python train_siamese_esm.py \
    --train_file path/to/train.csv \
    --val_file path/to/val.csv \
    --output_dir ./trained_model \
    --epochs 5 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --max_length 512 \
    --pretrained_model facebook/esm1b_t33_650M_UR50S \
    --device cuda

    Explanation of Command-Line Arguments
--train_file (Required): Path to the training CSV file containing the protein pairs and labels.

--val_file (Required): Path to the validation CSV file.
--output_dir: Directory where the trained model will be saved. Defaults to the current directory (./).

--epochs: Number of epochs to train the model. Default is 3.

--batch_size: Batch size for both training and validation. Default is 4.

--learning_rate: Learning rate for the optimizer. Default is 1e-5.

--max_length: Maximum sequence length for tokenization. Sequences longer than this will be truncated. Default is 1024.

--pretrained_model: Name or path of the pre-trained ESM model to use. Default is 'facebook/esm1b_t33_650M_UR50S'.

--num_labels: Number of output labels for classification. For binary classification, this should be 2. Default is 2.

--device: Device to run the training on. Options are 'cpu' or 'cuda'. Default is 'cuda' if available, otherwise 'cpu'.



Summary of Args
Positional Arguments: None. Some arguments are optional and some are required.

Required Arguments:

--train_file: Training data file.
--val_file: Validation data file.

Optional Arguments:

--output_dir: Where to save the trained model.
--epochs: Number of epochs to train.
--batch_size: Batch size.
--learning_rate: Learning rate.
--max_length: Maximum tokenization length.
--pretrained_model: Pre-trained model name or path.
--num_labels: Number of labels for classification.
--device: Compute device to use.


Example Command:


python train_siamese_esm.py \
    --train_file path/to/train.csv \
    --val_file path/to/val.csv \
    --output_dir ./trained_model \
    --epochs 5 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --max_length 1024 \
    --pretrained_model facebook/esm1b_t33_650M_UR50S \
    --device cuda \
    --freeze_layers 20


Sample command to run the script:

python siamese-ppi-esm4.py --train_file new_train_biogrid-humRND.csv --val_file val_friends-hum-RND.csv --output_dir out-tbt --epochs 5 --max_length 1024 --device cuda --freeze_layers 20

python siamese-ppi-esm4.py --train_file train.csv --val_file val.csv --output_dir out-tbt --epochs 5 --max_length 1024 --device cuda --freeze_layers 26


"""
#!/usr/bin/env python3

"""
Developed by K. Salehi-Ashtiani and ChatGPT

This script defines a Siamese-style BERT-Twin Protein Encoder model (ppiBTPE) for predicting protein-protein interactions (PPIs). The model is initialized from scratch (random weights) using a transformer architecture defined by an ESM model configuration, and includes optional layer freezing functionality for fine-tuning.



The updated ssi­amese_ppi_btpe.py still accepts:

--train_file

--val_file

--model_config (formerly your pretrained‐ESM flag; now points at the ESM config for architecture)

--num_labels

--freeze_layers
--num_layers # e.g. when you start from scratch
--epochs
--batch_size
--learning_rate
--max_length
--output_dir
--device

python siamese_ppi_btpe.py \
  --train_file path/to/train.csv \
  --val_file   path/to/val.csv

--model_config facebook/esm1b_t33_650M_UR50S
--num_labels 2
--freeze_layers 20 #--freeze_layers 0 when starting from scratch
--num_layers # specify when start from scratch 12
--epochs 3
--batch_size 4
--learning_rate 1e-5
--max_length 1024
--output_dir ./
--device cuda (falls back to CPU if no GPU)


python train_ppiBTPE1.py --train_file train.csv --val_file val.csv --output_dir out-tbt --epochs 5 --max_length 1024 --device cuda --freeze_layers --num_layers 12


"""
#!/usr/bin/env python3

"""
Developed by K. Salehi-Ashtiani and ChatGPT

This script defines a Siamese-style BERT-Twin Protein Encoder model (ppiBTPE) for predicting protein-protein interactions (PPIs). The model is initialized from scratch (random weights) using a transformer architecture defined by an ESM model configuration, with optional layer freezing and configurable depth for fine-tuning or scratch training.

**************
Requirements:
Make a new conda env with python 3.10, e.g., $conda create -n esm python=3.10, then $conda activate esm
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install pandas
pip install fair-esm #Install the ESM Library
pip install biopython
pip install transformers
conda install tqdm
**************Additional notes on this version

When starting to train from scratch, do '--freeze_layers 0' instead of not including the argument, that way all layers (including embeddings) remain trainable. If you left the flag off, you’d end up freezing 20 layers by default.

Data are padded to a max_length parameters; therefore, use the same or smaller --max_length when you fine-tune or infer.

Important: When fine-tuning, make sure you specify where the checkpoint is, e.g.,:
--checkpoint out-tbt/ppiBTPE_epoch_12.pth \

Example:
$ python train_ppiBTPE3.py --train_file train01_s.csv --val_file val01_s.csv --output_dir out-tbt --checkpoint out-tbt/ppiBTPE_epoch_1.pth --epochs 2 --max_length 1024 --batch_size 1 --device cuda --freeze_layers 10 --num_layers 14



"""
#!/usr/bin/env python3

"""
Developed by K. Salehi-Ashtiani and ChatGPT

This script defines a Siamese-style BERT-Twin Protein Encoder model (ppiBTPE) for predicting protein-protein interactions (PPIs).
Supports:
 - training from scratch (--num_layers)
 - partial fine-tuning (--freeze_layers)
 - loading weights from a checkpoint (--checkpoint)



"""
import argparse
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
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate.')
parser.add_argument('--max_length', type=int, default=1024,
                    help='Maximum sequence length for tokenization.')

# Misc
parser.add_argument('--output_dir', type=str, default='./',
                    help='Directory to save checkpoints and final model.')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'],
                    help='Device to run training on.')
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

    for e in range(args.epochs):
        print(f"Epoch {e+1}/{args.epochs}")
        model.train()
        train_loss=0
        for b in tqdm(train_loader, desc='Train'):
            optim.zero_grad()
            logits = model(b['input_ids1'].to(device), b['attention_mask1'].to(device),
                           b['input_ids2'].to(device), b['attention_mask2'].to(device))
            loss = crit(logits, b['labels'].to(device))
            loss.backward(); optim.step()
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
