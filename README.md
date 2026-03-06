# ppiBTEP

A Siamese (twin-branch) protein-protein interaction classifier built on ESM-1b ([Rives et al., 2021](https://doi.org/10.1073/pnas.2016239118)). Also designated SiameseBTPE (BERT-Twin Protein Encoder).

![ppiBTEP Architecture](assets/ppiBTEP.png)

## Overview

ppiBTEP processes each protein independently through a shared ESM-1b encoder -- no cross-sequence attention is used between the two proteins. Each branch extracts the `[CLS]` token embedding from the final transformer layer, the two embeddings are concatenated, and a dropout + linear classification head produces binary interaction predictions with softmax probabilities.

Unlike the cross-encoding approach (see [ppiDCE](https://github.com/kouroshSA/ppiDCE)), ppiBTEP must capture interaction-predictive features entirely from each protein's own sequence context. This makes it faster per pair and allows protein representations to be precomputed and reused, at the cost of not modeling direct inter-protein residue dependencies.

The model was developed for the *Prochlorococcus marinus* MED4 interactome, where it serves as one component of a tri-model consensus framework (alongside [ppiGPLM](https://github.com/kouroshSA/ppiGPLM) and [ppiDCE](https://github.com/kouroshSA/ppiDCE)) for computational PPI screening.

## Architecture

| Parameter | Value |
|-----------|-------|
| Foundation | ESM-1b (facebook/esm1b_t33_650M_UR50S) |
| Strategy | Siamese / twin-branch |
| Layers | 6-18 (configurable) |
| Classification | Concat [CLS_A, CLS_B] -> Dropout(0.1) -> Linear -> 2 |
| Max sequence length | 1,024 tokens |
| Optimizer | AdamW (lr = 1 x 10^-5) |
| Loss | Cross-Entropy |

### Siamese vs Cross-Encoder

| | ppiDCE (Cross-Encoder) | ppiBTEP (Siamese) |
|---|---|---|
| Input | `[CLS] Seq_A [SEP] Seq_B` (joint) | `[CLS] Seq_A` and `[CLS] Seq_B` (separate) |
| Cross-attention | Full bidirectional at every layer | None |
| Classification | Single [CLS] -> Linear | Concat [CLS_A, CLS_B] -> Linear |
| Complexity | O((n+m)^2) | O(n^2) + O(m^2) |
| Speed | Slower (joint encoding) | Faster (independent, reusable) |

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- conda (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/kouroshSA/ppiBTEP.git
cd ppiBTEP

# Create a conda environment
conda create -n esm python=3.10
conda activate esm
pip install -r requirements.txt
```

## Repository Structure

```
ppiBTEP/
|-- train_ppiBTPE3b.py                   # Training script
|-- inference_ppiBTPE_2GPU.py            # Batch inference script (multi-GPU)
|-- roc_analysis_color_threshold_F1e.py  # ROC curve analysis with F1 optimization
|-- assets/
|   |-- ppiBTEP.png                      # Model architecture diagram
|   |-- ppiBTEP_architecture.svg         # Detailed architecture flow (SVG)
|   +-- ppiBTEP_architecture.png         # Detailed architecture flow (PNG)
|-- requirements.txt
|-- LICENSE
+-- README.md
```

## Usage

### Data Format

Training and inference use CSV files with columns: `seq1, seq2, label`

- `seq1`, `seq2`: Amino acid sequences
- `label`: `0` or `enemies` (non-interacting), `1` or `friends` (interacting)

For inference-only input, only the first two columns are required.

### Training

```bash
# Train from scratch with 12 layers
python train_ppiBTPE3b.py \
    --train_file train.csv \
    --val_file val.csv \
    --model_config facebook/esm1b_t33_650M_UR50S \
    --num_layers 12 \
    --freeze_layers 0 \
    --epochs 20 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_length 1024 \
    --output_dir ./out \
    --device cuda
```

#### Key training options

- `--num_layers N`: Total transformer layers (6, 8, 12, 16, or 18)
- `--freeze_layers N`: Freeze bottom N layers (use 0 for training from scratch)
- `--checkpoint path.pth`: Resume from a saved checkpoint
- `--model_config`: ESM model config (default: `facebook/esm1b_t33_650M_UR50S`)

**Important:** When training from scratch, use `--freeze_layers 0` to ensure all layers (including embeddings) remain trainable. The default is 20, which would freeze most layers.

### Inference

```bash
python inference_ppiBTPE_2GPU.py \
    --model_path out/ppiBTPE_epoch_17.pth \
    --model_config facebook/esm1b_t33_650M_UR50S \
    --num_layers 12 \
    --input_file test_pairs.csv \
    --output_file predictions.csv \
    --batch_size 4 \
    --max_length 1024 \
    --device cuda
```

Multi-GPU inference:
```bash
python inference_ppiBTPE_2GPU.py \
    --model_path out/ppiBTPE_final.pth \
    --model_config facebook/esm1b_t33_650M_UR50S \
    --num_layers 12 \
    --input_file test_pairs.csv \
    --output_file predictions.csv \
    --device cuda:0,1
```

Output CSV columns: `seq1, seq2, Prediction, Probability_Friends, Probability_Enemies`

### ROC Analysis

Evaluate model predictions using ROC curve analysis with threshold-colored visualization and F1 optimization:

```bash
python roc_analysis_color_threshold_F1e.py \
    --input_csv probabilities.csv \
    --output_file roc_curve.png
```

The input CSV should have two columns: PRS (positive) and RRS (random/negative) probability values.

## Architecture Diagram

See `assets/ppiBTEP_architecture.svg` for a detailed flow diagram covering:
- **A.** Model architecture (Siamese / twin-branch strategy)
- **B.** ppiBTEP (Siamese) vs ppiDCE (cross-encoder) comparison
- **C.** Training pipeline
- **D.** Inference pipeline

## Citation

If you use this software, please cite:

```
Daakour, S. et al. (2026).
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
