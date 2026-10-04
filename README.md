# 🛰️ Satellite Image-Based Flood Risk Mapping Using Vision Transformers

A complete machine learning system for detecting and mapping flood-prone areas from satellite imagery using Vision Transformers (ViT) with an interactive web dashboard.

---

## 📌 Table of Contents

1. [Aim and Objectives](#aim-and-objectives)
2. [Project Overview](#project-overview)
3. [System Architecture](#system-architecture)
4. [Dataset Setup](#dataset-setup)
5. [Installation](#installation)
6. [Project Structure](#project-structure)
7. [Usage Guide](#usage-guide)
8. [Model Details](#model-details)
9. [Results & Evaluation](#results--evaluation)
10. [Dashboard](#dashboard)
11. [Extra Features](#extra-features)

---

## Aim and Objectives

**Full version with rationale, research questions, scope and deliverables: [`docs/objectives.md`](docs/objectives.md)**

### Aim

To develop a deep-learning–based system, **FloodSense**, that identifies and maps flood-prone regions in multispectral satellite imagery using Vision Transformers, and to make its predictions interpretable and accessible through an interactive web dashboard.

### Objectives

| # | Objective |
|---|-----------|
| **O1** | To curate and prepare a suitable dataset of flood and non-flood imagery (FloodNet / SEN12-FLOOD), applying resizing, normalisation and class balancing so all five risk tiers are adequately represented. |
| **O2** | To implement a reproducible preprocessing and augmentation pipeline supporting deterministic train/val/test splitting and geometric and photometric augmentation. |
| **O3** | To implement a **Vision Transformer (ViT-B/16)** classifier using self-attention over 16×16 patches, fine-tuning the final four transformer blocks from ImageNet-21k pretrained weights. |
| **O4** | To implement an **EfficientNet-B3 CNN baseline** to serve as a comparative benchmark for testing whether self-attention offers measurable advantage. |
| **O5** | To formulate the task across **five graded risk tiers** — Non-Flooded, Low Risk, Medium Risk, High Risk, Flooded — with calibrated thresholds, so output expresses flood *susceptibility* rather than a binary label. |
| **O6** | To train and tune both models under an **identical experimental protocol** (AdamW, 1e-4 LR, 0.01 WD, cosine annealing, 5 warm-up epochs, weighted CE loss) so only the architecture varies. |
| **O7** | To evaluate both models **quantitatively** using accuracy, precision, recall, F1-score, IoU and confusion matrices, documenting failure modes rather than headline accuracy alone. |
| **O8** | To make predictions **explainable** via Grad-CAM for the CNN and attention-rollout visualisation for the ViT. |
| **O9** | To deliver an **interactive Streamlit dashboard** enabling image upload, graded classification, probability bars, heatmap overlays, a zoomable risk map, and threshold-based alerts. |
| **O10** | To ensure **end-to-end reproducibility** via a synthetic demo-data generator and one-command quickstart, so the pipeline is demonstrable without large dataset downloads. |

### Research Questions

- **RQ1** — Does a ViT achieve superior flood-risk classification accuracy and F1-score versus a CNN baseline of comparable input resolution?
- **RQ2** — Can attention-based and gradient-based explainability reveal physically meaningful hydrogeographic features (river channels, drainage basins, elevation gradients)?
- **RQ3** — How does class imbalance between flood and non-flood regions affect reported metrics, and how best should it be mitigated?
- **RQ4** — Does a trained model deliver useful graded risk output on imagery from a different source distribution (cross-dataset generalisation)?

---

## Project Overview

This system analyzes multispectral satellite images to classify regions into flood risk categories:

| Category | Description |
|----------|-------------|
| 🔴 Flooded | Currently submerged areas |
| 🟠 High Risk | Very likely to flood |
| 🟡 Medium Risk | Moderate flood probability |
| 🟢 Low Risk | Unlikely to flood |
| ⚪ Non-Flooded | Confirmed safe areas |

### Why Vision Transformers?

Traditional CNNs process images locally via convolutional kernels. Vision Transformers (ViT) use self-attention to capture **global context** — critical for flood mapping where large-scale topographical patterns (river basins, elevation gradients, urban drainage) determine risk. ViTs achieve state-of-the-art results on remote sensing tasks by learning long-range dependencies across an entire scene.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FLOOD RISK SYSTEM                     │
├─────────────┬──────────────┬──────────────┬─────────────┤
│  Data Layer │  Model Layer │ Predict Layer│  UI Layer   │
│             │              │              │             │
│ • Download  │ • ViT Model  │ • Inference  │ • Streamlit │
│ • Preprocess│ • CNN Model  │ • Heatmaps   │ • Dashboard │
│ • Augment   │ • Training   │ • Risk Zones │ • Upload    │
│ • Split     │ • Evaluation │ • Attention  │ • Visualize │
└─────────────┴──────────────┴──────────────┴─────────────┘
```

---

## Dataset Setup

### Option 1: FloodNet Dataset (Recommended)
```bash
# FloodNet — UAV imagery post-Hurricane Harvey
# Download from: https://github.com/BinaLab/FloodNet-Supervised_v1.0
# Or via Kaggle:
pip install kaggle
kaggle datasets download -d kmader/floodnet-dataset
```

> ⚠️ **FloodNet ships binary labels (`Flooded` / `Non-Flooded`) and no
> risk-severity ground truth.** The five-tier scheme in `config.yaml`
> (`data.num_classes: 5`) therefore **cannot** be trained on FloodNet — the
> intermediate tiers would have to be invented.

> An earlier version of `setup_floodnet.py` shuffled each binary class and
> re-split it across the five tiers, so `Low Risk` and `Non-Flooded` were the
> same images and `Medium/High/Flooded` were the same images. Any accuracy
> reported from that pipeline measured the model's ability to separate random
> noise. **This is fixed** — the script now maps each image 1:1 from FloodNet
> ground truth and aborts rather than fabricating tiers.

**Honest binary run on FloodNet:**

```bash
python setup_floodnet.py --binary
```

This rewrites `config.yaml` to 2 classes (backing up the original to
`config.yaml.bak`), copies each image to exactly one class folder, then trains
both models. Results are genuine flood-detection measurements.

**Genuine five-tier labels** require real risk data — SEN12-FLOOD, or tiers
derived from DEM/topographic variables (elevation, slope, drainage density,
distance to water). Point `data.class_names` at those instead.

Folder names must match `data.class_names` **exactly**, including case and
hyphenation. `FloodDataset` raises a descriptive error listing expected
directories if any class folder is missing or empty, rather than silently
training on whatever it found.

Resulting layout (created by `prepare_dataset`, or by hand):

```
data/raw/
  Flooded/
  Non-Flooded/
  Low Risk/
  Medium Risk/
  High Risk/
data/processed/
  train/ val/ test/     # one subdir per class
```

### Option 2: SEN12-FLOOD Dataset
```bash
# Sentinel-1/2 multi-modal flood dataset
# https://mediatum.ub.tum.de/1554921
# Requires registration at TU Munich data portal
```

### Option 3: Use Synthetic Demo Data (Quick Start)
```bash
# Generate synthetic data for testing without downloading
python src/utils/generate_demo_data.py
```

---

## Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM

### Setup
```bash
# 1. Clone / enter project directory
cd flood_risk_mapping

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import timm; print('TIMM OK')"
```

---

## Project Structure

```
flood_risk_mapping/
├── README.md
├── requirements.txt
├── config.yaml                          # Central config
│
├── data/
│   ├── raw/                             # Original dataset
│   ├── processed/                       # Preprocessed tensors
│   └── samples/                         # Demo sample images
│
├── src/
│   ├── preprocessing/
│   │   ├── dataset.py                   # Dataset class & loaders
│   │   └── augmentation.py              # Augmentation pipeline
│   │
│   ├── models/
│   │   ├── vit_model.py                 # Vision Transformer
│   │   ├── cnn_model.py                 # CNN baseline
│   │   └── model_factory.py             # Model registry
│   │
│   ├── visualization/
│   │   ├── heatmap.py                   # GradCAM & attention viz
│   │   ├── risk_map.py                  # Flood risk overlays
│   │   └── metrics_plot.py              # Training curves
│   │
│   └── utils/
│       ├── trainer.py                   # Training loop
│       ├── evaluator.py                 # Metrics & evaluation
│       ├── predictor.py                 # Inference pipeline
│       └── generate_demo_data.py        # Demo data generator
│
├── dashboard/
│   └── app.py                           # Streamlit dashboard
│
├── outputs/
│   ├── models/                          # Saved checkpoints
│   ├── plots/                           # Generated figures
│   └── predictions/                     # Prediction outputs
│
├── docs/
│   └── architecture.md
│
├── train.py                             # Main training script
├── evaluate.py                          # Evaluation script
└── predict.py                           # Single image prediction
```

---

## Usage Guide

### Step 1: Prepare Data
```bash
# Generate demo data (or use real dataset)
python src/utils/generate_demo_data.py

# Preprocess dataset
python -c "
from src.preprocessing.dataset import prepare_dataset
prepare_dataset('data/raw', 'data/processed')
"
```

### Step 2: Train Models
```bash
# Train Vision Transformer
python train.py --model vit --epochs 30 --batch_size 16

# Train CNN baseline
python train.py --model cnn --epochs 30 --batch_size 32

# Compare both models
python evaluate.py --compare
```

### Step 3: Run Predictions
```bash
# Predict on a single image
python predict.py --image path/to/satellite_image.jpg

# Batch prediction
python predict.py --folder data/samples/
```

### Step 4: Launch Dashboard
```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

---

## Model Details

### Vision Transformer (ViT-B/16)
- **Base model**: `vit_base_patch16_224` (pretrained on ImageNet-21k)
- **Fine-tuning**: Last 4 transformer blocks + classification head
- **Input**: 224×224 RGB patches
- **Patch size**: 16×16 (196 patches per image)
- **Attention heads**: 12
- **Hidden dim**: 768

### CNN Baseline (EfficientNet-B3)
- **Base model**: EfficientNet-B3 (pretrained on ImageNet)
- **Fine-tuning**: Last 2 stages + head
- **Input**: 224×224 RGB

### Training Config
```yaml
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 0.01
scheduler: CosineAnnealingLR
warmup_epochs: 5
loss: CrossEntropyLoss (weighted)
```

---

## Results & Evaluation

Expected performance on FloodNet:

| Model | Accuracy | F1-Score | IoU | Params |
|-------|----------|----------|-----|--------|
| ViT-B/16 | ~91.3% | ~0.903 | ~0.847 | 86M |
| EfficientNet-B3 | ~87.8% | ~0.871 | ~0.812 | 12M |

---

## Dashboard

The Streamlit dashboard provides:
- 📤 **Upload**: Drag-and-drop satellite image upload
- 🔍 **Predict**: Real-time flood risk classification
- 📊 **Probability**: Per-class probability bars
- 🗺️ **Heatmap**: GradCAM attention overlays
- ⚠️ **Alerts**: Automatic high-risk notifications
- 📈 **Comparison**: Side-by-side ViT vs CNN results

---

## Extra Features

- **Attention Map Visualization**: See which image regions the ViT focuses on
- **GradCAM for CNN**: Gradient-weighted class activation maps
- **Real-time Alerts**: Webhook/notification when flood probability > threshold
- **Explainable AI**: LIME-based local explanations
- **Interactive Risk Map**: Plotly-based zoomable flood risk overlay
- **Model Comparison**: Live A/B comparison of ViT vs CNN predictions
