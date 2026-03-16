# 🛰️ Satellite Image-Based Flood Risk Mapping Using Vision Transformers

A complete machine learning system for detecting and mapping flood-prone areas from satellite imagery using Vision Transformers (ViT) with an interactive web dashboard.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Dataset Setup](#dataset-setup)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Usage Guide](#usage-guide)
7. [Model Details](#model-details)
8. [Results & Evaluation](#results--evaluation)
9. [Dashboard](#dashboard)
10. [Extra Features](#extra-features)

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

# After download, organize as:
data/raw/
  train/
    Flooded/
    Non-Flooded/
  test/
    Flooded/
    Non-Flooded/
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
