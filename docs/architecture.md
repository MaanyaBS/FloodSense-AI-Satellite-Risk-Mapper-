# System Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FLOOD RISK MAPPING SYSTEM                        │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Data Layer  │───▶│  Model Layer │───▶│  Prediction Layer    │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│          │                  │                       │               │
│          ▼                  ▼                       ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Visualization│   │   Training   │    │  Dashboard (UI)      │  │
│  │     Layer    │   │   Pipeline   │    │  Streamlit App       │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Data Layer

### Input Sources
```
Satellite Images (JPEG/PNG/GeoTIFF)
    ↓
FloodNet Dataset           SEN12-FLOOD Dataset      Demo/Synthetic Data
(UAV, post-Harvey)         (Sentinel-1/2, EU)        (Generated locally)
    ↓                           ↓                          ↓
                    Raw Image Directory
                    data/raw/<ClassName>/
```

### Preprocessing Pipeline
```
Raw Image
    ↓
1. Load & Convert to RGB (PIL)
    ↓
2. Resize to 224×224 (Bilinear interpolation)
    ↓
3. [TRAIN ONLY] Augmentation:
   ├── HorizontalFlip (p=0.5)
   ├── VerticalFlip (p=0.3)
   ├── Rotate ±30° (p=0.5)
   ├── BrightnessContrast / HueSaturation / CLAHE (p=0.6)
   ├── GaussianNoise (p=0.3)
   └── CoarseDropout (p=0.2)
    ↓
4. Normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    ↓
5. ToTensor → float32 (3, 224, 224)
```

### Dataset Splits
```
Raw Data (N images per class)
    ├── Train  70%  → WeightedRandomSampler (handles imbalance)
    ├── Val    15%  → Standard DataLoader
    └── Test   15%  → Standard DataLoader
```

---

## 2. Model Layer

### Vision Transformer (Primary)
```
Input (B, 3, 224, 224)
    ↓
Patch Embedding: 16×16 patches → 196 tokens + 1 CLS token
    ↓
Position Encoding (learnable)
    ↓
Transformer Encoder (12 blocks):
  Each block:
    ├── LayerNorm
    ├── Multi-Head Self-Attention (12 heads, dim=64 each)
    │     Q, K, V projections → Attention weights → Output
    ├── Residual connection
    ├── LayerNorm
    ├── MLP (dim 768 → 3072 → 768, GELU)
    └── Residual connection
    ↓
CLS token → LayerNorm → (768-dim feature vector)
    ↓
Classification Head:
  LayerNorm → Dropout(0.1) → Linear(768→384) → GELU → Dropout(0.1) → Linear(384→5)
    ↓
Logits (B, 5) → Softmax → Probabilities
```

### CNN Baseline (EfficientNet-B3)
```
Input (B, 3, 224, 224)
    ↓
Stem Conv (3→40 channels, 3×3, stride 2)
    ↓
7 MBConv Stages (mobile inverted bottleneck blocks)
    ├── Stage 1: 40ch, squeeze-excitation ratio 0.25
    ├── Stage 2: 24ch → 32ch
    ├── Stage 3: 32ch → 48ch
    ├── Stage 4: 48ch → 96ch
    ├── Stage 5: 96ch → 136ch
    ├── Stage 6: 136ch → 232ch (UNFREEZE from here)
    └── Stage 7: 232ch → 384ch (UNFREEZE)
    ↓
Conv Head (384 → 1536)
    ↓
Global Average Pooling → (1536-dim feature vector)
    ↓
Classification Head:
  BatchNorm → Dropout(0.3) → Linear(1536→512) → ReLU → BN → Dropout(0.2) → Linear(512→5)
    ↓
Logits (B, 5) → Softmax → Probabilities
```

---

## 3. Training Pipeline

```
TrainLoader ──────────────────────────────────────────────┐
                                                           │
Epoch Loop:                                               ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Forward Pass (AMP fp16)                                     │
  │      Images → Model → Logits                                 │
  │                                                              │
  │  Loss: CrossEntropyLoss(weight=class_weights, smoothing=0.1) │
  │                                                              │
  │  Backward: GradScaler.scale(loss).backward()                 │
  │                                                              │
  │  Optimizer: AdamW                                            │
  │    ├── Backbone params: lr × 0.1 (conservative fine-tuning)  │
  │    └── Head params: lr (full learning rate)                  │
  │                                                              │
  │  Gradient Clipping: max_norm=1.0                             │
  └──────────────────────────────────────────────────────────────┘
           │
           ▼
  Validation Loop → val_acc, val_loss
           │
           ▼
  LR Schedule: Warmup (5 epochs) + CosineAnnealing
           │
           ▼
  Early Stopping: patience=8 epochs
           │
           ▼
  Checkpoint: Save best val_acc model
```

---

## 4. Prediction Layer

```
Input Image
    ↓
Preprocessing (get_val_transforms)
    ↓
Model Forward Pass
    ↓
Softmax → Class Probabilities [p0, p1, p2, p3, p4]
    ↓
┌─────────────────────────────────────────────────────────┐
│  Classification:  argmax(probs) → predicted class        │
│  Risk Score:      weighted sum = Σ(i × pi) / (N-1)       │
│  Risk Level:      score < 0.20 → Low                     │
│                   score < 0.45 → Moderate                │
│                   score < 0.70 → High                    │
│                   score ≥ 0.70 → Critical                │
│  Alert:           p[Flooded] + p[High Risk] > threshold  │
└─────────────────────────────────────────────────────────┘
    ↓
PredictionResult (class, confidence, risk, alert, probs)
```

---

## 5. Visualization Layer

### Attention Rollout (ViT)
```
All 12 Transformer Blocks → Attention Weight Matrices (B, H, N, N)
    ↓
Per-block: fuse heads (mean) → discard low-attention tokens → add identity
    ↓
Compose: A_rollout = A_1 × A_2 × ... × A_12
    ↓
CLS row of A_rollout → (196,) patch attention weights
    ↓
Reshape to (14, 14) → Upsample to (224, 224) → Normalize [0,1]
    ↓
OpenCV COLORMAP_INFERNO → RGB overlay
```

### GradCAM (CNN)
```
Forward pass → Activations at target layer (B, C, h, w)
    ↓
Backward pass (for target class) → Gradients (B, C, h, w)
    ↓
Global average pool gradients → weights (C,)
    ↓
Weighted sum of activations → (h, w) raw CAM
    ↓
ReLU → Normalize [0,1] → Resize to (224, 224)
    ↓
OpenCV COLORMAP_JET → RGB overlay
```

---

## 6. Dashboard Layer

```
Streamlit App
├── Sidebar
│   ├── Model selector (ViT / CNN / Both)
│   ├── Attention toggle
│   ├── Interactive map toggle
│   ├── Alert threshold slider
│   └── Device info
│
├── Tab 1 — Predict
│   ├── Image upload (drag & drop)
│   ├── Flood alert / safe banner
│   ├── Risk badge with class + confidence
│   ├── Metric cards (flood prob, high risk prob, confidence)
│   ├── Probability bars (all 5 classes)
│   ├── Interactive Plotly risk map OR matplotlib risk map
│   ├── Attention map tabs (ViT rollout / CNN GradCAM)
│   └── JSON download button
│
├── Tab 2 — Model Comparison
│   ├── Architecture specs (side-by-side cards)
│   ├── Performance metrics table
│   └── Why ViT wins explanation
│
└── Tab 3 — About
    ├── System description
    ├── Risk categories
    ├── Datasets
    └── References
```

---

## Data Flow Summary

```
Satellite Image
      │
      ▼
  Preprocess ─────────────────────────────────────────────────────┐
      │                                                            │
      ▼                                                            │
  ViT Model ──► probabilities ──► risk classification ──► alert?   │
      │              │                                             │
      │         ViT Attention                                      │
      │              │                                             │
      ▼              ▼                                             │
  CNN Model ──► probabilities                                      │
      │              │                                             │
      │          GradCAM                                           │
      │              │                                             │
      └──────────────┴─────────────────────────────────────────────┘
                          │
                          ▼
              Dashboard Visualization
              ├── Risk overlay map
              ├── Attention / GradCAM heatmaps
              ├── Probability bars
              ├── Alert notifications
              └── JSON export
```
