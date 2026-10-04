# Aim and Objectives

> **FloodSense — Satellite Image-Based Flood Risk Mapping Using Vision Transformers**
> This document states the aim and objectives of the project. It is intended to be
> lifted directly into Chapter 1 (Introduction) of the project report.

---

## 1. Aim

To develop a deep-learning–based system, **FloodSense**, that identifies and maps
flood-prone regions in multispectral satellite imagery using Vision Transformers,
and to make its predictions interpretable and accessible through an interactive web
dashboard.

The aim is deliberately two-part. A model that achieves high accuracy but cannot
explain *why* a region was flagged is of limited operational value to disaster-response
planners, so **predictive performance** and **interpretability** are treated as
co-equal requirements rather than performance first, explainability last.

---

## 2. Objectives

The project will be pursued with the following objectives:

| # | Objective | Rationale |
|---|-----------|-----------|
| **O1** | To curate and prepare a suitable dataset of flood and non-flood imagery, drawing on **FloodNet** (UAV imagery captured after Hurricane Harvey) and/or **SEN12-FLOOD** (Sentinel-1/2 multimodal flood data), applying resizing, normalisation and class balancing so that all five risk tiers are adequately represented. | Flood datasets are small and heavily imbalanced; without deliberate curation, a model will appear accurate while ignoring minority flood classes. |
| **O2** | To implement a **reproducible preprocessing and augmentation pipeline** supporting deterministic train/validation/test splitting and geometric and photometric augmentation (flips, rotation, brightness/contrast, Gaussian noise). | Reproducibility is a stated requirement for academic evaluation; augmentation mitigates overfitting on small datasets. |
| **O3** | To implement a **Vision Transformer (ViT-B/16) classifier** using self-attention over 16×16 image patches, fine-tuning the final four transformer blocks and the classification head from ImageNet-21k pretrained weights, for graded flood-risk classification. | Self-attention captures global context across a scene, which is central to flood mapping where basin morphology and drainage patterns span the whole image. |
| **O4** | To implement an **EfficientNet-B3 CNN baseline** to serve as a comparative benchmark, enabling an empirical test of whether self-attention yields a measurable advantage over convolutional feature extraction. | A baseline is required before any claim of architectural superiority can be substantiated. |
| **O5** | To formulate the classification task across **five graded risk tiers** — Non-Flooded, Low Risk, Medium Risk, High Risk, and Flooded — with calibrated probability thresholds, so that output expresses flood *susceptibility* rather than a binary flooded/not-flooded label. | Graded output supports triage: resources can be directed at "High Risk" regions before inundation occurs, rather than only after. |
| **O6** | To train and tune both models under an **identical experimental protocol** (AdamW, learning rate 1e-4, weight decay 0.01, cosine annealing with 5 warm-up epochs, weighted cross-entropy loss, gradient clipping, early stopping). | Fair comparison requires that only the architecture varies between the two arms of the experiment. |
| **O7** | To evaluate both models **quantitatively** using accuracy, precision, recall, F1-score, IoU (Intersection over Union) and confusion matrices, and to document and analyse failure modes rather than reporting headline accuracy alone. | Disaster-response systems are safety-relevant; recall on flooded regions and false-negative rates matter more than aggregate accuracy. |
| **O8** | To make predictions **explainable** by implementing Grad-CAM for the CNN baseline and attention-rollout visualisation for the ViT, so that the image regions driving each prediction can be inspected and defended. | Explainability is a requirement for trust and for expert validation, not an optional extra. |
| **O9** | To deliver an **interactive web dashboard** (Streamlit) allowing a non-technical user to upload a satellite image, obtain a graded risk classification with per-class probability bars, view heatmap overlays and a zoomable risk map, and receive automated alerts when predicted risk exceeds a defined threshold. | Operational usability by non-specialists is the practical route from research artefact to usable tool. |
| **O10** | To ensure **end-to-end reproducibility** by providing a synthetic demo-data generator and a one-command quickstart, so the full pipeline — training, evaluation, inference and dashboard — can be demonstrated and benchmarked without requiring large dataset downloads. | Independent verification by reviewers is not possible if the pipeline requires multi-gigabyte inputs. |

---

## 3. Research Questions

These questions operationalise the objectives and structure the evaluation.

| # | Research question | Addressed by |
|---|-------------------|--------------|
| **RQ1** | Does a Vision Transformer achieve superior flood-risk classification accuracy and F1-score compared to a CNN baseline of comparable input resolution? | O3, O4, O6, O7 |
| **RQ2** | Can attention-based and gradient-based explainability reveal physically meaningful hydrogeographic features, such as river channels, drainage basins and topographic gradients? | O8 |
| **RQ3** | How does class imbalance between flood and non-flood regions affect reported performance metrics, and how best should it be mitigated? | O1, O6, O7 |
| **RQ4** | Can a trained model deliver useful graded risk output on imagery from a source distribution different from its training data (cross-dataset generalisation)? | O1, O5, O7 |

---

## 4. Scope

**In scope**

- Binary and graded flood-risk classification from overhead imagery using the five defined tiers
- Two model architectures (ViT-B/16 and EfficientNet-B3) under a controlled comparison
- Offline evaluation on FloodNet and/or SEN12-FLOOD
- Explainability tooling (Grad-CAM, attention rollout)
- A single-image and batch inference pipeline with threshold-based risk alerting
- A Streamlit dashboard for non-technical users

**Out of scope (explicitly deferred)**

- Real-time or operational flood forecasting, and hydrological simulation
- Per-pixel semantic segmentation of water bodies (this is image-*classification*, not segmentation)
- Time-series change detection and temporal flood evolution
- Deployment at production scale, cloud infrastructure, and automated data pipelines
- Mobile or field-application interfaces

---

## 5. Success Criteria and Deliverables

| Objective | Deliverable | Measure of success |
|-----------|-------------|--------------------|
| O1, O2 | `src/preprocessing/dataset.py`, `src/preprocessing/augmentation.py` | Deterministic splits; augmentation pipeline reproducible under a fixed seed |
| O3 | `src/models/vit_model.py` | ViT-B/16 trains to convergence; checkpoint saved to `outputs/models/vit_best.pth` |
| O4 | `src/models/cnn_model.py` | EfficientNet-B3 trains under the identical protocol; checkpoint saved |
| O5 | `config.yaml` (`data.num_classes`, `prediction.risk_thresholds`) | All five tiers produced with calibrated probabilities |
| O6 | `train.py`, `config.yaml` (`training.*`) | Both models trained with identical hyperparameters; only architecture differs |
| O7 | `evaluate.py`, `src/utils/evaluator.py` | Metrics table and confusion matrices generated via `python evaluate.py --compare` |
| O8 | `src/visualization/heatmap.py` | Grad-CAM and attention overlays generated for sample images |
| O9 | `dashboard/app.py` | Dashboard serves at `localhost:8501`, accepts an upload, returns a graded prediction with overlays |
| O10 | `src/utils/generate_demo_data.py`, `quickstart.sh` | Full pipeline demonstrable from a clean clone without dataset downloads |

---

## 6. Expected Contribution

The project is expected to contribute:

1. An open, reproducible reference implementation of ViT-based flood-risk classification covering data curation, training, evaluation, explainability and deployment, rather than model architecture alone.
2. An empirical, protocol-controlled comparison of self-attention against convolutional baselines for flood mapping, reported with IoU and confusion matrices rather than accuracy alone.
3. A defensible bridge between a research model and an operational interface, by pairing every prediction with a visual explanation and a confidence score that a non-specialist can interpret.