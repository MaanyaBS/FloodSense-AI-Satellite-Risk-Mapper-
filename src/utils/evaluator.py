"""
src/utils/evaluator.py
Comprehensive model evaluation: accuracy, F1, confusion matrix,
per-class metrics, and comparison between models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config_loader import load_config

cfg = load_config()
CLASS_NAMES = cfg["data"]["class_names"]


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Run full evaluation on a DataLoader.

    Returns:
        Dictionary containing:
        - accuracy, f1_macro, f1_weighted
        - per_class_metrics (dict)
        - confusion_matrix (ndarray)
        - all_probs (ndarray) shape (N, num_classes)
        - all_labels (ndarray) shape (N,)
        - all_preds (ndarray) shape (N,)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating", ncols=80):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=-1)

        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    # Per-class metrics
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # Multiclass AUC (one-vs-rest)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "auc_macro": auc,
        "confusion_matrix": cm,
        "per_class_metrics": report,
        "all_probs": y_prob,
        "all_labels": y_true,
        "all_preds": y_pred,
    }


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def compare_models(
    vit_results: Dict,
    cnn_results: Dict,
) -> Dict:
    """
    Compare ViT vs CNN evaluation results.

    Returns:
        Dictionary with side-by-side metric comparison.
    """
    metrics = ["accuracy", "f1_macro", "f1_weighted", "auc_macro"]
    comparison = {}

    for m in metrics:
        vit_val = vit_results.get(m, float("nan"))
        cnn_val = cnn_results.get(m, float("nan"))
        delta = vit_val - cnn_val
        comparison[m] = {
            "vit": vit_val,
            "cnn": cnn_val,
            "delta": delta,
            "winner": "ViT" if delta > 0 else "CNN",
        }

    return comparison


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------
def print_evaluation_report(results: Dict, model_name: str = "Model"):
    """Pretty-print evaluation results."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {model_name} Evaluation Report")
    print(sep)
    print(f"  Accuracy:    {results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)")
    print(f"  F1 Macro:    {results['f1_macro']:.4f}")
    print(f"  F1 Weighted: {results['f1_weighted']:.4f}")
    print(f"  AUC (macro): {results['auc_macro']:.4f}")
    print()
    print("  Per-class metrics:")
    for cls in CLASS_NAMES:
        m = results["per_class_metrics"].get(cls, {})
        p = m.get("precision", 0)
        r = m.get("recall", 0)
        f = m.get("f1-score", 0)
        s = m.get("support", 0)
        print(f"    {cls:<18} P={p:.3f}  R={r:.3f}  F1={f:.3f}  N={int(s)}")
    print(sep)


def print_comparison_report(comparison: Dict):
    """Pretty-print model comparison."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  ViT vs CNN — Performance Comparison")
    print(sep)
    print(f"  {'Metric':<20} {'ViT':>10} {'CNN':>10} {'Δ':>10} {'Winner':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for metric, vals in comparison.items():
        print(
            f"  {metric:<20} "
            f"{vals['vit']:>10.4f} "
            f"{vals['cnn']:>10.4f} "
            f"{vals['delta']:>+10.4f} "
            f"{vals['winner']:>8}"
        )
    print(sep)
