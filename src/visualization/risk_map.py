"""
src/visualization/risk_map.py
Flood risk zone overlays, color-coded maps, and interactive Plotly figures.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from src.utils.config_loader import load_config

cfg = load_config()
CLASS_NAMES = cfg["data"]["class_names"]
CLASS_COLORS = [tuple(c) for c in cfg["data"]["class_colors"]]
PLOTS_DIR = Path(cfg["output"]["plots_dir"])
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Color-coded risk zone overlay
# ---------------------------------------------------------------------------
def create_risk_zone_overlay(
    image: np.ndarray,
    risk_heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Create a color-coded risk zone overlay on a satellite image.

    Risk levels are color-coded:
        0.0 – 0.2 → Green  (Safe)
        0.2 – 0.4 → Yellow (Low Risk)
        0.4 – 0.6 → Orange (Medium Risk)
        0.6 – 0.8 → Red    (High Risk)
        0.8 – 1.0 → Dark Red (Flooded)

    Args:
        image: (H, W, 3) uint8 RGB satellite image.
        risk_heatmap: (H, W) float in [0, 1] — higher = more risk.
        alpha: Overlay transparency.

    Returns:
        (H, W, 3) uint8 composite image.
    """
    H, W = image.shape[:2]
    colored = np.zeros((H, W, 3), dtype=np.uint8)

    # Thresholds and colors (RGB)
    thresholds = [0.2, 0.4, 0.6, 0.8, 1.01]
    colors = [
        (0, 200, 0),      # Green
        (255, 255, 0),    # Yellow
        (255, 140, 0),    # Orange
        (220, 50, 0),     # Red
        (100, 0, 0),      # Dark red
    ]

    for i, (thresh, color) in enumerate(zip(thresholds, colors)):
        if i == 0:
            mask = risk_heatmap < thresh
        else:
            mask = (risk_heatmap >= thresholds[i - 1]) & (risk_heatmap < thresh)
        colored[mask] = color

    overlay = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
    return overlay


def add_risk_legend(ax: plt.Axes) -> None:
    """Add a flood risk legend to a matplotlib Axes."""
    legend_items = [
        mpatches.Patch(color="#00C800", label="Safe (0–20%)"),
        mpatches.Patch(color="#FFFF00", label="Low Risk (20–40%)"),
        mpatches.Patch(color="#FF8C00", label="Medium Risk (40–60%)"),
        mpatches.Patch(color="#DC3200", label="High Risk (60–80%)"),
        mpatches.Patch(color="#640000", label="Flooded (80–100%)"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=9,
        framealpha=0.85,
        title="Flood Risk",
        title_fontsize=10,
    )


# ---------------------------------------------------------------------------
# Matplotlib risk map
# ---------------------------------------------------------------------------
def plot_risk_map(
    image: np.ndarray,
    risk_heatmap: np.ndarray,
    title: str = "Flood Risk Map",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    2-panel figure: satellite image | color-coded risk overlay.
    """
    overlay = create_risk_zone_overlay(image, risk_heatmap)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    axes[0].imshow(image)
    axes[0].set_title("Satellite Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Flood Risk Overlay", fontsize=12)
    axes[1].axis("off")
    add_risk_legend(axes[1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Plotly interactive risk map
# ---------------------------------------------------------------------------
def create_interactive_risk_map(
    image: np.ndarray,
    risk_heatmap: np.ndarray,
    title: str = "Interactive Flood Risk Map",
) -> go.Figure:
    """
    Create an interactive Plotly figure with risk heatmap overlay.

    Users can zoom, pan, and hover to inspect risk values.
    """
    H, W = image.shape[:2]

    fig = go.Figure()

    # Base satellite image
    fig.add_trace(
        go.Image(
            z=image,
            name="Satellite",
            hovertemplate="x=%{x}<br>y=%{y}<extra>Satellite</extra>",
        )
    )

    # Risk heatmap overlay (semi-transparent)
    fig.add_trace(
        go.Heatmap(
            z=risk_heatmap,
            colorscale=[
                [0.0, "rgba(0,200,0,0.0)"],
                [0.2, "rgba(0,200,0,0.4)"],
                [0.4, "rgba(255,255,0,0.5)"],
                [0.6, "rgba(255,140,0,0.6)"],
                [0.8, "rgba(220,50,0,0.7)"],
                [1.0, "rgba(100,0,0,0.8)"],
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="Risk Level",
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["Safe", "Low", "Medium", "High", "Critical", "Flooded"],
                thickness=15,
            ),
            hovertemplate="Risk: %{z:.2f}<extra></extra>",
            name="Risk Overlay",
            opacity=0.55,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1a1a2e")),
        height=500,
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="#f8f9fa",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


# ---------------------------------------------------------------------------
# Training metrics plots
# ---------------------------------------------------------------------------
def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation loss + accuracy curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training History", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-s", label="Val", markersize=4)
    axes[0].set_title("Loss", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train", markersize=4)
    axes[1].plot(epochs, history["val_acc"], "r-s", label="Val", markersize=4)
    axes[1].set_title("Accuracy", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    model_name: str = "Model",
    normalize: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot normalized confusion matrix with color coding.
    """
    import seaborn as sns

    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
        title = f"{model_name} — Normalized Confusion Matrix"
    else:
        cm_plot = cm
        fmt = "d"
        title = f"{model_name} — Confusion Matrix"

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(
    vit_history: Dict,
    cnn_history: Dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ViT vs CNN training curve comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ViT vs CNN — Performance Comparison", fontsize=14, fontweight="bold")

    for ax, metric, ylabel in zip(
        axes,
        [("val_loss", "Validation Loss"), ("val_acc", "Validation Accuracy")],
        ["Loss", "Accuracy"],
    ):
        key, label = metric
        ep_vit = range(1, len(vit_history.get(key, [])) + 1)
        ep_cnn = range(1, len(cnn_history.get(key, [])) + 1)
        ax.plot(ep_vit, vit_history.get(key, []), "b-o", label="ViT-B/16", markersize=4)
        ax.plot(ep_cnn, cnn_history.get(key, []), "r-s", label="EfficientNet-B3", markersize=4)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
