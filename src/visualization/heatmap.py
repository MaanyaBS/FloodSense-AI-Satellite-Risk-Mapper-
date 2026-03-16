"""
src/visualization/heatmap.py
Attention map extraction (ViT) and GradCAM (CNN) visualization.
Also includes Explainable AI utilities.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.preprocessing.augmentation import denormalize
from src.utils.config_loader import load_config

cfg = load_config()
CLASS_NAMES = cfg["data"]["class_names"]
ALPHA = cfg["prediction"]["heatmap_alpha"]
PLOTS_DIR = Path(cfg["output"]["plots_dir"])
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# ViT Attention Map Visualization
# ---------------------------------------------------------------------------
def extract_vit_attention(
    model: nn.Module,
    image_tensor: torch.Tensor,
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """
    Extract attention rollout map from a ViT model.

    Uses the "Attention Rollout" technique (Abnar & Zuidema, 2020).

    Args:
        model: FloodViT instance.
        image_tensor: (1, 3, H, W) input tensor.
        head_fusion: How to aggregate multi-head attention ('mean', 'max', 'min').
        discard_ratio: Fraction of lowest-attention tokens to discard.

    Returns:
        Attention map as (H, W) float array in [0, 1].
    """
    model.eval()
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    attention_list = []

    def get_attention(module, input, output):
        try:
            B, N, C = input[0].shape
            if not hasattr(module, "num_heads"):
                return
            num_heads = module.num_heads
            head_dim = C // num_heads

            with torch.no_grad():
                qkv = module.qkv(input[0])
                qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                scale = head_dim ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                attention_list.append(attn.detach().cpu())
        except Exception:
            pass

    hooks = []
    for block in model.backbone.blocks:
        hooks.append(block.attn.register_forward_hook(get_attention))

    with torch.no_grad():
        _ = model(image_tensor)

    for h in hooks:
        h.remove()

    if not attention_list:
        size = cfg["data"]["image_size"]
        return np.ones((size, size), dtype=np.float32) * 0.5

    try:
        # Attention Rollout
        N = attention_list[0].shape[-1]  # num_tokens (e.g. 197 for ViT-B/16)
        result = torch.eye(N)            # (N, N)

        for attn in attention_list:
            # attn shape: (B, num_heads, N, N)
            # Step 1 — fuse heads → (B, N, N)
            if head_fusion == "mean":
                attn_fused = attn.mean(dim=1)
            elif head_fusion == "max":
                attn_fused = attn.max(dim=1).values
            else:
                attn_fused = attn.min(dim=1).values

            # attn_fused: (B, N, N) — take batch 0 → (N, N)
            attn_mat = attn_fused[0]  # (N, N)

            # Step 2 — discard low-attention tokens (per row)
            flat = attn_mat.view(N, -1)                      # (N, N)
            threshold = flat.quantile(discard_ratio, dim=-1, # (N,)
                                      keepdim=True)           # (N, 1)
            flat = torch.where(flat < threshold,
                               torch.zeros_like(flat), flat)
            attn_mat = flat.view(N, N)

            # Step 3 — add identity (residual) and row-normalise
            attn_mat = attn_mat + torch.eye(N)
            attn_mat = attn_mat / attn_mat.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Step 4 — compose
            result = torch.matmul(attn_mat, result)

        # CLS row → attention from CLS to every patch
        cls_attn = result[0, 1:]                     # (num_patches,)
        num_patches = int(cls_attn.numel() ** 0.5)
        mask = cls_attn.reshape(num_patches, num_patches).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    except Exception:
        # Robust fallback — uniform map
        img_size = cfg["data"]["image_size"]
        return np.ones((img_size, img_size), dtype=np.float32) * 0.5

    # Upsample to image size
    img_size = cfg["data"]["image_size"]
    mask_resized = cv2.resize(mask, (img_size, img_size))
    return mask_resized.astype(np.float32)


# ---------------------------------------------------------------------------
# GradCAM for CNN
# ---------------------------------------------------------------------------
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN models.

    Usage:
        gradcam = GradCAM(model, target_layer)
        cam = gradcam(image_tensor, class_idx)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self._activations = output.detach()

        def bwd_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute GradCAM heatmap.

        Returns:
            cam: (H, W) float array in [0, 1].
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device).requires_grad_(True)

        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global average pool of gradients
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Normalize and resize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        img_size = cfg["data"]["image_size"]
        cam = cv2.resize(cam, (img_size, img_size))
        return cam.astype(np.float32)


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------
def apply_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    Blend a heatmap over an RGB image.

    Args:
        image: (H, W, 3) uint8 RGB image.
        heatmap: (H, W) float array in [0, 1].
        colormap: OpenCV colormap code.
        alpha: Heatmap overlay opacity.

    Returns:
        Blended (H, W, 3) uint8 image.
    """
    heatmap_u8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_u8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
    return overlay


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_attention_comparison(
    image: np.ndarray,
    vit_attention: np.ndarray,
    cnn_gradcam: np.ndarray,
    pred_class: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side comparison: original | ViT attention | CNN GradCAM.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Prediction: {pred_class}", fontsize=14, fontweight="bold")

    axes[0].imshow(image)
    axes[0].set_title("Satellite Image", fontsize=11)
    axes[0].axis("off")

    overlay_vit = apply_heatmap_overlay(image, vit_attention, cv2.COLORMAP_INFERNO)
    axes[1].imshow(overlay_vit)
    axes[1].set_title("ViT Attention Rollout", fontsize=11)
    axes[1].axis("off")

    overlay_cnn = apply_heatmap_overlay(image, cnn_gradcam, cv2.COLORMAP_JET)
    axes[2].imshow(overlay_cnn)
    axes[2].set_title("CNN GradCAM", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_probability_bars(
    class_probs: dict,
    title: str = "Flood Risk Probabilities",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of class probabilities."""
    RISK_COLORS = ["#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#8e1a1a"]

    classes = list(class_probs.keys())
    probs = list(class_probs.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(classes, probs, color=RISK_COLORS[:len(classes)], height=0.6)

    for bar, prob in zip(bars, probs):
        ax.text(
            max(bar.get_width() - 0.03, 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center",
            ha="right" if prob > 0.15 else "left",
            fontweight="bold",
            color="white" if prob > 0.15 else "black",
            fontsize=10,
        )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig