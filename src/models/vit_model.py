"""
src/models/vit_model.py
Vision Transformer (ViT) model for flood risk classification.
Uses a pretrained ViT-B/16 from TIMM and fine-tunes it.
"""

from typing import Optional, Tuple
import os

import torch
import torch.nn as nn
import timm
from einops import rearrange

from src.utils.config_loader import load_config

cfg = load_config()
model_cfg = cfg["models"]["vit"]
NUM_CLASSES = cfg["data"]["num_classes"]


# ---------------------------------------------------------------------------
# Attention extraction hook
# ---------------------------------------------------------------------------
class AttentionExtractor:
    """
    Hooks into the last attention block of a ViT to extract attention maps
    for visualization.
    """

    def __init__(self):
        self.attention_maps = []
        self._hooks = []

    def register(self, model: nn.Module):
        self.clear()
        for block in model.blocks:
            h = block.attn.register_forward_hook(self._hook_fn)
            self._hooks.append(h)

    def _hook_fn(self, module, input, output):
        pass

    def register_attn_weights(self, model: nn.Module):
        self.clear()
        for block in model.blocks:
            h = block.attn.register_forward_hook(self._capture_weights)
            self._hooks.append(h)

    def _capture_weights(self, module, input, output):
        pass

    def clear(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.attention_maps = []


# ---------------------------------------------------------------------------
# ViT Model
# ---------------------------------------------------------------------------
class FloodViT(nn.Module):

    def __init__(
        self,
        model_name: str = None,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = None,
        finetune_layers: int = None,
        dropout: float = None,
    ):
        super().__init__()

        self.model_name = model_name or model_cfg["name"]
        self.num_classes = num_classes
        self.pretrained = pretrained if pretrained is not None else model_cfg["pretrained"]
        self.finetune_layers = finetune_layers or model_cfg["finetune_layers"]

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=0,
            drop_rate=dropout or model_cfg["dropout"],
            attn_drop_rate=model_cfg["attention_dropout"],
        )

        hidden_dim = self.backbone.embed_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout or model_cfg["dropout"]),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._freeze_backbone()

    def _freeze_backbone(self):

        for param in self.backbone.parameters():
            param.requires_grad = False

        total_blocks = len(self.backbone.blocks)
        unfreeze_from = max(0, total_blocks - self.finetune_layers)

        for i, block in enumerate(self.backbone.blocks):
            if i >= unfreeze_from:
                for param in block.parameters():
                    param.requires_grad = True

        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())

        print(
            f"[ViT] Trainable: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.1f}%)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def forward_with_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features

    def get_attention_maps(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        attention_output = {}

        def hook_fn(module, input, output):
            B, N, C = input[0].shape

            qkv = module.qkv(input[0]).reshape(
                B, N, 3, module.num_heads, C // module.num_heads
            ).permute(2, 0, 3, 1, 4)

            q, k, v = qkv.unbind(0)

            scale = (C // module.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)

            attention_output["attn"] = attn.detach()

        last_block = self.backbone.blocks[-1]
        hook = last_block.attn.register_forward_hook(hook_fn)

        with torch.no_grad():
            logits = self.forward(x)

        hook.remove()

        attn_maps = attention_output.get("attn", None)
        return logits, attn_maps

    def predict_proba(self, x: torch.Tensor):

        self.eval()
        with torch.no_grad():
            logits = self.forward(x)

        return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_vit(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> FloodViT:

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FloodViT(num_classes=num_classes, pretrained=pretrained)

    # FIXED SECTION
    if checkpoint_path and os.path.exists(checkpoint_path):

        state = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(state["model_state_dict"])

        print(f"[ViT] Loaded checkpoint: {checkpoint_path}")

    return model.to(device)