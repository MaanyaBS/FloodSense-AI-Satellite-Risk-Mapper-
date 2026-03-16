"""
src/models/cnn_model.py
EfficientNet-B3 CNN baseline for flood risk classification.
Used for performance comparison with the Vision Transformer.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import timm

from src.utils.config_loader import load_config

cfg = load_config()
model_cfg = cfg["models"]["cnn"]
NUM_CLASSES = cfg["data"]["num_classes"]


class FloodCNN(nn.Module):
    """
    EfficientNet-B3 fine-tuned for flood risk classification.

    Architecture:
        EfficientNet-B3 backbone (pretrained ImageNet)
        → Freeze early stages
        → Unfreeze last N stages
        → Custom classification head with dropout
    """

    def __init__(
        self,
        model_name: str = None,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = None,
        finetune_stages: int = None,
        dropout: float = None,
    ):
        super().__init__()

        self.model_name = model_name or model_cfg["name"]
        self.num_classes = num_classes
        self.pretrained = pretrained if pretrained is not None else model_cfg["pretrained"]
        self.finetune_stages = finetune_stages or model_cfg["finetune_stages"]

        # Load pretrained backbone
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=0,             # Remove head
            global_pool="avg",
        )

        in_features = self.backbone.num_features   # 1536 for B3

        # Classification head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout or model_cfg["dropout"]),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

        self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all layers then unfreeze the last N stages."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        # EfficientNet stages are in backbone.blocks
        # Unfreeze last finetune_stages block groups
        blocks = list(self.backbone.blocks)
        unfreeze_from = max(0, len(blocks) - self.finetune_stages)

        for block_group in blocks[unfreeze_from:]:
            for param in block_group.parameters():
                param.requires_grad = True

        # Always unfreeze conv_head and bn2
        for layer_name in ["conv_head", "bn2"]:
            layer = getattr(self.backbone, layer_name, None)
            if layer:
                for param in layer.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[CNN] Trainable: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.1f}%)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 3, H, W)

        Returns:
            logits: (B, num_classes)
        """
        features = self.backbone(x)        # (B, in_features)
        logits = self.classifier(features)
        return logits

    def forward_with_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features

    def get_gradcam_ready_model(self):
        """Return the target layer for GradCAM visualization."""
        # Last convolutional block before global pool
        return self.backbone.blocks[-1][-1].conv_pwl

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_cnn(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> FloodCNN:
    """
    Build and optionally load a CNN model.

    Args:
        num_classes: Output classes.
        pretrained: Whether to use ImageNet pretrained weights.
        checkpoint_path: Path to a saved fine-tuned checkpoint.
        device: Target device.

    Returns:
        FloodCNN model.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FloodCNN(num_classes=num_classes, pretrained=pretrained)

    if checkpoint_path and torch.os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"[CNN] Loaded checkpoint: {checkpoint_path}")

    return model.to(device)
