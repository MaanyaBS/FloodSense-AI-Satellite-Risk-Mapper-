"""
src/utils/predictor.py
Inference pipeline: single image and batch prediction with
probability scores, risk classification, and alert generation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.preprocessing.augmentation import denormalize
from src.preprocessing.dataset import load_single_image
from src.utils.config_loader import load_config

cfg = load_config()
CLASS_NAMES = cfg["data"]["class_names"]
ALERT_THRESHOLD = cfg["prediction"]["alert_threshold"]
RISK_THRESHOLDS = cfg["prediction"]["risk_thresholds"]


# ---------------------------------------------------------------------------
# Prediction result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PredictionResult:
    image_path: str
    predicted_class: str
    predicted_idx: int
    class_probabilities: Dict[str, float]
    confidence: float
    risk_level: str
    alert: bool
    alert_message: str = ""
    attention_map: Optional[np.ndarray] = None
    gradcam_map: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "predicted_class": self.predicted_class,
            "predicted_idx": self.predicted_idx,
            "class_probabilities": self.class_probabilities,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "alert": self.alert,
            "alert_message": self.alert_message,
        }


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class FloodPredictor:
    """
    Unified inference pipeline supporting ViT and CNN models.

    Usage:
        predictor = FloodPredictor(model, model_type='vit')
        result = predictor.predict('path/to/image.jpg')
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "vit",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Single image prediction
    # ------------------------------------------------------------------
    def predict(self, image_input) -> PredictionResult:
        """
        Predict flood risk for a single image.

        Args:
            image_input: File path (str/Path) or PIL Image or
                         numpy array (H, W, 3) or tensor (1, 3, H, W).

        Returns:
            PredictionResult with full details.
        """
        tensor, image_path = self._prepare_input(image_input)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        class_probs = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
        risk_level = self._classify_risk(probs)
        alert = self._check_alert(probs)
        alert_msg = self._build_alert_message(pred_class, confidence) if alert else ""

        return PredictionResult(
            image_path=str(image_path),
            predicted_class=pred_class,
            predicted_idx=pred_idx,
            class_probabilities=class_probs,
            confidence=confidence,
            risk_level=risk_level,
            alert=alert,
            alert_message=alert_msg,
        )

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------
    def predict_batch(
        self, image_paths: List[str]
    ) -> List[PredictionResult]:
        """Predict flood risk for a list of image paths."""
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                print(f"  Error predicting {path}: {e}")
        return results

    def predict_folder(self, folder: str) -> List[PredictionResult]:
        """Predict all images in a folder."""
        folder_path = Path(folder)
        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        paths = [
            str(p) for p in folder_path.iterdir()
            if p.suffix.lower() in valid_exts
        ]
        print(f"Found {len(paths)} images in {folder}")
        return self.predict_batch(paths)

    # ------------------------------------------------------------------
    # Dual-model prediction (ViT + CNN comparison)
    # ------------------------------------------------------------------
    @staticmethod
    def compare_predictions(
        vit_result: PredictionResult,
        cnn_result: PredictionResult,
    ) -> Dict:
        """Compare predictions from ViT and CNN on the same image."""
        agreement = vit_result.predicted_class == cnn_result.predicted_class
        return {
            "agree": agreement,
            "vit_prediction": vit_result.predicted_class,
            "cnn_prediction": cnn_result.predicted_class,
            "vit_confidence": vit_result.confidence,
            "cnn_confidence": cnn_result.confidence,
            "vit_probs": vit_result.class_probabilities,
            "cnn_probs": cnn_result.class_probabilities,
            "consensus_risk": vit_result.risk_level if agreement
            else "Uncertain (models disagree)",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_input(self, image_input) -> Tuple[torch.Tensor, str]:
        """Convert various input formats to a (1, 3, H, W) tensor."""
        if isinstance(image_input, (str, Path)):
            tensor = load_single_image(str(image_input))
            return tensor, str(image_input)
        elif isinstance(image_input, Image.Image):
            from src.preprocessing.augmentation import get_val_transforms
            transform = get_val_transforms()
            return transform(image_input).unsqueeze(0), "<PIL Image>"  # ← FIXED
        elif isinstance(image_input, np.ndarray):
            from src.preprocessing.augmentation import get_val_transforms
            img = Image.fromarray(image_input.astype(np.uint8))
            transform = get_val_transforms()
            return transform(img).unsqueeze(0), "<numpy array>"
        elif isinstance(image_input, torch.Tensor):
            if image_input.dim() == 3:
                return image_input.unsqueeze(0), "<tensor>"
            return image_input, "<tensor>"
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

    def _classify_risk(self, probs: np.ndarray) -> str:
        """Map probability vector to human-readable risk level."""
        risk_score = sum(i * p for i, p in enumerate(probs)) / (len(probs) - 1)

        if risk_score < RISK_THRESHOLDS["low"]:
            return "🟢 Low Risk"
        elif risk_score < RISK_THRESHOLDS["medium"]:
            return "🟡 Moderate Risk"
        elif risk_score < RISK_THRESHOLDS["high"]:
            return "🟠 High Risk"
        else:
            return "🔴 Critical Risk"

    def _check_alert(self, probs: np.ndarray) -> bool:
        """Trigger alert if probability of high-risk or flooded exceeds threshold."""
        high_risk_prob = probs[-1] + probs[-2]  # Flooded + High Risk
        return bool(high_risk_prob > ALERT_THRESHOLD)

    def _build_alert_message(self, pred_class: str, confidence: float) -> str:
        return (
            f"⚠️ FLOOD ALERT: {pred_class} detected with "
            f"{confidence*100:.1f}% confidence. "
            f"Immediate action recommended."
        )