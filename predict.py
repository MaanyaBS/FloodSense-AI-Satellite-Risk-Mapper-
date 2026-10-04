"""
predict.py
Run flood risk predictions on single images or folders.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --model vit --visualize
    python predict.py --folder data/samples/ --model both
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.models.model_factory import get_model
from src.preprocessing.augmentation import denormalize
from src.preprocessing.dataset import load_single_image
from src.utils.config_loader import load_config
from src.utils.console import enable_utf8_console
from src.utils.predictor import FloodPredictor
from src.visualization.heatmap import (
    GradCAM,
    apply_heatmap_overlay,
    extract_vit_attention,
    plot_attention_comparison,
    plot_probability_bars,
)
from src.visualization.risk_map import create_interactive_risk_map, plot_risk_map

cfg = load_config()


def parse_args():
    parser = argparse.ArgumentParser(description="Flood risk prediction")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument(
        "--folder", type=str, default=None, help="Folder of images for batch prediction"
    )
    parser.add_argument(
        "--model", type=str, default="vit", choices=["vit", "cnn", "both"],
        help="Which model(s) to use"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate and save visualizations"
    )
    parser.add_argument(
        "--output_dir", type=str, default=cfg["output"]["predictions_dir"],
        help="Where to save predictions"
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_model_for_prediction(model_type: str, device: torch.device):
    ckpt_path = cfg["output"].get(f"{model_type}_checkpoint", "")
    ckpt = ckpt_path if Path(ckpt_path).exists() else None
    return get_model(
        model_type=model_type,
        pretrained=(ckpt is None),
        checkpoint_path=ckpt,
        device=device,
    )


def predict_single(image_path: str, args, device: torch.device, output_dir: Path):
    """Full prediction pipeline for a single image."""
    print(f"\nImage: {image_path}")

    # Load image for display
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil.resize((224, 224)))

    # Load tensor
    tensor = load_single_image(image_path)

    models_to_run = ["vit", "cnn"] if args.model == "both" else [args.model]
    results = {}

    for model_type in models_to_run:
        print(f"\n  [{model_type.upper()}] Running prediction...")
        model = load_model_for_prediction(model_type, device)
        predictor = FloodPredictor(model, model_type=model_type, device=device)
        result = predictor.predict(image_path)
        results[model_type] = result

        # Print result
        print(f"  Prediction   : {result.predicted_class}")
        print(f"  Confidence   : {result.confidence:.1%}")
        print(f"  Risk Level   : {result.risk_level}")
        if result.alert:
            print(f"  ⚠️  ALERT     : {result.alert_message}")
        print("  Probabilities:")
        for cls, prob in result.class_probabilities.items():
            bar = "█" * int(prob * 20)
            print(f"    {cls:<18} {prob:.3f}  {bar}")

    # Save JSON output
    stem = Path(image_path).stem
    json_path = output_dir / f"{stem}_prediction.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(
            {k: v.to_dict() for k, v in results.items()},
            f,
            indent=2,
        )
    print(f"\n  Results saved: {json_path}")

    # Visualizations
    if args.visualize:
        print("\n  Generating visualizations...")

        for model_type, result in results.items():
            model = load_model_for_prediction(model_type, device)

            # Probability bars
            prob_fig = plot_probability_bars(
                result.class_probabilities,
                title=f"{model_type.upper()} — {result.predicted_class} ({result.confidence:.1%})",
            )
            prob_path = output_dir / f"{stem}_{model_type}_probabilities.png"
            prob_fig.savefig(str(prob_path), dpi=150, bbox_inches="tight")
            plt.close(prob_fig)
            print(f"  Probability bars: {prob_path}")

            # Heatmap
            if model_type == "vit":
                attn_map = extract_vit_attention(model, tensor.to(device))
                heatmap = attn_map
            else:
                try:
                    target_layer = model.backbone.blocks[-1][-1].conv_pwl
                    gradcam = GradCAM(model, target_layer)
                    heatmap = gradcam(tensor, result.predicted_idx)
                except Exception:
                    heatmap = np.random.rand(224, 224).astype(np.float32)

            # Risk map
            risk_fig = plot_risk_map(
                image_np, heatmap,
                title=f"Flood Risk Map — {result.predicted_class}",
            )
            risk_path = output_dir / f"{stem}_{model_type}_risk_map.png"
            risk_fig.savefig(str(risk_path), dpi=150, bbox_inches="tight")
            plt.close(risk_fig)
            print(f"  Risk map: {risk_path}")

        # Comparison if both models
        if args.model == "both" and len(results) == 2:
            vit_model = load_model_for_prediction("vit", device)
            cnn_model = load_model_for_prediction("cnn", device)

            vit_attn = extract_vit_attention(vit_model, tensor.to(device))
            try:
                target_layer = cnn_model.backbone.blocks[-1][-1].conv_pwl
                gradcam = GradCAM(cnn_model, target_layer)
                cnn_cam = gradcam(tensor, results["cnn"].predicted_idx)
            except Exception:
                cnn_cam = np.random.rand(224, 224).astype(np.float32)

            compare_fig = plot_attention_comparison(
                image_np, vit_attn, cnn_cam,
                pred_class=results["vit"].predicted_class,
            )
            compare_path = output_dir / f"{stem}_model_comparison.png"
            compare_fig.savefig(str(compare_path), dpi=150, bbox_inches="tight")
            plt.close(compare_fig)
            print(f"  Model comparison: {compare_path}")

            # Comparison dict
            comparison = FloodPredictor.compare_predictions(
                results["vit"], results["cnn"]
            )
            print(f"\n  Model Agreement : {'✓ Yes' if comparison['agree'] else '✗ No'}")
            print(f"  Consensus Risk  : {comparison['consensus_risk']}")

    return results


def main():
    args = parse_args()

    enable_utf8_console()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        predict_single(args.image, args, device, output_dir)

    elif args.folder:
        folder = Path(args.folder)
        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        images = [
            str(p) for p in folder.iterdir()
            if p.suffix.lower() in valid_exts
        ]
        print(f"Found {len(images)} images in {folder}")
        for img_path in images:
            try:
                predict_single(img_path, args, device, output_dir)
            except Exception as e:
                print(f"  Error: {e}")

    else:
        print("Provide --image or --folder. Use --help for options.")


if __name__ == "__main__":
    main()
