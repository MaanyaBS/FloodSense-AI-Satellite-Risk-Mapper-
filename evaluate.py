"""
evaluate.py
Evaluate trained models and generate comparison reports.

Usage:
    python evaluate.py --model vit
    python evaluate.py --model cnn
    python evaluate.py --compare            # ViT vs CNN side-by-side
"""

import argparse
from pathlib import Path

import torch

from src.models.model_factory import get_model
from src.preprocessing.dataset import get_dataloaders
from src.utils.config_loader import load_config, set_seed
from src.utils.console import enable_utf8_console
from src.utils.evaluator import (
    compare_models,
    evaluate_model,
    print_comparison_report,
    print_evaluation_report,
)
from src.visualization.risk_map import plot_confusion_matrix, plot_model_comparison

cfg = load_config()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate flood risk models")
    parser.add_argument(
        "--model", type=str, default="vit", choices=["vit", "cnn"],
        help="Which model to evaluate"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare both models"
    )
    parser.add_argument(
        "--data_dir", type=str, default=cfg["data"]["processed_dir"]
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: project.seed from config.yaml)"
    )
    return parser.parse_args()


def load_trained_model(model_type: str, device: torch.device):
    """Load a model from its best checkpoint."""
    ckpt_key = f"{model_type}_checkpoint"
    ckpt_path = cfg["output"].get(ckpt_key, "")

    model = get_model(
        model_type=model_type,
        pretrained=False,         # Don't need ImageNet weights for eval
        checkpoint_path=ckpt_path if Path(ckpt_path).exists() else None,
        device=device,
    )
    return model


def main():
    args = parse_args()

    enable_utf8_console()

    # Same seed as training, so the weighted sampler and any augmentation
    # randomness replay identically and the comparison is apples-to-apples.
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    plots_dir = Path(cfg["output"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Data
    loaders = get_dataloaders(args.data_dir)
    test_loader = loaders["test"]

    if args.compare:
        print("\nLoading both models for comparison...")
        vit_model = load_trained_model("vit", device)
        cnn_model = load_trained_model("cnn", device)

        print("\nEvaluating ViT...")
        vit_results = evaluate_model(vit_model, test_loader, device)

        print("\nEvaluating CNN...")
        cnn_results = evaluate_model(cnn_model, test_loader, device)

        print_evaluation_report(vit_results, "ViT-B/16")
        print_evaluation_report(cnn_results, "EfficientNet-B3")

        comparison = compare_models(vit_results, cnn_results)
        print_comparison_report(comparison)

        # Confusion matrices
        plot_confusion_matrix(
            vit_results["confusion_matrix"],
            model_name="ViT-B/16",
            save_path=str(plots_dir / "vit_confusion_matrix.png"),
        )
        plot_confusion_matrix(
            cnn_results["confusion_matrix"],
            model_name="EfficientNet-B3",
            save_path=str(plots_dir / "cnn_confusion_matrix.png"),
        )

        # Load training histories if available
        try:
            import torch as th
            vit_ckpt = th.load(cfg["output"]["vit_checkpoint"], map_location="cpu")
            cnn_ckpt = th.load(cfg["output"]["cnn_checkpoint"], map_location="cpu")
            plot_model_comparison(
                vit_ckpt.get("history", {}),
                cnn_ckpt.get("history", {}),
                save_path=str(plots_dir / "model_comparison.png"),
            )
            print(f"\nComparison plot: {plots_dir / 'model_comparison.png'}")
        except Exception:
            pass

    else:
        print(f"\nEvaluating {args.model.upper()}...")
        model = load_trained_model(args.model, device)
        results = evaluate_model(model, test_loader, device)
        print_evaluation_report(results, args.model.upper())

        cm_path = plots_dir / f"{args.model}_confusion_matrix.png"
        plot_confusion_matrix(
            results["confusion_matrix"],
            model_name=args.model.upper(),
            save_path=str(cm_path),
        )
        print(f"\nConfusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
