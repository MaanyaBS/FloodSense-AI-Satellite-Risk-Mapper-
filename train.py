"""
train.py
Main entry point for training ViT or CNN flood risk models.

Usage:
    python train.py --model vit --epochs 30
    python train.py --model cnn --epochs 30 --batch_size 32
    python train.py --model vit --data_dir data/processed
"""

import argparse
import os
import sys
from pathlib import Path
import multiprocessing

import torch

from src.models.model_factory import get_model
from src.preprocessing.dataset import get_dataloaders, prepare_dataset
from src.utils.config_loader import load_config
from src.utils.trainer import Trainer
from src.visualization.risk_map import plot_training_history

cfg = load_config()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train flood risk classification models"
    )
    parser.add_argument(
        "--model", type=str, default="vit", choices=["vit", "cnn"],
        help="Model architecture to train"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: from config)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (default: from config)"
    )
    parser.add_argument(
        "--data_dir", type=str, default=cfg["data"]["processed_dir"],
        help="Path to processed dataset directory"
    )
    parser.add_argument(
        "--raw_dir", type=str, default=cfg["data"]["raw_dir"],
        help="Path to raw dataset (for auto-preparation)"
    )
    parser.add_argument(
        "--prepare", action="store_true",
        help="Prepare/split raw dataset before training"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate synthetic demo data and train"
    )
    parser.add_argument(
        "--no_pretrained", action="store_true",
        help="Train from scratch without ImageNet weights"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda / cpu (auto-detected if not specified)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Flood Risk Mapper — Training")
    print(f"  Model: {args.model.upper()}  |  Device: {device}")
    print(f"{'='*60}")

    # Demo data generation
    if args.demo:
        print("\nGenerating synthetic demo data...")
        from src.utils.generate_demo_data import generate_demo_data
        generate_demo_data(n_per_class=150)
        args.prepare = True

    # Dataset preparation
    if args.prepare or not Path(args.data_dir).exists():
        print("\nPreparing dataset splits...")
        prepare_dataset(args.raw_dir, args.data_dir)

    # Data loaders
    print("\nLoading data...")
    loaders = get_dataloaders(
        processed_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # Model
    print("\nBuilding model...")
    model = get_model(
        model_type=args.model,
        pretrained=not args.no_pretrained,
        device=device,
    )

    # Class weights for imbalanced data
    class_weights = loaders["train"].dataset.get_class_weights()
    print(f"Class weights: {class_weights.numpy().round(3)}")

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        model_name=args.model,
        class_weights=class_weights,
        device=device,
    )

    # Train
    history = trainer.train(epochs=args.epochs)

    # Save training curves
    plots_dir = Path(cfg["output"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plots_dir / f"{args.model}_training_history.png"

    plot_training_history(
        history,
        model_name=args.model.upper(),
        save_path=str(plot_path)
    )

    print(f"\nTraining curves saved to: {plot_path}")

    # Final evaluation on test set
    print("\nRunning final test evaluation...")

    #trainer.load_best_checkpoint()

    from src.utils.evaluator import evaluate_model, print_evaluation_report
    from src.visualization.risk_map import plot_confusion_matrix

    results = evaluate_model(model, loaders["test"], device=device)

    print_evaluation_report(results, model_name=args.model.upper())

    cm_path = plots_dir / f"{args.model}_confusion_matrix.png"

    plot_confusion_matrix(
        results["confusion_matrix"],
        model_name=args.model.upper(),
        save_path=str(cm_path),
    )

    print(f"Confusion matrix saved to: {cm_path}")
    print(f"\nCheckpoint: {cfg['output'][f'{args.model}_checkpoint']}")


# ---------------------------------------------------------------------
# Windows multiprocessing fix
# ---------------------------------------------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()