"""
src/utils/trainer.py
Complete training pipeline with mixed precision, early stopping,
learning rate scheduling, and checkpoint management.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config_loader import load_config


# ---------------------------------------------------------------------------
# FIX 1: Reduce CPU memory pressure (important for Windows + CPU training)
# ---------------------------------------------------------------------------
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"


cfg = load_config()
train_cfg = cfg["training"]


# ---------------------------------------------------------------------------
# Utility: warm-up + cosine scheduler
# ---------------------------------------------------------------------------
class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_epochs:
            scale = self._step / max(self.warmup_epochs, 1)
        else:
            progress = (self._step - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            scale = 0.5 * (1.0 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = self.min_lr + (base_lr - self.min_lr) * scale

    def get_last_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """
    End-to-end training loop for flood risk models.

    Features:
    - Mixed precision (fp16) training
    - Warmup + cosine LR scheduling
    - Class-weighted cross-entropy loss
    - Early stopping
    - Best model checkpointing
    - Per-epoch metrics logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str = "model",
        class_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Loss function
        weights = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

        # Optimizer — separate LR for backbone vs head
        backbone_params = [
            p for n, p in model.named_parameters()
            if "classifier" not in n and p.requires_grad
        ]
        head_params = [
            p for n, p in model.named_parameters()
            if "classifier" in n and p.requires_grad
        ]

        self.optimizer = AdamW(
            [
                {"params": backbone_params, "lr": train_cfg["learning_rate"] * 0.1},
                {"params": head_params, "lr": train_cfg["learning_rate"]},
            ],
            weight_decay=train_cfg["weight_decay"],
        )

        # LR scheduler
        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_epochs=train_cfg["warmup_epochs"],
            total_epochs=train_cfg["epochs"],
            min_lr=train_cfg["min_lr"],
        )

        # Mixed precision (disabled automatically on CPU)
        self.use_amp = train_cfg["mixed_precision"] and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # History
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.output_dir = Path(cfg["output"]["models_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------
    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="  Train", leave=False, ncols=90)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), train_cfg["gradient_clip"]
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.3f}",
            )

        return total_loss / total, correct / total

    @torch.no_grad()
    def _val_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(self.val_loader, desc="  Val  ", leave=False, ncols=90):
            images, labels = images.to(self.device), labels.to(self.device)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def train(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        n_epochs = epochs or train_cfg["epochs"]
        patience = train_cfg["early_stopping_patience"]

        print(f"\n{'='*60}")
        print(f"  Training {self.model_name.upper()} for {n_epochs} epochs")
        print(f"  Device: {self.device}  |  AMP: {self.use_amp}")
        print(f"{'='*60}\n")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._val_epoch()

            self.scheduler.step()
            elapsed = time.time() - t0
            lr = self.scheduler.get_last_lr()[0]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(lr)

            improved = ""
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_acc)
                improved = " ✓ (best)"
            else:
                self.patience_counter += 1

            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"T-Loss: {train_loss:.4f}  T-Acc: {train_acc:.4f} | "
                f"V-Loss: {val_loss:.4f}  V-Acc: {val_acc:.4f} | "
                f"LR: {lr:.2e} | {elapsed:.1f}s{improved}"
            )

            if self.patience_counter >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement.")
                break

        print(f"\nTraining complete. Best val acc: {self.best_val_acc:.4f}")
        return self.history

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, val_acc: float):
        ckpt_path = self.output_dir / f"{self.model_name}_best.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_acc": val_acc,
                "history": self.history,
            },
            ckpt_path,
        )

    def load_best_checkpoint(self):
        ckpt_path = self.output_dir / f"{self.model_name}_best.pth"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model_state_dict"])
            print(f"Loaded best checkpoint (val_acc={state['val_acc']:.4f})")
        else:
            print(f"No checkpoint found at {ckpt_path}")