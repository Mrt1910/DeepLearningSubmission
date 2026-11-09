"""
src/training/train.py

Training loop for the LULC U-Net segmentation model.

Key design choices (report-defensible)
───────────────────────────────────────
Mixed-precision (torch.amp):
  float16 activations halve memory, allowing batch size 16 at 512×512 on the
  A6000.  The GradScaler prevents underflow in the backward pass.

AdamW + CosineAnnealingLR:
  AdamW decouples L2 weight decay from the adaptive learning-rate update,
  which improves generalisation vs. vanilla Adam.  Cosine annealing drives
  the LR smoothly to zero over the full training run, avoiding the sharp LR
  drop of StepLR.  No warm restarts: the dataset is large enough that a
  single cosine cycle (T_max = max_epochs) suffices.

Early stopping on validation mIoU (patience=10 epochs):
  Prevents over-fitting and saves compute.  The best checkpoint is restored
  automatically when training finishes.

Gradient clipping (max_norm=1.0):
  Prevents exploding gradients in early epochs when the encoder weights are
  adapting from ImageNet features to SAR features.

Usage
─────
  python src/training/train.py \\
      --patch-dir data/patches \\
      --checkpoint-dir checkpoints \\
      --epochs 50 \\
      --batch-size 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.data.dataset import build_loaders
from src.models.losses import SegmentationLoss, compute_class_weights
from src.models.unet import build_unet
from src.training.metrics import (
    build_metrics,
    compute_metrics,
    format_metrics,
    reset_metrics,
    update_metrics,
)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    metrics,
) -> float:
    model.train()
    reset_metrics(metrics)
    total_loss = 0.0

    for sar, lbl in tqdm(loader, desc="  train", leave=False):
        sar = sar.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            logits = model(sar)
            loss = criterion(logits, lbl)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        with torch.no_grad():
            update_metrics(metrics, logits, lbl)

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, metrics) -> tuple[float, dict]:
    model.eval()
    reset_metrics(metrics)
    total_loss = 0.0

    for sar, lbl in tqdm(loader, desc="  eval ", leave=False):
        sar = sar.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type):
            logits = model(sar)
            loss = criterion(logits, lbl)
        total_loss += loss.item()
        update_metrics(metrics, logits, lbl)

    return total_loss / len(loader), compute_metrics(metrics)


def train(
    patch_dir: Path,
    checkpoint_dir: Path,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    max_epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 8,
    patience: int = 20,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = build_loaders(patch_dir, batch_size=batch_size, num_workers=num_workers)
    print(f"Patches — train:{len(loaders['train'].dataset)}  "
          f"val:{len(loaders['val'].dataset)}  "
          f"test:{len(loaders['test'].dataset)}")

    # ── Class weights ─────────────────────────────────────────────────────────
    weights_path = patch_dir / "class_weights.npy"
    if weights_path.exists():
        import numpy as np
        class_weights = torch.tensor(
            np.load(weights_path), dtype=torch.float32
        ).to(device)
        print("Loaded cached class weights.")
    else:
        print("Computing class weights from training split…")
        class_weights = compute_class_weights(patch_dir).to(device)

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    model = build_unet(encoder_name=encoder_name, encoder_weights=encoder_weights)
    model = model.to(device)
    criterion = SegmentationLoss(class_weights=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # ReduceLROnPlateau halves the LR when val mIoU stops improving for 5 epochs.
    # This adapts to the loss landscape rather than decaying on a fixed schedule,
    # which is important for a heterogeneous multi-biome dataset where convergence
    # speed varies between runs.
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=5, min_lr=lr / 100)
    scaler = torch.amp.GradScaler()
    metrics = build_metrics(device)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_miou = 0.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs}")

        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion,
                                     scaler, device, metrics)
        train_results = compute_metrics(metrics)

        val_loss, val_results = evaluate(model, loaders["val"], criterion, device, metrics)
        scheduler.step(val_results["miou"])  # reduce LR when mIoU plateaus

        miou = val_results["miou"]
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  train loss={train_loss:.4f}  val loss={val_loss:.4f}  "
              f"val mIoU={miou:.4f}  val OA={val_results['oa']:.4f}  LR={current_lr:.2e}")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_miou": train_results["miou"],
            "val_miou": miou,
            "train_oa": train_results["oa"],
            "val_oa": val_results["oa"],
        }
        history.append(row)
        with open(checkpoint_dir / "history.json", "w") as fh:
            json.dump(history, fh, indent=2)

        if miou > best_miou:
            best_miou = miou
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_miou": miou,
                    "encoder_name": encoder_name,
                    "in_channels": model.encoder.conv1.in_channels,
                },
                checkpoint_dir / "best_model.pth",
            )
            print(f"  ✓  New best mIoU={best_miou:.4f} → saved checkpoint")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping after {epoch} epochs (no improvement for {patience}).")
                break

    # ── Final evaluation on test split ────────────────────────────────────────
    print("\n  Loading best checkpoint for test evaluation…")
    ckpt = torch.load(checkpoint_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_results = evaluate(model, loaders["test"], criterion, device, metrics)
    print(f"\n  Test  loss={test_loss:.4f}")
    print(format_metrics(test_results))

    with open(checkpoint_dir / "test_results.json", "w") as fh:
        json.dump(test_results, fh, indent=2)
    print(f"\n  Results saved to {checkpoint_dir / 'test_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir",       default="data/patches")
    parser.add_argument("--checkpoint-dir",  default="checkpoints")
    parser.add_argument("--encoder",         default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight-decay",type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int,   default=8)
    parser.add_argument("--patience",    type=int,   default=20)
    args = parser.parse_args()

    train(
        patch_dir       = Path(args.patch_dir),
        checkpoint_dir  = Path(args.checkpoint_dir),
        encoder_name    = args.encoder,
        encoder_weights = args.encoder_weights,
        max_epochs      = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        num_workers     = args.num_workers,
        patience        = args.patience,
    )
