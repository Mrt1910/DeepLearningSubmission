"""
src/training/metrics.py

Segmentation metrics via torchmetrics: Overall Accuracy, mean IoU,
per-class IoU, and per-class F1.

All metrics exclude IGNORE_INDEX=255 pixels via the `ignore_index` argument
supported natively by torchmetrics >= 1.0.
"""

from __future__ import annotations

import numpy as np
import torch
import torchmetrics

from src.data.download import IGNORE_INDEX, NUM_CLASSES, WORLDCOVER_CLASSES


CLASS_NAMES = [WORLDCOVER_CLASSES[k] for k in sorted(WORLDCOVER_CLASSES)]


def build_metrics(device: torch.device | str) -> dict[str, torchmetrics.Metric]:
    """
    Return a dict of torchmetrics objects, all moved to `device`.

    Keys: 'oa', 'miou', 'iou_per_class', 'f1_per_class'
    """
    kwargs = dict(
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        task="multiclass",
    )
    metrics = {
        "oa":           torchmetrics.Accuracy(**kwargs),
        "miou":         torchmetrics.JaccardIndex(**kwargs, average="macro"),
        "iou_per_class":torchmetrics.JaccardIndex(**kwargs, average="none"),
        "f1_per_class": torchmetrics.F1Score(**kwargs, average="none"),
    }
    return {k: v.to(device) for k, v in metrics.items()}


def reset_metrics(metrics: dict) -> None:
    for m in metrics.values():
        m.reset()


def update_metrics(
    metrics: dict,
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """preds: (B, C, H, W) logits  |  targets: (B, H, W) int64"""
    pred_classes = preds.argmax(dim=1)
    for m in metrics.values():
        m.update(pred_classes, targets)


def compute_metrics(metrics: dict) -> dict[str, float | list[float]]:
    """
    Return a plain-Python dict of scalar / list values.

    mIoU is computed only over classes that actually appear in the evaluated
    split (IoU > 0 or class is present in ground truth).  Averaging over
    genuinely absent classes (e.g. Mangroves in a temperate training region)
    would unfairly penalise the model for not predicting a class it has never
    seen and that does not exist in the scene.
    """
    iou_per_class = metrics["iou_per_class"].compute().tolist()
    f1_per_class  = metrics["f1_per_class"].compute().tolist()

    # mIoU over present classes only
    present = [iou for iou in iou_per_class if iou > 0.0]
    miou = float(np.mean(present)) if present else 0.0

    results: dict = {}
    results["oa"]            = metrics["oa"].compute().item()
    results["miou"]          = miou
    results["miou_all"]      = metrics["miou"].compute().item()   # kept for reference
    results["iou_per_class"] = iou_per_class
    results["f1_per_class"]  = f1_per_class
    return results


def format_metrics(results: dict) -> str:
    lines = [
        f"OA   : {results['oa']:.4f}",
        f"mIoU : {results['miou']:.4f}",
        "Per-class IoU / F1:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        iou = results["iou_per_class"][i]
        f1  = results["f1_per_class"][i]
        lines.append(f"  {name:<28s}  IoU={iou:.3f}  F1={f1:.3f}")
    return "\n".join(lines)
