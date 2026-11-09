"""
src/models/losses.py

Combined loss for LULC segmentation: weighted CrossEntropy + Dice.

Design rationale
────────────────
CrossEntropyLoss alone is prone to class-imbalance dominance: rare classes
(e.g. Snow/Ice class 6 in temperate zones, Mangroves class 9 overall) yield
tiny gradient contributions compared to the majority classes (Tree cover,
Built-up).  We counter this with two complementary mechanisms:

1. Class weights — inverse-frequency weights computed from the training patch
   distribution, clipped at MAX_CLASS_WEIGHT to prevent over-correction for
   extremely rare classes that appear in only a handful of patches and whose
   few examples would otherwise dominate training.

2. Dice loss — penalises the overlap between predicted and ground-truth
   segmentation masks; naturally balances classes because it normalises by
   the union (Dice denominator = |pred| + |gt|, so a small class with few
   correct pixels still contributes meaningfully).

Combining CE + Dice with equal weight (λ=0.5 each) has been validated in
multiple medical and remote-sensing segmentation benchmarks (e.g. Isensee
et al., nnU-Net, 2021).  Both losses respect IGNORE_INDEX=255 so no-data
pixels are excluded.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.download import IGNORE_INDEX, NUM_CLASSES

MAX_CLASS_WEIGHT = 10.0  # cap to prevent extreme up-weighting of near-absent classes


class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.

    Computes Dice per class over the batch, ignores IGNORE_INDEX pixels,
    then averages across classes that are actually present in the batch.
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX, smooth: float = 1.0) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W)  targets: (B, H, W) int64
        valid = targets != self.ignore_index
        targets_clean = targets.clone()
        targets_clean[~valid] = 0  # dummy index — these pixels are masked out

        probs = F.softmax(logits, dim=1)   # (B, C, H, W)
        B, C, H, W = probs.shape

        # One-hot encode targets: (B, C, H, W)
        one_hot = F.one_hot(targets_clean, num_classes=C).permute(0, 3, 1, 2).float()

        # Zero out invalid positions in both pred and target
        mask = valid.unsqueeze(1).float()
        probs = probs * mask
        one_hot = one_hot * mask

        # Dice per class (sum over B, H, W)
        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        cardinality  = (probs + one_hot).sum(dim=(0, 2, 3))

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Average only over classes present in this batch
        present = one_hot.sum(dim=(0, 2, 3)) > 0
        if present.any():
            return 1.0 - dice_per_class[present].mean()
        return torch.tensor(0.0, device=logits.device)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Lin et al., 2017).

    Focal Loss down-weights easy examples (high confidence correct predictions)
    and focuses training on hard, misclassified examples.  This is complementary
    to class weighting: class weights address frequency imbalance, while focal
    loss addresses difficulty imbalance within each class.

    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    γ=2 is the standard value from the original paper.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index, reduction="none"
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)                    # (B, H, W)
        pt = torch.exp(-ce_loss)                              # probability of correct class
        focal = ((1 - pt) ** self.gamma) * ce_loss
        valid = targets != self.ignore_index
        return focal[valid].mean() if valid.any() else focal.mean()


class SegmentationLoss(nn.Module):
    """
    Combined CrossEntropyLoss (with class weights) + DiceLoss.

    Focal Loss was trialled but removed after empirical results showed it
    destabilised training on the mixed-biome SAR dataset: its sharp gradients
    for confident-but-wrong predictions caused water/wetland/mangrove confusion
    to compound rather than resolve.  CE+Dice is a simpler, more stable
    combination that has been validated across remote-sensing benchmarks.

    Parameters
    ----------
    class_weights : 1-D float32 tensor of length NUM_CLASSES, or None.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)


def compute_class_weights(
    patch_dir,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
    max_weight: float = MAX_CLASS_WEIGHT,
) -> torch.Tensor:
    """
    Scan all training-split patches, count pixels per class, return inverse-
    frequency weights clipped to [1, max_weight] and normalised so the mean
    weight is 1.0.

    Saved as data/patches/class_weights.npy and returned as a CPU tensor.
    """
    from pathlib import Path
    patch_dir = Path(patch_dir)
    counts = np.zeros(num_classes, dtype=np.int64)

    for f in sorted((patch_dir / "train").glob("patch_*.npz")):
        lbl = np.load(f)["lbl"]
        for c in range(num_classes):
            counts[c] += (lbl == c).sum()

    counts = counts.astype(np.float64)
    counts = np.where(counts == 0, 1, counts)  # avoid divide-by-zero for absent classes
    inv_freq = counts.sum() / (num_classes * counts)
    inv_freq = np.clip(inv_freq, 1.0, max_weight)
    inv_freq /= inv_freq.mean()  # normalise so loss scale is unchanged on average

    np.save(patch_dir / "class_weights.npy", inv_freq.astype(np.float32))
    return torch.tensor(inv_freq, dtype=torch.float32)
