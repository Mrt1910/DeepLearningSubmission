"""
src/data/dataset.py

PyTorch Dataset for the LULC patch archive produced by preprocess.py.

Each .npz file contains:
    sar  — float32 (3, 512, 512): normalised VV_dB / VH_dB / VV-VH bands
    lbl  — uint8  (512, 512):     class indices 0-10, IGNORE_INDEX=255
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LULCDataset(Dataset):
    def __init__(self, split_dir: Path | str) -> None:
        self.files = sorted(Path(split_dir).glob("patch_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No patch_*.npz files found in {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx])
        sar = torch.from_numpy(data["sar"])                      # (3, H, W) float32
        lbl = torch.from_numpy(data["lbl"].astype(np.int64))    # (H, W)    int64
        return sar, lbl


def build_loaders(
    patch_dir: Path | str = Path("data/patches"),
    batch_size: int = 16,
    num_workers: int = 8,
) -> dict[str, DataLoader]:
    """
    Return a dict with 'train', 'val', 'test' DataLoaders.

    Batch size 16 at 512×512 × float32 × 3 bands ≈ 96 MB/batch; fits
    comfortably in the A6000's 48 GB VRAM alongside the model's activations.
    num_workers=8 pre-fetches from NVMe in parallel; pin_memory accelerates
    the CPU→GPU transfer for all three splits.
    """
    patch_dir = Path(patch_dir)
    loaders = {}
    for split in ("train", "val", "test"):
        ds = LULCDataset(patch_dir / split)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),  # avoids small last-batch BN instability
        )
    return loaders
