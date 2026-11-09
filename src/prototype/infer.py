"""
src/prototype/infer.py

Run the trained U-Net on a single SAR GeoTIFF and produce a full-scene
segmentation map using a sliding-window (patch-and-stitch) approach.

Patch-and-stitch with overlap
──────────────────────────────
Inference on a 32 km × 32 km scene at 10 m is ~3200 × 3200 pixels, which
fits in one forward pass on the A6000 (48 GB) but patch-and-stitch is used
anyway because:
  1. The prototype code stays the same regardless of GPU memory.
  2. Overlapping patches allow averaging of logits at seam boundaries,
     eliminating the tile-boundary artefacts that non-overlapping stitching
     can produce at class transitions.

Logits are accumulated in a (C, H, W) float32 accumulator array; a count
map tracks how many patches contributed to each pixel.  After all patches,
the accumulator is divided by the count and argmax is taken.

Preprocessing at inference:
  Same pipeline as training: linear→dB→Lee filter→normalise with the
  band statistics saved during preprocess_all().  This uses the functions
  from preprocess.py directly — no code duplication.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch

from src.data.download import IGNORE_INDEX, NUM_CLASSES
from src.models.unet import build_unet

INFER_PATCH_SIZE = 512
INFER_STRIDE     = 256    # 50 % overlap


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder_name = ckpt.get("encoder_name", "resnet50")
    in_channels  = ckpt.get("in_channels", 4)
    model = build_unet(encoder_name=encoder_name, encoder_weights=None,
                       in_channels=in_channels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


def load_band_stats(patch_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(patch_dir / "band_stats.npz")
    return stats["mean"].astype(np.float32), stats["std"].astype(np.float32)


def preprocess_sar_for_inference(sar_path: Path, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Return normalised (3, H, W) float32 array, identical pipeline to training."""
    from src.data.preprocess import process_sar, normalise
    sar = process_sar(sar_path)
    return normalise(sar, mean, std)


def _forward_patch(model, patch_np, device, use_tta=True):
    """
    Run one patch through the model, optionally with Test-Time Augmentation.

    TTA applies horizontal flip, vertical flip, and both together (4 total
    orientations), runs each through the model, undoes the flip on the output
    probability map, then averages.  This exploits the fact that SAR backscatter
    is geometrically symmetric — a flipped scene is physically valid — and
    typically adds 1–2% OA with no additional training.
    """
    augmentations = [(False, False), (True, False), (False, True), (True, True)]
    accum = None
    for hflip, vflip in augmentations:
        p = patch_np.copy()
        if hflip: p = np.flip(p, axis=2).copy()
        if vflip: p = np.flip(p, axis=1).copy()
        t = torch.from_numpy(p).unsqueeze(0).to(device)
        with torch.amp.autocast(device_type=device.type):
            logits = model(t)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        if hflip: probs = np.flip(probs, axis=2).copy()
        if vflip: probs = np.flip(probs, axis=1).copy()
        accum = probs if accum is None else accum + probs
        if not use_tta:
            break
    return accum / (4 if use_tta else 1)


def predict_scene(
    sar_norm: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int = INFER_PATCH_SIZE,
    stride: int = INFER_STRIDE,
    num_classes: int = NUM_CLASSES,
    use_tta: bool = True,
) -> np.ndarray:
    """
    Slide a window over the scene, accumulate softmax logits, return argmax.

    Returns (H, W) int16 prediction map.
    """
    _, H, W = sar_norm.shape
    accum  = np.zeros((num_classes, H, W), dtype=np.float32)
    counts = np.zeros((H, W),              dtype=np.float32)

    rows = list(range(0, H - patch_size + 1, stride))
    cols = list(range(0, W - patch_size + 1, stride))

    # Include the edge patch if the scene doesn't divide evenly
    if rows[-1] + patch_size < H:
        rows.append(H - patch_size)
    if cols[-1] + patch_size < W:
        cols.append(W - patch_size)

    with torch.no_grad():
        for r in rows:
            for c in cols:
                patch = sar_norm[:, r:r + patch_size, c:c + patch_size]
                probs = _forward_patch(model, patch, device, use_tta=use_tta)
                accum[:, r:r + patch_size, c:c + patch_size] += probs
                counts[r:r + patch_size, c:c + patch_size]   += 1.0

    counts = np.where(counts > 0, counts, 1)
    avg_probs = accum / counts
    return avg_probs.argmax(axis=0).astype(np.int16)


def infer_scene(
    sar_path: Path,
    checkpoint_path: Path,
    patch_dir: Path,
    output_path: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Full inference pipeline for one monthly composite.

    Returns (prediction_map (H,W), rasterio_profile_dict).
    Optionally saves as GeoTIFF to output_path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    mean, std = load_band_stats(patch_dir)
    sar_norm = preprocess_sar_for_inference(sar_path, mean, std)

    with rasterio.open(sar_path) as src:
        profile = src.profile.copy()

    profile.update(count=1, dtype="int16", compress="lzw", nodata=IGNORE_INDEX)
    pred = predict_scene(sar_norm, model, device)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(pred[np.newaxis, :, :])

    return pred, profile
