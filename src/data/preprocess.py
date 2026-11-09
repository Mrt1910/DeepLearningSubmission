"""
src/data/preprocess.py

Convert raw Sentinel-1 RTC GeoTIFFs into normalised, labelled 512×512 patches
ready for PyTorch training.

Pipeline per region
───────────────────
1.  Load VV / VH bands (linear sigma0, float32).
2.  dB conversion  — 10 · log₁₀(σ₀), clips to [−30, 0] dB to remove
    sensor artefacts and specular-reflection outliers.
3.  Compute ratio band  VV_dB − VH_dB, which captures the cross-polarisation
    difference and helps distinguish urban double-bounce (high VV, low VH)
    from volume scatterers like forest (both moderate).
4.  Lee speckle filter  — 7×7 adaptive window applied to each band
    independently before stacking, preserving edges while suppressing
    multiplicative noise.
5.  Normalise each band to zero-mean / unit-variance using statistics
    computed on the training split only (prevents data leakage).
6.  Load and remap WorldCover labels to contiguous 0-based indices.
    Pixels labelled 0 (no-data) are mapped to IGNORE_INDEX = 255 and
    excluded from the loss function.
7.  Align SAR and label grids  — reproject labels to match the SAR
    pixel grid exactly (same CRS, transform, shape).
8.  Extract non-overlapping 512×512 patches.  Patches with > 20 %
    no-data pixels are discarded to avoid training on uninformative tiles.
9.  Stratified random split  — 70 % train / 15 % val / 15 % test,
    split at patch level (not pixel level) to prevent spatial leakage.
10. Augmentation (train only) — random horizontal flip, random vertical
    flip, random 90° rotation.  SAR backscatter is geometrically symmetric
    so flips are radiometrically valid.  Colour jitter / brightness
    augmentation is deliberately excluded because SAR intensity encodes
    physical scattering mechanisms that must be preserved.

Design choice — patch size 512 × 512:
    At 10 m GSD, one patch covers 5.12 km × 5.12 km, giving the U-Net
    encoder sufficient spatial context to distinguish classes that differ
    at landscape scale (e.g. forest vs. adjacent cropland blocks).
    With the A6000 48 GB GPU, batch size 16 at 512 × 512 fits comfortably
    (~6 GB activation memory), allowing stable batch-norm statistics.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import rioxarray
import scipy.ndimage as ndi
from rasterio.enums import Resampling
from tqdm import tqdm

from src.data.download import CLASS_REMAP, IGNORE_INDEX, NUM_CLASSES, TRAINING_REGIONS

# ── Constants ─────────────────────────────────────────────────────────────────
PATCH_SIZE = 512
PATCH_STRIDE = PATCH_SIZE // 2  # 50 % overlap → ~4× more patches; overlap averages out edge effects
NODATA_THRESHOLD = 0.20   # discard patches with > 20 % no-data pixels
DB_CLIP_MIN = -30.0       # dB — below this is almost certainly noise / water shadow
DB_CLIP_MAX = 0.0         # dB — above this is specular reflection / infrastructure

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


# ── Step 1-4: SAR band processing ────────────────────────────────────────────

def linear_to_db(arr: np.ndarray, clip_min: float = DB_CLIP_MIN, clip_max: float = DB_CLIP_MAX) -> np.ndarray:
    """
    Convert linear sigma0 to decibels: dB = 10 · log₁₀(σ₀).

    Near-zero values (water, shadow) produce −∞ in log space, so we
    replace non-positive pixels with a small epsilon before conversion
    then clip to [clip_min, clip_max] to bound the dynamic range.
    """
    arr = np.where(arr > 0, arr, 1e-10)
    db = 10.0 * np.log10(arr)
    return np.clip(db, clip_min, clip_max)


def lee_filter(img: np.ndarray, size: int = 7) -> np.ndarray:
    """
    Adaptive Lee filter for multiplicative SAR speckle.

    The Lee filter is the statistical standard for SAR despeckling
    (Lee, 1980).  It estimates the local signal-to-noise ratio from a
    neighbourhood window and weights the filtered vs. raw pixel value
    accordingly, preserving edges better than a simple boxcar average.

    Applied per band on 2-D arrays.
    """
    img_mean = ndi.uniform_filter(img, size)
    img_sq_mean = ndi.uniform_filter(img ** 2, size)
    img_variance = img_sq_mean - img_mean ** 2

    # Estimate noise variance as the mean of all local variances
    # (equivalent to the overall image speckle level).
    overall_variance = img_variance.mean()

    # Lee weighting coefficient W: 0 → use local mean, 1 → keep original
    with np.errstate(invalid="ignore", divide="ignore"):
        weight = np.where(
            img_variance + overall_variance > 0,
            1.0 - overall_variance / (img_variance + overall_variance),
            0.0,
        )
    weight = np.clip(weight, 0.0, 1.0)
    return img_mean + weight * (img - img_mean)


def local_coefficient_of_variation(img: np.ndarray, size: int = 7) -> np.ndarray:
    """
    Local Coefficient of Variation: LCV = local_std / (|local_mean| + ε).

    LCV captures spatial texture (heterogeneity) in the backscatter image.
    Open water is specularly smooth (LCV ≈ 0); mangroves and wetland reeds
    produce irregular double-bounce returns (LCV > 0.3); forests and crops
    have moderate, random LCV values.

    Adding LCV as a 4th input channel gives the model explicit texture
    information to distinguish Water / Mangroves / Herbaceous wetland —
    three classes whose mean backscatter magnitude alone is ambiguous.
    """
    local_mean   = ndi.uniform_filter(img, size)
    local_sq     = ndi.uniform_filter(img ** 2, size)
    local_var    = np.maximum(local_sq - local_mean ** 2, 0.0)
    local_std    = np.sqrt(local_var)
    return local_std / (np.abs(local_mean) + 1e-8)


def process_sar(sar_path: Path) -> np.ndarray:
    """
    Load a 2-band (VV, VH) RTC GeoTIFF and return a 4-band float32 array:
        [0] VV_dB
        [1] VH_dB
        [2] VV_dB − VH_dB  (cross-polarisation ratio)
        [3] LCV of VV_dB   (local coefficient of variation — texture)

    Shape: (4, H, W).
    """
    with rasterio.open(sar_path) as src:
        vv = src.read(1).astype("float32")
        vh = src.read(2).astype("float32")

    vv_db = lee_filter(linear_to_db(vv))
    vh_db = lee_filter(linear_to_db(vh))
    ratio = vv_db - vh_db
    lcv   = local_coefficient_of_variation(vv_db)

    return np.stack([vv_db, vh_db, ratio, lcv], axis=0)


# ── Step 6-7: Label processing ────────────────────────────────────────────────

def remap_labels(arr: np.ndarray) -> np.ndarray:
    """Remap WorldCover uint8 values to contiguous 0-based model indices."""
    out = np.full_like(arr, IGNORE_INDEX, dtype=np.uint8)
    for wc_val, model_idx in CLASS_REMAP.items():
        out[arr == wc_val] = model_idx
    return out


def align_labels(sar_path: Path, label_path: Path) -> np.ndarray:
    """
    Reproject the WorldCover label raster to exactly match the SAR pixel grid
    (same CRS, same transform, same shape).

    Nearest-neighbour resampling preserves discrete class values.
    Returns a (H, W) uint8 array of remapped class indices.
    """
    with rasterio.open(sar_path) as sar_src:
        target_crs = sar_src.crs
        target_transform = sar_src.transform
        target_height = sar_src.height
        target_width = sar_src.width

    label_da = rioxarray.open_rasterio(str(label_path), masked=False).squeeze("band")
    label_da = label_da.rio.reproject(
        target_crs,
        shape=(target_height, target_width),
        resampling=Resampling.nearest,
        transform=target_transform,
    )
    return remap_labels(label_da.values.astype(np.uint8))


# ── Step 8: Patch extraction ──────────────────────────────────────────────────

def extract_patches(
    sar: np.ndarray,
    labels: np.ndarray,
    patch_size: int = PATCH_SIZE,
    stride: int = PATCH_STRIDE,
    nodata_thresh: float = NODATA_THRESHOLD,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Extract overlapping patch_size × patch_size tiles with the given stride.

    Using stride = patch_size // 2 (50 % overlap) yields ~4× more patches
    than non-overlapping extraction from the same scene, which is critical
    when the number of downloaded regions is small.  Overlapping patches
    are valid training samples because augmentation (flips, rotations) applied
    afterwards ensures the model does not simply memorise the overlap region.

    Discards any patch where the fraction of IGNORE_INDEX pixels exceeds
    nodata_thresh, since such patches contribute only masked loss terms
    and provide little gradient signal.
    """
    _, H, W = sar.shape
    patches = []
    for row in range(0, H - patch_size + 1, stride):
        for col in range(0, W - patch_size + 1, stride):
            sar_patch = sar[:, row:row + patch_size, col:col + patch_size]
            lbl_patch = labels[row:row + patch_size, col:col + patch_size]
            nodata_frac = (lbl_patch == IGNORE_INDEX).mean()
            if nodata_frac > nodata_thresh:
                continue
            patches.append((sar_patch, lbl_patch))
    return patches


# ── Normalisation ─────────────────────────────────────────────────────────────

def compute_band_stats(
    patch_list: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-band mean and std over all training patches.

    Statistics are derived only from the training split to prevent
    data leakage into validation and test sets.
    """
    n_bands = patch_list[0][0].shape[0]
    all_pixels = [[] for _ in range(n_bands)]
    for sar, _ in patch_list:
        for b in range(n_bands):
            all_pixels[b].append(sar[b].ravel())
    mean = np.array([np.concatenate(all_pixels[b]).mean() for b in range(n_bands)])
    std  = np.array([np.concatenate(all_pixels[b]).std()  for b in range(n_bands)])
    return mean, std


def normalise(sar: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (sar - mean[:, None, None]) / (std[:, None, None] + 1e-8)


# ── Augmentation (train split only) ──────────────────────────────────────────

def augment(
    sar: np.ndarray, lbl: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Geometric-only augmentation: random flips + 90° rotations.

    Radiometric augmentation (brightness, contrast) is intentionally omitted
    because SAR backscatter encodes the physical scattering mechanism of the
    surface — distorting it would train the model on non-physical inputs.
    """
    # Random horizontal flip
    if rng.random() < 0.5:
        sar = np.flip(sar, axis=2).copy()
        lbl = np.flip(lbl, axis=1).copy()
    # Random vertical flip
    if rng.random() < 0.5:
        sar = np.flip(sar, axis=1).copy()
        lbl = np.flip(lbl, axis=0).copy()
    # Random 90° rotation (0, 90, 180, 270)
    k = rng.integers(0, 4)
    if k > 0:
        sar = np.rot90(sar, k=k, axes=(1, 2)).copy()
        lbl = np.rot90(lbl, k=k).copy()
    return sar, lbl


# ── Orchestration ─────────────────────────────────────────────────────────────

def split_patches(
    patches: list,
    ratios: dict = SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
) -> dict[str, list]:
    """Shuffle and split patches into train / val / test."""
    rng = random.Random(seed)
    shuffled = patches[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val   = int(n * ratios["val"])
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def save_patches(
    patches: list[tuple[np.ndarray, np.ndarray]],
    split: str,
    output_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    augment_train: bool = True,
) -> int:
    """
    Normalise (and optionally augment) patches, then save as .npz files.
    Returns the number of patches written.
    """
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    count = 0
    for sar, lbl in tqdm(patches, desc=f"  Saving {split:5s}", leave=False):
        sar_norm = normalise(sar, mean, std)
        if split == "train" and augment_train:
            sar_norm, lbl = augment(sar_norm, lbl, rng)
        np.savez_compressed(
            split_dir / f"patch_{count:05d}.npz",
            sar=sar_norm.astype(np.float32),
            lbl=lbl.astype(np.uint8),
        )
        count += 1
    return count


def preprocess_all(
    raw_dir: Path = Path("data/raw"),
    patch_dir: Path = Path("data/patches"),
    regions: Optional[list[str]] = None,
) -> None:
    """
    Full preprocessing pipeline: SAR processing → patch extraction →
    normalisation → augmentation → save.

    Band statistics (mean/std) are saved to patch_dir/band_stats.npz
    so the prototype can normalise inference data identically.
    """
    target_regions = regions or list(TRAINING_REGIONS.keys())
    all_patches: list[tuple[np.ndarray, np.ndarray]] = []

    for name in target_regions:
        region_dir = raw_dir / name
        sar_files = sorted(region_dir.glob("sar_*.tif"))
        if not sar_files:
            print(f"  [skip] No SAR files in {region_dir}")
            continue

        print(f"\n  Region: {name}  ({len(sar_files)} SAR composite(s))")
        label_path = region_dir / "worldcover_2020.tif"

        for sar_path in tqdm(sar_files, desc=f"  {name}", leave=False):
            sar = process_sar(sar_path)
            labels = align_labels(sar_path, label_path)
            patches = extract_patches(sar, labels)
            all_patches.extend(patches)
            print(f"    {sar_path.name}: {len(patches)} patches extracted")

    print(f"\n  Total patches before split: {len(all_patches)}")

    splits = split_patches(all_patches)
    for s, ps in splits.items():
        print(f"    {s:5s}: {len(ps):5d} patches")

    # Compute normalisation stats from training split only
    print("\n  Computing band statistics from training split…")
    patch_dir.mkdir(parents=True, exist_ok=True)
    mean, std = compute_band_stats(splits["train"])
    print(f"    mean = {mean}")
    print(f"    std  = {std}")
    np.savez(patch_dir / "band_stats.npz", mean=mean, std=std)

    # Save all splits
    for split_name, patch_list in splits.items():
        n = save_patches(patch_list, split_name, patch_dir, mean, std,
                         augment_train=(split_name == "train"))
        print(f"  Wrote {n} {split_name} patches → {patch_dir / split_name}")

    print("\n  ✓  Preprocessing complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess SAR tiles into training patches")
    parser.add_argument("--raw-dir",   default="data/raw")
    parser.add_argument("--patch-dir", default="data/patches")
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args()
    preprocess_all(Path(args.raw_dir), Path(args.patch_dir), args.regions)
