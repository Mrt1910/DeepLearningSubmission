# LULC Segmentation from Sentinel-1 SAR

Land Use and Land Cover (LULC) semantic segmentation using Sentinel-1 SAR data as the sole input.  
Model: U-Net with ResNet-34 encoder. Labels: ESA WorldCover 2020 (11 classes, 10 m).  
Data: Microsoft Planetary Computer (anonymous access, no account required).

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ required. CUDA 11.8+ recommended for training.

---

## Full pipeline (run in order)

### 1. Download training data

Downloads Sentinel-1 RTC monthly composites (all 12 months, 2020) and WorldCover 2020 labels for seven geographically diverse regions.

```bash
# Test with one region first
python -m src.data.download --regions germany_bw --months 1 4

# Download all seven regions, all twelve months
python -m src.data.download
```

Output: `data/raw/<region>/sar_2020_MM.tif` (float32 linear sigma0) and `worldcover_2020.tif` (uint8).

### 2. Preprocess into patches

Converts raw GeoTIFFs to normalised 512×512 patches: dB conversion → Lee speckle filter → label remapping → patch extraction → normalisation → augmentation → 70/15/15 split.

```bash
python -m src.data.preprocess --raw-dir data/raw --patch-dir data/patches
```

Output: `data/patches/{train,val,test}/patch_NNNNN.npz`, `band_stats.npz`.

### 3. Train the model

```bash
python -m src.training.train \
    --patch-dir data/patches \
    --checkpoint-dir checkpoints \
    --epochs 100 \
    --batch-size 16
```

Saves `checkpoints/best_model.pth` (best val mIoU), `history.json` (training curves), and `test_results.json` after final evaluation.

### 4. Run the prototype

Input: centre lat/lon, start year/month → 12 consecutive monthly LULC maps.

```bash
# Kyoto, Japan — starting January 2020
python -m src.prototype.run \
    --lat 35.0116 --lon 135.7681 \
    --year 2020 --month 1 \
    --checkpoint checkpoints/best_model.pth \
    --patch-dir data/patches \
    --output-dir results/kyoto_2020_01
```

Outputs (in `results/kyoto_2020_01/`):
- `figures/maps/lulc_YYYY_MM.png` — 12 colour-coded LULC maps
- `figures/timeseries.png` — stacked area chart of class composition over time
- `figures/change_bars.png` — bar chart of total class change
- `stats.json` — per-class percentages, month-to-month deltas, total change
- `pred/pred_YYYY_MM.tif` — GeoTIFF prediction rasters
- `sar/sar_YYYY_MM.tif` — downloaded SAR composites

---

## Project structure

```
exam_Martin/
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── download.py       # Planetary Computer → raw GeoTIFFs
│   │   ├── preprocess.py     # raw → normalised patches
│   │   └── dataset.py        # PyTorch Dataset / DataLoaders
│   ├── models/
│   │   ├── unet.py           # U-Net (ResNet-34 encoder)
│   │   └── losses.py         # CrossEntropy + Dice, class weights
│   ├── training/
│   │   ├── train.py          # training loop, mixed-precision, checkpointing
│   │   └── metrics.py        # OA, mIoU, per-class IoU/F1
│   └── prototype/
│       ├── fetch.py          # 12-month SAR download for a lat/lon point
│       ├── infer.py          # patch-and-stitch inference
│       ├── stats.py          # area percentages + change statistics
│       ├── viz.py            # figures
│       └── run.py            # CLI entry point
```

---

## WorldCover classes

| Index | Name                    |
|-------|-------------------------|
| 0     | Tree cover              |
| 1     | Shrubland               |
| 2     | Grassland               |
| 3     | Cropland                |
| 4     | Built-up                |
| 5     | Bare / sparse veg.      |
| 6     | Snow and ice            |
| 7     | Permanent water bodies  |
| 8     | Herbaceous wetland      |
| 9     | Mangroves               |
| 10    | Moss and lichen         |

No-data pixels are mapped to index 255 and excluded from loss and metrics.

---

## Key design decisions (report summary)

| Decision | Choice | Reason |
|---|---|---|
| SAR product | Sentinel-1 RTC (Planetary Computer) | Documented calibration + terrain correction pipeline; anonymous access |
| Labels | ESA WorldCover 2020 | Same sensor family, 10 m, 2020 temporal alignment |
| Composite | Monthly median | Multi-temporal speckle reduction without spatial blurring |
| Training months | All 12 months | Full phenological coverage → seasonal invariance |
| Training regions | 7 regions (Germany, Japan, Morocco, Brazil, Swiss Alps, Sundarbans, Finland) | Geographic and biome diversity; rare classes covered |
| Patch size | 512 × 512 | 5.12 km context; batch 16 fits in 48 GB VRAM |
| Encoder | ResNet-34 ImageNet | Transferable low-level features; manageable parameter count |
| Loss | CE (class-weighted) + Dice | Handles class imbalance; Dice stabilises rare classes |
| Augmentation | Flip + 90° rotation only | Geometric transforms are radiometrically valid for SAR |
| Normalisation | Per-band z-score (train stats only) | Prevents data leakage into val/test |
