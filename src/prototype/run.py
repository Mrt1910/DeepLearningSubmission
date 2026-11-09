"""
src/prototype/run.py

CLI entry point for the LULC prototype.

Usage:
    python src/prototype/run.py \\
        --lat 35.0116 --lon 135.7681 \\
        --year 2020 --month 1 \\
        --checkpoint checkpoints/best_model.pth \\
        --patch-dir data/patches \\
        --output-dir results/kyoto_2020_01

What it does:
    1.  Downloads 12 monthly Sentinel-1 RTC composites (lat±0.15°, lon±0.15°).
    2.  Runs the trained U-Net on each composite (patch-and-stitch inference).
    3.  Computes per-class area percentages, month-to-month changes, total change.
    4.  Saves 12 colour-coded LULC maps, a stacked time-series chart, and a
        change bar chart.
    5.  Prints a summary table to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.prototype.fetch import fetch_12_months
from src.prototype.infer import infer_scene
from src.prototype.stats import monthly_stats
from src.prototype.viz import save_change_bars, save_lulc_maps, save_timeseries
from src.data.download import WORLDCOVER_CLASSES

CLASS_NAMES = [WORLDCOVER_CLASSES[k] for k in sorted(WORLDCOVER_CLASSES)]


def _month_label(year: int, month_offset: int, start_month: int, start_year: int) -> str:
    month = (start_month - 1 + month_offset) % 12 + 1
    year  = start_year + (start_month - 1 + month_offset) // 12
    return f"{year}-{month:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LULC prototype: 12 monthly maps from Sentinel-1 SAR"
    )
    parser.add_argument("--lat",        type=float, required=True,  help="Centre latitude (°N)")
    parser.add_argument("--lon",        type=float, required=True,  help="Centre longitude (°E)")
    parser.add_argument("--year",       type=int,   required=True,  help="Start year")
    parser.add_argument("--month",      type=int,   required=True,  help="Start month (1-12)")
    parser.add_argument("--checkpoint", type=str,   default="checkpoints/best_model.pth")
    parser.add_argument("--patch-dir",  type=str,   default="data/patches")
    parser.add_argument("--output-dir", type=str,   default="results/run")
    args = parser.parse_args()

    output_dir     = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)
    patch_dir       = Path(args.patch_dir)
    sar_dir         = output_dir / "sar"
    pred_dir        = output_dir / "pred"
    fig_dir         = output_dir / "figures"

    print(f"\n{'='*60}")
    print(f"  LULC Prototype")
    print(f"  Location : {args.lat:.4f}°N  {args.lon:.4f}°E")
    print(f"  Period   : {args.year}-{args.month:02d} → 12 months")
    print(f"  Output   : {output_dir}")
    print(f"{'='*60}\n")

    # ── 1. Download SAR ──────────────────────────────────────────────────────
    print("Step 1/4 — Downloading 12 monthly SAR composites…")
    sar_paths = fetch_12_months(args.lat, args.lon, args.year, args.month, sar_dir)

    # ── 2. Inference ─────────────────────────────────────────────────────────
    print("\nStep 2/4 — Running U-Net inference on each month…")
    preds = []
    month_labels = []
    for i, sar_path in enumerate(sar_paths):
        label = _month_label(args.year, i, args.month, args.year)
        month_labels.append(label)
        if sar_path is None:
            print(f"  [skip] Month {label} — no SAR data")
            preds.append(None)
            continue
        pred_path = pred_dir / f"pred_{label.replace('-', '_')}.tif"
        pred, _ = infer_scene(sar_path, checkpoint_path, patch_dir, pred_path)
        preds.append(pred)
        print(f"  {label} → {pred_path.name}")

    # Fill missing months with the nearest valid prediction (forward-fill)
    for i in range(len(preds)):
        if preds[i] is None:
            prev = next((preds[j] for j in range(i - 1, -1, -1) if preds[j] is not None), None)
            preds[i] = prev if prev is not None else preds[0]

    # ── 3. Statistics ─────────────────────────────────────────────────────────
    print("\nStep 3/4 — Computing statistics…")
    stats = monthly_stats(preds)
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  Saved → {stats_path}")

    # ── 4. Visualisation ──────────────────────────────────────────────────────
    print("\nStep 4/4 — Saving figures…")
    save_lulc_maps(preds, month_labels, fig_dir)
    ts_path = save_timeseries(stats["monthly_pct"], month_labels, fig_dir)
    cb_path = save_change_bars(stats["total_change"], fig_dir)
    print(f"  Time-series  → {ts_path}")
    print(f"  Change bars  → {cb_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {month_labels[0]} to {month_labels[-1]}")
    print(f"{'='*60}")
    print(f"  {'Class':<30s}  {'Start%':>7s}  {'End%':>7s}  {'Change':>8s}")
    print(f"  {'-'*58}")
    for name in CLASS_NAMES:
        start = stats["monthly_pct"][0].get(name, 0)
        end   = stats["monthly_pct"][-1].get(name, 0)
        delta = stats["total_change"].get(name, 0)
        print(f"  {name:<30s}  {start:7.2f}  {end:7.2f}  {delta:+8.2f}")
    print(f"\n  All outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
