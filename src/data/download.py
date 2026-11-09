"""
src/data/download.py

Download Sentinel-1 RTC composites and ESA WorldCover 2020 labels
from Microsoft Planetary Computer for the LULC training dataset.

Design decisions
────────────────
Sentinel-1 RTC (Radiometrically Terrain Corrected):
  Microsoft processed all GRD scenes against the 30 m Copernicus DEM,
  applying orbit correction, thermal-noise removal, radiometric calibration
  (sigma0 in linear power) and terrain flattening. Using this product means
  calibration and terrain correction are handled by a documented, reproducible
  pipeline rather than a local SNAP installation, while still leaving dB
  conversion, speckle filtering and normalisation for our own preprocessing
  step — which keeps the report's preprocessing section honest and complete.

ESA WorldCover 2020 (10 m, 11 classes):
  Produced by ESA from Sentinel-1 + Sentinel-2 composites of calendar year
  2020, giving temporal alignment with our SAR data.  At 10 m it matches the
  native resolution of Sentinel-1 IW GRD, minimising label-imagery mismatch.

Median composite:
  Taking the pixel-wise median over all acquisitions in a time window acts as
  multi-temporal speckle filtering without blurring spatial edges, and is
  robust to outliers (wind-roughened water, residual RFI, wet-soil events).
  We use seasonal snapshots (Jan, Apr, Jul, Oct) for training so the model
  learns to map SAR backscatter patterns to land cover across different
  phenological states — which is exactly what the prototype does at inference.

UTM per region:
  Each training bbox uses the local UTM zone so all pixel arithmetic is in
  metres, consistent with the 10 m resolution specification.
"""

from __future__ import annotations

import calendar
import os
from pathlib import Path
from typing import Optional

import dask
import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rioxarray
import stackstac
import xarray as xr
from tqdm import tqdm

# Parallelise dask across available cores, capped to avoid OOM on large stacks.
_N_WORKERS = min(16, (os.cpu_count() or 4))
dask.config.set(scheduler="threads", num_workers=_N_WORKERS)

# ── WorldCover class map (kept here so it is importable by preprocessing) ────
WORLDCOVER_CLASSES: dict[int, str] = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# Remapped to contiguous 0-based indices used inside the model.
# Keys = original WorldCover values, values = model class indices.
CLASS_REMAP: dict[int, int] = {
    10: 0,   # Tree cover
    20: 1,   # Shrubland
    30: 2,   # Grassland
    40: 3,   # Cropland
    50: 4,   # Built-up
    60: 5,   # Bare / sparse vegetation
    70: 6,   # Snow and ice
    80: 7,   # Permanent water bodies
    90: 8,   # Herbaceous wetland
    95: 9,   # Mangroves
    100: 10, # Moss and lichen
    0: 255,  # No-data — ignored in loss
}
NUM_CLASSES = 11
IGNORE_INDEX = 255  # passed to CrossEntropyLoss

# ── Training regions ─────────────────────────────────────────────────────────
# Four geographically and climatically diverse regions ensure the model does
# not overfit to one SAR signature style (e.g. European agricultural texture).
# BBox format: [west, south, east, north] in WGS84 degrees.
TRAINING_REGIONS: dict[str, dict] = {
    # ── Original four regions ────────────────────────────────────────────────
    "germany_bw": {
        "bbox": [8.0, 48.0, 8.5, 48.5],
        "epsg": 32632,
        "description": "Baden-Württemberg, Germany — cropland, forest, urban",
    },
    "japan_tohoku": {
        "bbox": [140.5, 38.0, 141.0, 38.5],
        "epsg": 32654,
        "description": "Tohoku, Japan — rice paddies, conifer forest, coastal",
    },
    "morocco_atlas": {
        "bbox": [-6.5, 33.0, -6.0, 33.5],
        "epsg": 32630,
        "description": "Morocco Atlas foothills — bare soil, sparse shrubland",
    },
    "brazil_parana": {
        "bbox": [-51.5, -24.5, -51.0, -24.0],
        "epsg": 32722,
        "description": "Paraná, Brazil — tropical forest, soy agriculture",
    },
    # ── Three new regions targeting underrepresented classes ─────────────────
    # Snow and ice + Moss and lichen:
    #   The Valais/Engadin area of the Swiss Alps sits above 2000 m and has
    #   permanent snowfields (Snow/ice class), high-altitude rocky areas
    #   (Bare/sparse), alpine meadows, and some moss/lichen on rock faces.
    #   Downloading January gives maximum snow extent; July gives summer state.
    "swiss_alps": {
        "bbox": [7.8, 46.3, 8.3, 46.8],
        "epsg": 32632,
        "description": "Swiss Alps (Valais) — snow/ice, bare rock, alpine grassland, moss/lichen",
    },
    # Mangroves:
    #   The Sundarbans on the India/Bangladesh coast is the world's largest
    #   mangrove forest and is densely covered by WorldCover Mangroves class.
    #   The low-salinity, shallow-canopy mangrove structure produces a
    #   distinctive dual-pol SAR signature (moderate VV, lower VH than
    #   inland forest) that is learnable from the Sentinel-1 RTC product.
    "sundarbans": {
        "bbox": [89.0, 22.0, 89.5, 22.5],
        "epsg": 32645,
        "description": "Sundarbans, Bangladesh — mangroves, water bodies",
    },
    # Permanent water bodies + Herbaceous wetland:
    #   The Finnish lake district contains thousands of large, open freshwater
    #   lakes (clean Permanent water bodies signal — specular reflection, near-zero
    #   backscatter) surrounded by boreal forest (Tree cover) and peat bogs
    #   (Herbaceous wetland).  Adding this region recovers the water-bodies IoU
    #   that degraded when Danube Delta and Sundarbans introduced ambiguous
    #   water/wetland/mangrove boundary pixels.
    "finland_lakes": {
        "bbox": [25.5, 62.0, 26.0, 62.5],
        "epsg": 32635,
        "description": "Finnish lake district — open water, boreal forest, peat bogs",
    },
}

# Months sampled for training: one per meteorological season.
# Using four seasonal composites per region × same static WorldCover label
# forces the model to be invariant to phenological SAR changes — a necessary
# property because the prototype runs on arbitrary months at inference.
TRAINING_MONTHS = list(range(1, 13))  # all 12 months — triples training data vs 4-month schedule
SAR_YEAR = 2020
SAR_RESOLUTION = 10  # metres
MAX_ITEMS_PER_QUERY = 20   # ~5–10 acquisitions is sufficient for a monthly median
CHUNK_SIZE = 512   # smaller chunks → lower peak RAM per dask task


# ── Planetary Computer connection ─────────────────────────────────────────────

def _get_client() -> pystac_client.Client:
    """Open an authenticated Planetary Computer STAC client."""
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


# ── Sentinel-1 RTC helpers ────────────────────────────────────────────────────

def _search_s1(
    client: pystac_client.Client,
    bbox: list[float],
    start: str,
    end: str,
) -> list:
    """Return signed Sentinel-1 RTC STAC items for bbox × date range."""
    search = client.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        max_items=MAX_ITEMS_PER_QUERY,
    )
    items = list(search.items())
    print(f"      {len(items):3d} S1-RTC items  ({start} → {end})")
    return items


def _median_composite(
    items: list,
    bbox: list[float],
    epsg: int,
) -> xr.DataArray:
    """
    Build a (band=2, y, x) median composite from a list of Sentinel-1 RTC items.

    Bands: [vv, vh] in linear sigma0 (not yet converted to dB — that happens
    in preprocess.py so we preserve the raw values here).

    Chunking strategy: CHUNK_SIZE × CHUNK_SIZE spatial tiles processed
    independently by dask, keeping peak RAM well within the server's 256 GB.
    """
    stack = stackstac.stack(
        items,
        assets=["vv", "vh"],
        bounds_latlon=bbox,
        epsg=epsg,
        resolution=SAR_RESOLUTION,
        dtype="float32",
        fill_value=np.float32(0.0),
        rescale=False,
        chunksize=CHUNK_SIZE,
    )
    # median across time; dask computes chunk by chunk → bounded memory
    composite = stack.median(dim="time", skipna=True)
    return composite.compute()  # triggers dask graph execution


def _save_tif(arr: xr.DataArray, path: Path, dtype: str = "float32") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.rio.to_raster(str(path), driver="GTiff", compress="lzw", dtype=dtype)


# ── ESA WorldCover helpers ────────────────────────────────────────────────────

def _download_worldcover(
    client: pystac_client.Client,
    bbox: list[float],
    epsg: int,
    output_path: Path,
) -> None:
    """
    Mosaic all WorldCover 2020 tiles covering bbox, reproject to the same
    UTM grid as the SAR composite, and save as uint8 GeoTIFF.

    Nearest-neighbour resampling preserves discrete class values.
    """
    search = client.search(
        collections=["esa-worldcover"],
        bbox=bbox,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(f"No WorldCover 2020 tiles found for bbox {bbox}")
    print(f"      {len(items):3d} WorldCover tile(s)")

    tiles = []
    for item in items:
        signed = planetary_computer.sign(item)
        href = signed.assets["map"].href
        tile = rioxarray.open_rasterio(href, masked=False).squeeze("band")
        tiles.append(tile)

    if len(tiles) > 1:
        from rioxarray.merge import merge_arrays
        mosaic = merge_arrays(tiles)
    else:
        mosaic = tiles[0]

    mosaic = mosaic.rio.clip_box(*bbox, crs="EPSG:4326")
    mosaic = mosaic.rio.reproject(
        f"EPSG:{epsg}",
        resolution=SAR_RESOLUTION,
        resampling=rasterio.enums.Resampling.nearest,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.rio.to_raster(str(output_path), driver="GTiff", compress="lzw", dtype="uint8")


# ── Per-region orchestration ──────────────────────────────────────────────────

def _median_composite_safe(items, bbox, epsg, month):
    """
    Wrapper around _median_composite that retries with progressively fewer
    items when a corrupted tile causes a RasterioIOError.

    Azure Blob Storage occasionally serves partially-written tiles (truncated
    byte count).  Since we only need ~3 acquisitions for a robust monthly
    median, we can safely drop the last item(s) and retry.
    """
    import rasterio.errors
    candidate_items = list(items)
    while len(candidate_items) >= 2:
        try:
            return _median_composite(candidate_items, bbox, epsg)
        except Exception as e:
            if "RasterioIOError" in type(e).__name__ or "Read error" in str(e) or "IReadBlock" in str(e):
                print(f"  │  [warn] Corrupted tile in item {len(candidate_items)}, "
                      f"retrying with {len(candidate_items) - 1} items…")
                candidate_items = candidate_items[:-1]
            else:
                raise
    raise RuntimeError(f"Month {month:02d}: all items failed with read errors — skipping.")


def download_region(
    name: str,
    region: dict,
    output_dir: Path,
    months: list[int] = TRAINING_MONTHS,
) -> None:
    """
    Download seasonal SAR composites and WorldCover labels for one training region.

    Files written
    ─────────────
    data/raw/<name>/sar_2020_MM.tif     — 2-band (VV, VH) float32 linear sigma0
    data/raw/<name>/worldcover_2020.tif — uint8 WorldCover class values
    """
    print(f"\n  ┌─ {name}")
    print(f"  │  {region['description']}")
    client = _get_client()
    region_dir = output_dir / name

    # ── SAR seasonal composites ──────────────────────────────────────────────
    for m in months:
        out = region_dir / f"sar_{SAR_YEAR}_{m:02d}.tif"
        if out.exists():
            print(f"  │  [skip] sar_{SAR_YEAR}_{m:02d}.tif")
            continue

        _, last_day = calendar.monthrange(SAR_YEAR, m)
        start = f"{SAR_YEAR}-{m:02d}-01"
        end   = f"{SAR_YEAR}-{m:02d}-{last_day:02d}"
        print(f"  │  Month {m:02d}: searching…")

        items = _search_s1(client, region["bbox"], start, end)
        if len(items) < 2:
            print(f"  │  [warn] Only {len(items)} acquisition(s) — skipping month {m:02d}")
            continue

        print(f"  │  Month {m:02d}: computing median composite…")
        try:
            composite = _median_composite_safe(items, region["bbox"], region["epsg"], m)
        except RuntimeError as e:
            print(f"  │  [skip] {e}")
            continue
        _save_tif(composite, out)
        print(f"  │  Saved → {out.relative_to(output_dir.parent)}")

    # ── WorldCover labels (static — download once) ───────────────────────────
    wc_out = region_dir / "worldcover_2020.tif"
    if wc_out.exists():
        print(f"  │  [skip] worldcover_2020.tif")
    else:
        print(f"  │  Fetching WorldCover labels…")
        _download_worldcover(client, region["bbox"], region["epsg"], wc_out)
        print(f"  └─ Saved → {wc_out.relative_to(output_dir.parent)}")


# ── Top-level entry point ─────────────────────────────────────────────────────

def download_all(
    output_dir: Path = Path("data/raw"),
    regions: Optional[list[str]] = None,
    months: Optional[list[int]] = None,
) -> None:
    """Download training data for all (or a subset of) regions."""
    target_regions = regions or list(TRAINING_REGIONS.keys())
    target_months  = months  or TRAINING_MONTHS

    print(f"Output root : {output_dir.resolve()}")
    print(f"Regions     : {target_regions}")
    print(f"Months      : {target_months}")
    print(f"Dask workers: {_N_WORKERS}")

    for name in target_regions:
        download_region(name, TRAINING_REGIONS[name], output_dir, target_months)

    print("\n  ✓  All downloads complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Sentinel-1 RTC + WorldCover from Planetary Computer"
    )
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument(
        "--regions", nargs="+",
        choices=list(TRAINING_REGIONS.keys()),
        default=None,
        help="Regions to fetch (default: all four)",
    )
    parser.add_argument(
        "--months", nargs="+", type=int, default=None,
        help="Month numbers 1-12 (default: 1 4 7 10)",
    )
    args = parser.parse_args()

    download_all(
        output_dir=Path(args.output_dir),
        regions=args.regions,
        months=args.months,
    )
