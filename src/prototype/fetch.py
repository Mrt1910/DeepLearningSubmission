"""
src/prototype/fetch.py

Given a lat/lon centre and a start month, download 12 consecutive monthly
Sentinel-1 RTC median composites from Planetary Computer.

The bounding box is derived from the target coordinate by extending ±0.15°
(~16 km) in each direction, which gives a ~32 km × 32 km region — large
enough to see landscape-scale LULC patterns around a city centre, small
enough to compute in a few minutes on the server.

Outputs: <output_dir>/sar_YYYY_MM.tif  (one per month, 2-band float32 linear)
"""

from __future__ import annotations

import calendar
from pathlib import Path

import planetary_computer
import pystac_client
import stackstac

from src.data.download import (
    CHUNK_SIZE,
    SAR_RESOLUTION,
    _get_client,
    _median_composite,
    _save_tif,
)

# ~16 km radius around the target point
BOX_HALF_DEG = 0.15


def latlon_to_bbox(lat: float, lon: float) -> list[float]:
    return [lon - BOX_HALF_DEG, lat - BOX_HALF_DEG,
            lon + BOX_HALF_DEG, lat + BOX_HALF_DEG]


def _epsg_from_lon(lon: float) -> int:
    """Rough UTM zone from longitude (WGS84, northern hemisphere for lat>=0)."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone  # northern; adjust below for southern


def epsg_for_latlon(lat: float, lon: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def fetch_12_months(
    lat: float,
    lon: float,
    start_year: int,
    start_month: int,
    output_dir: Path,
) -> list[Path]:
    """
    Download 12 monthly composites starting at (start_year, start_month).

    Returns a list of 12 Path objects (some may be None if a month had
    fewer than 2 acquisitions and was skipped with a warning).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox = latlon_to_bbox(lat, lon)
    epsg = epsg_for_latlon(lat, lon)
    client = _get_client()
    paths: list[Path | None] = []

    for i in range(12):
        month = (start_month - 1 + i) % 12 + 1
        year  = start_year + (start_month - 1 + i) // 12

        out = output_dir / f"sar_{year}_{month:02d}.tif"
        if out.exists():
            print(f"  [skip] {out.name}")
            paths.append(out)
            continue

        _, last_day = calendar.monthrange(year, month)
        start_str = f"{year}-{month:02d}-01"
        end_str   = f"{year}-{month:02d}-{last_day:02d}"
        print(f"  Month {year}-{month:02d}: searching…")

        search = client.search(
            collections=["sentinel-1-rtc"],
            bbox=bbox,
            datetime=f"{start_str}/{end_str}",
            max_items=150,
        )
        items = list(search.items())
        print(f"    {len(items)} items found")

        if len(items) < 2:
            print(f"  [warn] Fewer than 2 acquisitions — skipping {year}-{month:02d}")
            paths.append(None)
            continue

        composite = _median_composite(items, bbox, epsg)
        _save_tif(composite, out)
        print(f"  Saved {out.name}")
        paths.append(out)

    return paths
