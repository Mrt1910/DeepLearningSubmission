"""
src/prototype/stats.py

Compute per-class area percentages and month-to-month change statistics
from a sequence of 12 LULC prediction maps.
"""

from __future__ import annotations

import numpy as np

from src.data.download import IGNORE_INDEX, NUM_CLASSES, WORLDCOVER_CLASSES

CLASS_NAMES = [WORLDCOVER_CLASSES[k] for k in sorted(WORLDCOVER_CLASSES)]


def area_percentages(pred: np.ndarray, num_classes: int = NUM_CLASSES) -> dict[str, float]:
    """
    Return {class_name: percentage} for valid (non-ignore) pixels.
    """
    valid = pred != IGNORE_INDEX
    total = valid.sum()
    if total == 0:
        return {name: 0.0 for name in CLASS_NAMES}
    result = {}
    for c, name in enumerate(CLASS_NAMES):
        count = ((pred == c) & valid).sum()
        result[name] = round(100.0 * count / total, 2)
    return result


def monthly_stats(preds: list[np.ndarray]) -> dict:
    """
    Given 12 monthly prediction maps (H×W int16), compute:

    monthly_pct   — list of 12 {class: %} dicts
    month_changes — list of 11 {class: Δ%} dicts (month i+1 minus month i)
    total_change  — {class: Δ%} (last month minus first month)
    """
    monthly_pct   = [area_percentages(p) for p in preds]

    month_changes = []
    for i in range(len(preds) - 1):
        delta = {
            name: round(monthly_pct[i + 1][name] - monthly_pct[i][name], 2)
            for name in CLASS_NAMES
        }
        month_changes.append(delta)

    total_change = {
        name: round(monthly_pct[-1][name] - monthly_pct[0][name], 2)
        for name in CLASS_NAMES
    }

    return {
        "monthly_pct":   monthly_pct,
        "month_changes": month_changes,
        "total_change":  total_change,
    }
