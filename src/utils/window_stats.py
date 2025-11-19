"""
Shared statistical helpers for window-based feature computation.
"""

from __future__ import annotations

import numpy as np

# ============================
# ROBUST STATISTICAL FUNCTIONS
# ============================
def mad(x: np.ndarray) -> float:
    """Median absolute deviation; robust to outliers."""
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

# ==========================
# MOMENT-BASED STATISTICS
# ==========================
def skewness(x: np.ndarray) -> float:
    """Sample skewness with NaN handling."""
    x = x[~np.isnan(x)]
    if x.size < 3:
        return np.nan
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))

def kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher); returns NaN if insufficient samples."""
    x = x[~np.isnan(x)]
    if x.size < 4:
        return np.nan
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    if s == 0:
        return -3.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)

# ==========================
# BASIC SAFE STATISTICS
# ==========================
def safe_stats(x: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, std, median) handling empty/all-NaN slices gracefully."""
    cleaned = x[~np.isnan(x)]
    if cleaned.size == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(cleaned))
    std = float(np.std(cleaned, ddof=1)) if cleaned.size > 1 else 0.0
    median = float(np.median(cleaned))

    return mean, std, median
