"""
Utility functions for raw signal preprocessing.

These helpers handle:
- interpolation limits based on sampling rate,
- physiological filtering (HR/SpO₂ sentinels and ranges),
- interpolation of HR/SpO₂ gaps,
- acceleration outlier removal based on a max threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

def interp_limit_from_seconds(fs_est: float, seconds: float, fallback: int = 5) -> int:
    """Convert a time threshold (s) into the number of consecutive samples to interpolate."""
    if not np.isfinite(fs_est) or fs_est <= 0:
        return fallback
    return max(1, int(round(fs_est * seconds)))

def apply_physio_filters(df: pd.DataFrame, fc_range: tuple[float, float], spo2_range: tuple[float, float]) -> None:
    """Replace sentinels and clamp out-of-range physiological values in place."""
    df["fc"].replace(999, np.nan, inplace=True)
    df["spo2"].replace(999, np.nan, inplace=True)
    df.loc[~df["fc"].between(*fc_range, inclusive="both"), "fc"] = np.nan
    df.loc[~df["spo2"].between(*spo2_range, inclusive="both"), "spo2"] = np.nan

def interpolate_channels(df: pd.DataFrame, limit: int) -> tuple[int, int]:
    """Interpolate fc and spo2 with the provided gap limit; returns counts of recovered samples."""
    fc_before = df["fc"].isna().sum()
    spo2_before = df["spo2"].isna().sum()
    df["fc"] = df["fc"].interpolate(limit=limit, limit_direction="both")
    df["spo2"] = df["spo2"].interpolate(limit=limit, limit_direction="both")
    fc_after = df["fc"].isna().sum()
    spo2_after = df["spo2"].isna().sum()
    return max(fc_before - fc_after, 0), max(spo2_before - spo2_after, 0)

def filter_acc_outliers(df: pd.DataFrame, acc_max: float) -> int:
    """Remove rows with implausible acceleration values; returns number of removed samples."""
    initial = len(df)
    mask_acc = (df[["acc_x", "acc_y", "acc_z"]].abs().max(axis=1) < acc_max)
    df.drop(index=df.index[~mask_acc], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return initial - len(df)
