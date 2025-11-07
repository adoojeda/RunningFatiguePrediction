"""
Shared utilities for kinematic feature computation within the RunningFatiguePrediction pipeline.

Pipeline overview
-----------------
The recommended processing flow for a new recording is:
    1. `src/data/preprocess.py`  → clean raw CSV files into `data/processed/clean_*.parquet`
    2. `src/features/kinematics.py` → add centred accelerations & magnitudes into `data/enriched/enriched_*.parquet`
    3. `src/data/metrics.py`    → derive Vtr, jerk, Fatigue Score and session aggregates
    4. `src/features/features_extraction.py` → compute sliding-window features joined with RPE metadata
    5. `src/analysis/*`         → run ad‑hoc visualisations and EDA on the generated datasets

This module centralises the kinematics maths so that the different stages stay consistent.

Environment flags
-----------------
- `RFP_DEFAULT_FS` (float)    → fallback sampling frequency when it cannot be inferred (default: 50 Hz)
- `RFP_HP_CUTOFF` (float)     → high-pass Butterworth cut-off frequency for acceleration filtering (default: 0.3 Hz)
- `RFP_VTR_SMOOTHING` (int)   → window size for optional Vtr smoothing in visualisations (default: 10 samples)
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.signal import butter, filtfilt

from src.config import get_config

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Configuration knobs (read from environment when available)
# ----------------------------------------------------------------------
CFG = get_config()
DEFAULT_FS: float = CFG.sampling.default_fs
DEFAULT_HP_CUTOFF: float = CFG.sampling.highpass_cutoff
DEFAULT_VTR_SMOOTHING: int = CFG.sampling.vtr_smoothing


# ----------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------
def estimate_sampling_rate(time: Sequence[float]) -> float:
    """Estimate sampling frequency (Hz) from a monotonic time vector."""
    array = pd.to_numeric(time, errors="coerce").dropna().to_numpy() if isinstance(time, pd.Series) else np.asarray(time)
    if array.size < 2:
        return np.nan
    diffs = np.diff(array)
    median_dt = np.median(diffs)
    if median_dt <= 0 or not np.isfinite(median_dt):
        return np.nan
    return 1.0 / median_dt


def centre_accelerations(df: pd.DataFrame, axes: Iterable[str] = ("X", "Y", "Z"), *, prefix: str = "Acc",
                         suffix: str = "_centered") -> pd.DataFrame:
    """
    Subtract the mean from each acceleration axis. Columns are created in place.
    """
    for axis in axes:
        col = f"{prefix}{axis}"
        centred = f"{col}{suffix}"
        if col not in df.columns:
            logger.warning("Acceleration column %s not found; skipping centring.", col)
            continue
        df[centred] = df[col] - df[col].mean()
    return df


def compute_acceleration_magnitudes(
    df: pd.DataFrame,
    axes: Iterable[str] = ("X", "Y", "Z"),
    *,
    prefix: str = "Acc",
    centred_suffix: str = "_centered",
    raw_mag_col: str = "Acc_mag",
    dyn_mag_col: str = "Acc_dyn_mag",
) -> pd.DataFrame:
    """Compute raw and centred acceleration magnitudes."""
    required_raw = [f"{prefix}{axis}" for axis in axes]
    required_centred = [f"{prefix}{axis}{centred_suffix}" for axis in axes]

    if all(col in df.columns for col in required_raw):
        df[raw_mag_col] = np.sqrt(sum(df[col] ** 2 for col in required_raw))
    else:
        missing = [col for col in required_raw if col not in df.columns]
        logger.warning("Raw acceleration columns missing (%s); Acc_mag not generated.", missing)

    if all(col in df.columns for col in required_centred):
        df[dyn_mag_col] = np.sqrt(sum(df[col] ** 2 for col in required_centred))
    else:
        missing = [col for col in required_centred if col not in df.columns]
        logger.warning("Centred acceleration columns missing (%s); Acc_dyn_mag not generated.", missing)

    return df


# ----------------------------------------------------------------------
# Advanced kinematics
# ----------------------------------------------------------------------
def highpass_filter(
    data: np.ndarray,
    fs: float,
    cutoff: float = DEFAULT_HP_CUTOFF,
    order: int = 3,
) -> np.ndarray:
    """Butterworth high-pass filter robust to NaNs and drift."""
    array = np.asarray(data, dtype=float)
    if not np.isfinite(fs) or fs <= 0 or array.size < 10:
        return array

    if np.isnan(array).any():
        idx = np.arange(array.size)
        array = np.interp(idx, idx[~np.isnan(array)], array[~np.isnan(array)])

    nyquist = 0.5 * fs
    normal_cutoff = min(cutoff / nyquist, 0.99)
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    try:
        return filtfilt(b, a, array, method="gust")
    except Exception:  
        return filtfilt(b, a, array)


def compute_translational_velocity(
    df: pd.DataFrame,
    *,
    time_col: str = "Relative_Time",
    accel_prefix: str = "Acc",
    centred_suffix: str = "_centered",
    axes: Iterable[str] = ("X", "Y", "Z"),
    velocity_prefix: str = "V",
    velocity_mag_col: str = "Vtr",
    default_fs: float = DEFAULT_FS,
    cutoff: float = DEFAULT_HP_CUTOFF,
) -> pd.DataFrame:
    """Integrate centred accelerations to obtain translational velocity per axis and magnitude."""
    required = [f"{accel_prefix}{axis}{centred_suffix}" for axis in axes]
    if time_col not in df.columns or any(col not in df.columns for col in required):
        missing = [col for col in [time_col, *required] if col not in df.columns]
        logger.warning("Missing columns for velocity computation: %s", missing)
        return df

    t = df[time_col].to_numpy(dtype=float)
    if t.size < 2:
        logger.warning("Not enough samples to integrate velocity.")
        return df

    fs = estimate_sampling_rate(t) or default_fs

    filtered = {}
    for axis in axes:
        col = f"{accel_prefix}{axis}{centred_suffix}"
        filt = highpass_filter(df[col].to_numpy(dtype=float), fs, cutoff=cutoff)
        filtered[axis] = filt - np.mean(filt)

    velocities = {}
    for axis in axes:
        velocities[axis] = integrate.cumtrapz(filtered[axis], x=t, initial=0.0)
        velocities[axis] -= np.mean(velocities[axis])
        df[f"{velocity_prefix}{axis}"] = velocities[axis]

    mag = np.sqrt(sum(velocities[axis] ** 2 for axis in axes))
    df[velocity_mag_col] = mag
    return df


def compute_jerk(
    df: pd.DataFrame,
    *,
    time_col: str = "Relative_Time",
    accel_prefix: str = "Acc",
    centred_suffix: str = "_centered",
    axes: Iterable[str] = ("X", "Y", "Z"),
    jerk_prefix: str = "jerk",
    jerk_mag_col: str = "jerk_mag",
) -> pd.DataFrame:
    """Differentiate centred accelerations to obtain jerk components and magnitude."""
    if time_col not in df.columns:
        logger.warning("Missing %s column for jerk computation.", time_col)
        return df

    t = df[time_col].to_numpy(dtype=float)
    if t.size < 2:
        logger.warning("Not enough samples to compute jerk.")
        return df

    if np.any(~np.isfinite(t)):
        logger.warning("Time vector contains non-finite values; jerk set to NaN.")
        for axis in axes:
            df[f"{jerk_prefix}{axis}"] = np.nan
        df[jerk_mag_col] = np.nan
        return df

    diffs = np.diff(t)
    if np.any(diffs <= 0):
        logger.warning("Time vector is not strictly increasing; jerk set to NaN.")
        for axis in axes:
            df[f"{jerk_prefix}{axis}"] = np.nan
        df[jerk_mag_col] = np.nan
        return df

    jerks = {}
    for axis in axes:
        col = f"{accel_prefix}{axis}{centred_suffix}"
        if col not in df.columns:
            logger.warning("Skipping jerk computation for %s (column missing).", col)
            df[f"{jerk_prefix}{axis}"] = np.nan
            continue
        values = df[col].to_numpy(dtype=float)
        if np.isnan(values).all():
            logger.warning("All samples are NaN for %s; jerk set to NaN.", col)
            df[f"{jerk_prefix}{axis}"] = np.nan
            continue

        if np.isnan(values).any():
            idx = np.arange(values.size)
            valid = ~np.isnan(values)
            values = np.interp(idx, idx[valid], values[valid])

        edge_order = 2 if values.size >= 3 else 1
        jerks[axis] = np.gradient(values, t, edge_order=edge_order)
        df[f"{jerk_prefix}{axis}"] = jerks[axis]

    valid_axes = [axis for axis in axes if axis in jerks]
    if valid_axes:
        df[jerk_mag_col] = np.sqrt(sum(jerks[axis] ** 2 for axis in valid_axes))
    else:
        df[jerk_mag_col] = np.nan
    return df


__all__ = [
    "DEFAULT_FS",
    "DEFAULT_HP_CUTOFF",
    "DEFAULT_VTR_SMOOTHING",
    "centre_accelerations",
    "compute_acceleration_magnitudes",
    "compute_translational_velocity",
    "compute_jerk",
    "estimate_sampling_rate",
    "highpass_filter",
]
