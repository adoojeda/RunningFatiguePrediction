"""
Utility functions for raw signal preprocessing.

These helpers handle:
- CSV loading and schema enforcement,
- interpolation limits based on sampling rate,
- physiological filtering (HR/SpO₂ sentinels and ranges),
- interpolation of HR/SpO₂ gaps,
- acceleration outlier removal based on a max threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.utils.schemas import validate_dataframe

# ==========================
# ERRORS AND DATA CLASSES
# ==========================
class PreprocessError(Exception):
    """Base error for preprocessing issues."""

class EmptyFileError(PreprocessError):
    """Raised when a CSV file has no data."""

class ColumnCountError(PreprocessError):
    """Raised when the incoming CSV does not contain the expected channels."""

@dataclass
class PreprocessStats:
    """Summary of per-file operations for richer logging."""

    samples_in: int = 0
    samples_out: int = 0
    interpolated_fc: int = 0
    interpolated_spo2: int = 0
    acc_outliers_removed: int = 0

    def as_dict(self) -> dict:
        return {
            "samples_in": self.samples_in,
            "samples_out": self.samples_out,
            "interpolated_fc": self.interpolated_fc,
            "interpolated_spo2": self.interpolated_spo2,
            "acc_outliers_removed": self.acc_outliers_removed,
        }

# ==========================
# RAW FILE HANDLING
# ==========================
def load_raw_file(filepath: str, expected_columns: int = 15) -> pd.DataFrame:
    """Read raw CSV data and enforce the expected column layout."""
    df = pd.read_csv(filepath, header=None)
    if df.empty:
        raise EmptyFileError("The file is empty.")
    if df.shape[1] < expected_columns:
        raise ColumnCountError(f"Expected {expected_columns} columns, detected {df.shape[1]}.")

    df.columns = [
        "time",
        "acc_x",
        "acc_y",
        "acc_z",
        "grav_x",
        "grav_y",
        "grav_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "roll",
        "pitch",
        "yaw",
        "fc",
        "spo2",
    ]
    return df

def ensure_numeric(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> None:
    """Cast specified columns (or all) to numeric dtype in place."""
    if columns is None:
        columns = df.columns
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def derive_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    """Create relative_time from absolute timestamps."""
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    if df.empty:
        raise PreprocessError("All timestamp values are NaN.")
    df["relative_time"] = df["time"] - df["time"].iloc[0]
    df.drop(columns=["time"], inplace=True, errors="ignore")
    return df

def finalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column ordering and validate the processed schema."""
    ordered_columns = ["relative_time"] + [col for col in df.columns if col != "relative_time"]
    df = df[ordered_columns]
    validate_dataframe(df, "processed")
    return df

# ==========================
# INTERPOLATION UTILITIES
# ==========================
def interp_limit_from_seconds(fs_est: float, seconds: float, fallback: int = 5) -> int:
    """Convert a time threshold (s) into the number of consecutive samples to interpolate."""
    if not np.isfinite(fs_est) or fs_est <= 0:
        return fallback
    return max(1, int(round(fs_est * seconds)))

# ================================
# PHYSIOLOGICAL FILTERING UTILITIES
# ================================
def apply_physio_filters(
    df: pd.DataFrame,
    fc_range: tuple[float, float],
    spo2_range: tuple[float, float],
) -> None:
    """Replace sentinels and clamp out-of-range physiological values in place."""
    df["fc"].replace(999, np.nan, inplace=True)
    df["spo2"].replace(999, np.nan, inplace=True)
    df.loc[~df["fc"].between(*fc_range, inclusive="both"), "fc"] = np.nan
    df.loc[~df["spo2"].between(*spo2_range, inclusive="both"), "spo2"] = np.nan

# ==================================
# CHANNEL INTERPOLATION AND RECOVERY
# ==================================
def interpolate_channels(df: pd.DataFrame, limit: int) -> tuple[int, int]:
    """Interpolate fc and spo2 with the provided gap limit; returns counts of recovered samples."""
    fc_before = df["fc"].isna().sum()
    spo2_before = df["spo2"].isna().sum()

    df["fc"] = df["fc"].interpolate(limit=limit, limit_direction="both")
    df["spo2"] = df["spo2"].interpolate(limit=limit, limit_direction="both")

    fc_after = df["fc"].isna().sum()
    spo2_after = df["spo2"].isna().sum()

    return max(fc_before - fc_after, 0), max(spo2_before - spo2_after, 0)

# ===========================
# ACCELERATION OUTLIER FILTER
# ===========================
def filter_acc_outliers(df: pd.DataFrame, acc_max: float) -> int:
    """Remove rows with implausible acceleration values; returns number of removed samples."""
    initial = len(df)
    mask_acc = df[["acc_x", "acc_y", "acc_z"]].abs().max(axis=1) < acc_max

    df.drop(index=df.index[~mask_acc], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return initial - len(df)
