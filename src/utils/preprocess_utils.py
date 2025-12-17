"""
Helper utilities for the preprocessing stage.

They cover:
- loading CSV files and enforcing the expected schema,
- converting time limits into interpolation limits,
- physiological filtering of HR and SpO₂,
- interpolation of short gaps,
- removal of implausible acceleration samples.
"""

# STANDARD LIBRARIES
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# PROJECT IMPORTS
from src.utils.schemas import validate_dataframe

# ERROR CLASSES
class PreprocessError(Exception):
    """Base error for preprocessing incidents."""

class EmptyFileError(PreprocessError):
    """Raised when a CSV contains no data."""

class ColumnCountError(PreprocessError):
    """Raised when the CSV does not contain the expected number of columns."""

# STATS DATACLASS
@dataclass
class PreprocessStats:
    """Per-file summary to enrich logs."""

    samples_in: int = 0
    samples_out: int = 0
    interpolated_hr: int = 0
    interpolated_spo2: int = 0
    acc_outliers_removed: int = 0

    def as_dict(self) -> dict:
        return {
            "samples_in": self.samples_in,
            "samples_out": self.samples_out,
            "interpolated_hr": self.interpolated_hr,
            "interpolated_spo2": self.interpolated_spo2,
            "acc_outliers_removed": self.acc_outliers_removed,
        }
    
# RAW FILE HANDLING
def load_raw_file(filepath: str, expected_columns: int = 15) -> pd.DataFrame:
    """Reads a raw CSV and enforces the expected layout."""
    df = pd.read_csv(filepath, header=None)
    if df.empty:
        raise EmptyFileError("The file is empty.")
    if df.shape[1] < expected_columns:
        raise ColumnCountError(f"Expected {expected_columns} columns, found {df.shape[1]}.")

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
        "hr",
        "spo2",
    ]
    return df

# DATAFRAME PREPROCESSING STEPS
def ensure_numeric(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> None:
    """Converts the indicated columns (or all) to numeric in place."""
    if columns is None:
        columns = df.columns
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def derive_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    """Creates `relative_time` from absolute timestamps."""
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    if df.empty:
        raise PreprocessError("All timestamp values are NaN.")
    df["relative_time"] = df["time"] - df["time"].iloc[0]
    df.drop(columns=["time"], inplace=True, errors="ignore")
    return df

def finalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Enforces the final column order and validates the processed schema."""
    ordered_columns = ["relative_time"] + [col for col in df.columns if col != "relative_time"]
    df = df[ordered_columns]
    validate_dataframe(df, "processed")
    return df

# INTERPOLATION UTILITIES
def interp_limit_from_seconds(fs_est: float, seconds: float, fallback: int = 5) -> int:
    """Converts a time span into number of consecutive samples to interpolate."""
    if not np.isfinite(fs_est) or fs_est <= 0:
        return fallback
    return max(1, int(round(fs_est * seconds)))

# PHYSIOLOGICAL FILTERING
def apply_physio_filters(
    df: pd.DataFrame,
    hr_range: tuple[float, float],
    spo2_range: tuple[float, float],
) -> None:
    """Replaces sentinels and nulls out-of-range physiological values."""
    df["hr"] = df["hr"].replace(999, np.nan)
    df["spo2"] = df["spo2"].replace(999, np.nan)
    df.loc[~df["hr"].between(*hr_range, inclusive="both"), "hr"] = np.nan
    df.loc[~df["spo2"].between(*spo2_range, inclusive="both"), "spo2"] = np.nan

# CHANNEL INTERPOLATION
def interpolate_channels(df: pd.DataFrame, limit: int) -> tuple[int, int]:
    """Interpolates HR and SpO₂ and returns the number of recovered samples per channel."""
    hr_before = df["hr"].isna().sum()
    spo2_before = df["spo2"].isna().sum()

    df["hr"] = df["hr"].interpolate(limit=limit, limit_direction="both")
    df["spo2"] = df["spo2"].interpolate(limit=limit, limit_direction="both")

    hr_after = df["hr"].isna().sum()
    spo2_after = df["spo2"].isna().sum()

    return max(hr_before - hr_after, 0), max(spo2_before - spo2_after, 0)

# ACCELERATION OUTLIER FILTER
def filter_acc_outliers(df: pd.DataFrame, acc_max: float) -> int:
    """Drops rows containing implausible accelerations and returns how many were removed."""
    initial = len(df)
    mask_acc = df[["acc_x", "acc_y", "acc_z"]].abs().max(axis=1) < acc_max

    df.drop(index=df.index[~mask_acc], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return initial - len(df)
