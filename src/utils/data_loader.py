"""
Utility module to simplify data loading across the running fatigue pipeline.

Main capabilities:
- Load processed or enriched parquet/CSV files and ensure temporal consistency.
- Derive a `second` column from `relative_time` for coarse aggregations.
- Average numeric signals per second (`average_per_second`).
- Load the unified feature dataset stored under data/results/.
- Concatenate multiple sessions for exploratory analysis or validation workflows.

This module is not part of the core pipeline but is handy for:
    - interactive dashboards
    - manual inspections
    - pipeline integrity checks
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd

# ===========================
# LOGGING SETUP
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ===========================
# PATH CONFIGURATION
# ===========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
DEFAULT_FEATURES_PATH = os.path.join(RESULTS_DIR, "features_dataset.parquet")

def _candidate_paths(name: str, directory: str, prefixes: Optional[List[str]] = None) -> List[str]:
    """Return possible file paths inside *directory* derived from *name* and optional prefixes."""
    if os.path.isabs(name):
        return [name]

    filename = name if name.endswith(".parquet") else f"{name}.parquet"
    candidates = [os.path.join(directory, filename)]
    for prefix in prefixes or []:
        prefixed = filename if filename.startswith(prefix) else f"{prefix}{filename}"
        candidates.append(os.path.join(directory, prefixed))

    seen = set()
    unique_candidates = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique_candidates.append(path)
    return unique_candidates

# ===========================
# CORE LOADERS
# ===========================
@lru_cache(maxsize=64)
def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load a processed or enriched file (parquet/csv) and ensure temporal consistency.

    Returns
    -------
    Optional[pandas.DataFrame]
        DataFrame containing at least:
            * `relative_time` (float seconds) or `time`
            * inertial sensors (`acc_*`, `grav_*`, `rot_*`, `roll`, `pitch`, `yaw`)
            * physiological channels (`fc`, `spo2`)
            * helper column `second`
        Returns None if the file cannot be read or is empty.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".parquet", ".parq"):
            df = pd.read_parquet(file_path)
        elif ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        if df.empty:
            raise ValueError(f"The file {os.path.basename(file_path)} is empty.")

        time_col = "relative_time" if "relative_time" in df.columns else "time"
        if time_col not in df.columns:
            raise KeyError("No time column found ('relative_time' or 'time').")

        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df.dropna(subset=[time_col], inplace=True)
        df["second"] = np.floor(df[time_col]).astype(int)

        logger.info("Loaded %s (%d rows, %d columns)", os.path.basename(file_path), len(df), len(df.columns))
        return df

    except Exception as exc:
        logger.error("Error loading %s: %s", file_path, exc, exc_info=True)
        return None

@lru_cache(maxsize=64)
def load_enriched_session(name: str, *, fallback_to_processed: bool = True) -> Optional[pd.DataFrame]:
    """
    Load a single session from `data/enriched/`.

    Parameters
    ----------
    name:
        File name (with or without `.parquet` / `enriched_` prefix) or absolute path.
    fallback_to_processed:
        When True, attempt to load the corresponding `clean_*.parquet` from `data/processed/`
        if the enriched file is missing.

    Returns
    -------
    Optional[pandas.DataFrame]
        DataFrame with the same column guarantees as `load_data`. Returns None if the
        session cannot be located.
    """

    for candidate in _candidate_paths(name, ENRICHED_DIR, ["enriched_"]):
        if os.path.exists(candidate):
            return load_data(candidate)

    if not fallback_to_processed:
        logger.warning("Enriched session %s not found in %s", name, ENRICHED_DIR)
        return None

    processed_base = os.path.splitext(os.path.basename(name))[0]
    if processed_base.startswith("enriched_"):
        processed_base = processed_base[len("enriched_"):]

    for candidate in _candidate_paths(processed_base, PROCESSED_DIR, ["clean_"]):
        if os.path.exists(candidate):
            logger.warning(
                "Enriched file %s missing; falling back to processed version %s",
                name,
                os.path.basename(candidate),
            )
            return load_data(candidate)

    logger.error("Session %s not found in enriched or processed directories.", name)
    return None

def list_enriched_sessions() -> List[str]:
    """Return the list of available enriched parquet files (sorted alphabetically)."""
    if not os.path.isdir(ENRICHED_DIR):
        return []
    return sorted(
        f for f in os.listdir(ENRICHED_DIR)
        if f.endswith(".parquet") and f.startswith("enriched_")
    )

def average_per_second(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Average numeric columns per second.

    Useful for visualisations or reducing temporal resolution. The returned DataFrame contains
    `Second` plus the averaged numeric columns.
    """
    try:
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or was not loaded correctly.")

        if "Second" not in df.columns:
            raise KeyError("Column 'Second' not found in DataFrame.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if columns is not None:
            numeric_cols = [col for col in numeric_cols if col in columns]
        numeric_cols = [col for col in numeric_cols if col != "Second"]

        if not numeric_cols:
            raise ValueError("No numeric columns available to average.")

        df_avg = (
            df.groupby("Second")[numeric_cols]
            .mean()
            .reset_index()
            .sort_values(by="Second")
            .reset_index(drop=True)
        )

        logger.info("Averaged %d rows (1 per second).", len(df_avg))
        return df_avg

    except Exception as exc:
        logger.error("Error averaging data per second: %s", exc, exc_info=True)
        return None

#============================
# BATCH LOADERS
# ===========================
@lru_cache(maxsize=8)
def load_all_sessions(limit: Optional[int] = None, prefer_enriched: bool = True) -> Optional[pd.DataFrame]:
    """
    Load and concatenate parquet files from data/enriched or data/processed.

    Parameters
    ----------
    limit:
        Optionally restrict the number of files loaded (useful for quick tests).
    prefer_enriched:
        When True, prioritise loading from data/enriched/. Falls back to data/processed/.
    """
    directories: List[str] = []
    if prefer_enriched and os.path.isdir(ENRICHED_DIR):
        directories.append(ENRICHED_DIR)
    if os.path.isdir(PROCESSED_DIR):
        directories.append(PROCESSED_DIR)

    if not directories:
        logger.warning("Neither data/enriched nor data/processed directories exist.")
        return None

    for directory in directories:
        files = [
            f for f in os.listdir(directory)
            if f.endswith(".parquet") and (f.startswith("enriched_") or f.startswith("clean_"))
        ]
        if not files:
            continue

        if limit:
            files = files[:limit]

        dfs = []
        for fname in files:
            path = os.path.join(directory, fname)
            df = load_data(path)
            if df is not None and not df.empty:
                df["file_id"] = os.path.splitext(fname)[0]
                dfs.append(df)

        if dfs:
            df_all = pd.concat(dfs, ignore_index=True)
            logger.info("📁 Loaded and concatenated %d files (%d total rows) from %s", len(dfs), len(df_all), directory)
            return df_all

    logger.warning("No parquet files were loaded successfully from the available directories.")
    return None

@lru_cache(maxsize=8)
def load_features_dataset(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load the unified feature dataset for downstream modelling/analysis.
    Defaults to data/results/features_dataset_3s_50olap.parquet.
    """
    try:
        dataset_path = path or DEFAULT_FEATURES_PATH
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Feature dataset not found at: {dataset_path}")

        df = pd.read_parquet(dataset_path)
        logger.info("Feature dataset loaded (%d windows, %d columns).", len(df), len(df.columns))
        return df

    except Exception as exc:
        logger.error("Error loading feature dataset: %s", exc, exc_info=True)
        return None

# ===========================
# LOCAL TEST
# ===========================
if __name__ == "__main__":
    print("Quick load test:")
    sessions = list_enriched_sessions()
    if sessions:
        print("First enriched session:", sessions[0])
        df_session = load_enriched_session(sessions[0])
        if df_session is not None:
            print(df_session.head(3))

    df_all = load_all_sessions(limit=1)
    if df_all is not None:
        print(df_all.head(3))
        df_avg = average_per_second(df_all)
        if df_avg is not None:
            print("\nAverage per second:")
            print(df_avg.head(3))

    df_features = load_features_dataset()
    if df_features is not None:
        print("\nFeature dataset:")
        print(df_features.head(3))

__all__ = [
    "load_data",
    "load_enriched_session",
    "list_enriched_sessions",
    "average_per_second",
    "load_all_sessions",
    "load_features_dataset",
]
