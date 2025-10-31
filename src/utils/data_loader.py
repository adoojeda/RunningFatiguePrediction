"""
Utility module to simplify data loading across the running fatigue pipeline.

Main capabilities:
- Load processed or enriched parquet/CSV files and ensure temporal consistency.
- Derive a `Second` column from `Relative_Time` for coarse aggregations.
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
from typing import List, Optional

import numpy as np
import pandas as pd

# ======================================================================
# LOGGING CONFIGURATION
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================================================
# PATH CONFIGURATION
# ======================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
DEFAULT_FEATURES_PATH = os.path.join(RESULTS_DIR, "features_dataset_5s_50olap.parquet")

# ======================================================================
# CORE LOADERS
# ======================================================================
def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load a processed or enriched file (parquet/csv) and ensure temporal consistency.

    Returns a DataFrame with a `Second` column derived from `Relative_Time`.
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

        time_col = "Relative_Time" if "Relative_Time" in df.columns else "Time"
        if time_col not in df.columns:
            raise KeyError("No time column found ('Relative_Time' or 'Time').")

        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df.dropna(subset=[time_col], inplace=True)
        df["Second"] = np.floor(df[time_col]).astype(int)

        logger.info("✅ Loaded %s (%d rows, %d columns)", os.path.basename(file_path), len(df), len(df.columns))
        return df

    except Exception as exc:
        logger.error("❌ Error loading %s: %s", file_path, exc, exc_info=True)
        return None

def average_per_second(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Average numeric columns per second.

    Useful for visualisations or reducing temporal resolution.
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

        logger.info("📊 Averaged %d rows (1 per second).", len(df_avg))
        return df_avg

    except Exception as exc:
        logger.error("❌ Error averaging data per second: %s", exc, exc_info=True)
        return None

#=======================================================================
# BATCH LOADERS
# ======================================================================
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
        logger.warning("⚠️ Neither data/enriched nor data/processed directories exist.")
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

    logger.warning("⚠️ No parquet files were loaded successfully from the available directories.")
    return None


def load_features_dataset(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load the unified feature dataset for downstream modelling/analysis.
    Defaults to data/results/features_dataset_5s_50olap.parquet.
    """
    try:
        dataset_path = path or DEFAULT_FEATURES_PATH
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Feature dataset not found at: {dataset_path}")

        df = pd.read_parquet(dataset_path)
        logger.info("✅ Feature dataset loaded (%d windows, %d columns).", len(df), len(df.columns))
        return df

    except Exception as exc:
        logger.error("❌ Error loading feature dataset: %s", exc, exc_info=True)
        return None

# ======================================================================
# LOCAL TEST
# ======================================================================
if __name__ == "__main__":
    print("🔍 Quick load test:")
    df_all = load_all_sessions(limit=1)
    if df_all is not None:
        print(df_all.head(3))
        df_avg = average_per_second(df_all)
        if df_avg is not None:
            print("\nAverage per second:")
            print(df_avg.head(3))

    df_features = load_features_dataset()
    if df_features is not None:
        print("\n✅ Feature dataset:")
        print(df_features.head(3))
