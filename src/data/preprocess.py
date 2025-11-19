"""
Running signal data preprocessing (pipeline stage 1/5):
- Clean raw CSV recordings into analysis-ready parquet files.
- Filter physical/physiological outliers and interpolate short gaps in HR/SpO₂.
- Keep gravity/rotation/orientation signals for downstream biomechanical models.
- Drop only the absolute timestamp column; create `Relative_Time` instead.

Output: `data/processed/clean_*.parquet`
Next stage: `python src/features/kinematics.py`
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import get_config
from src.utils.kinematics import DEFAULT_FS, estimate_sampling_rate
from src.utils.preprocess_utils import (
    apply_physio_filters,
    filter_acc_outliers,
    interpolate_channels,
    interp_limit_from_seconds,
)
from src.utils.schemas import validate_dataframe

# ===========================
# LOGGING
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===========================
# PATHS & CONFIG
# ===========================
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

CFG = get_config()
PHYSIO_RANGES = CFG.ranges
INTERP_MAX_GAP_SEC = CFG.interpolation.max_gap_seconds
MAX_WORKERS = CFG.workforce.max_workers

# ===========================
# CUSTOM ERRORS / DATA CLASSES
# ===========================
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

# ===========================
# HELPER FUNCTIONS
# ===========================
def _load_raw_file(filepath: str) -> pd.DataFrame:
    """Read raw CSV data and enforce the expected column layout."""
    df = pd.read_csv(filepath, header=None)
    if df.empty:
        raise EmptyFileError("The file is empty.")
    if df.shape[1] < 15:
        raise ColumnCountError(f"Expected 15 columns, detected {df.shape[1]}.")

    df.columns = [
        "time", "acc_x", "acc_y", "acc_z",
        "grav_x", "grav_y", "grav_z",
        "rot_x", "rot_y", "rot_z",
        "roll", "pitch", "yaw",
        "fc", "spo2",
    ]
    return df

def _ensure_numeric(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Cast specified columns to numeric dtype in place."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def _derive_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    """Create relative_time from absolute timestamps."""
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    if df.empty:
        raise PreprocessError("All timestamp values are NaN.")
    df["relative_time"] = df["time"] - df["time"].iloc[0]
    df.drop(columns=["time"], inplace=True, errors="ignore")
    return df

def _finalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column ordering and validate the processed schema."""
    ordered_columns = ["relative_time"] + [col for col in df.columns if col != "relative_time"]
    df = df[ordered_columns]
    validate_dataframe(df, "processed")
    return df

# ===========================
# MAIN PREPROCESSING
# ===========================
def preprocess_single_file(filepath: str) -> Optional[pd.DataFrame]:
    """
    Preprocess a single CSV file into an analysis-ready dataframe.

    Parameters
    ----------
    filepath:
        Path to the raw CSV file.

    Returns
    -------
    Optional[pandas.DataFrame]
        DataFrame with normalised columns and `relative_time`, or None on failure.
    """
    stats = PreprocessStats()
    try:
        df = _load_raw_file(filepath)
        stats.samples_in = len(df)
        _ensure_numeric(df, df.columns)
        validate_dataframe(df, "raw")
        df = _derive_relative_time(df)

        fs_est = estimate_sampling_rate(df["relative_time"]) or DEFAULT_FS
        interp_limit = interp_limit_from_seconds(fs_est, INTERP_MAX_GAP_SEC)

        apply_physio_filters(df, PHYSIO_RANGES.fc, PHYSIO_RANGES.spo2)
        stats.interpolated_fc, stats.interpolated_spo2 = interpolate_channels(df, interp_limit)
        stats.acc_outliers_removed = filter_acc_outliers(df, PHYSIO_RANGES.acc_max)
        df = _finalise_dataframe(df)
        stats.samples_out = len(df)

        logger.info(
            "Preprocessed %s | fs=%.2f Hz | stats=%s",
            os.path.basename(filepath),
            fs_est,
            stats.as_dict(),
        )
        return df

    except PreprocessError as exc:
        logger.error("Preprocessing error for %s: %s", os.path.basename(filepath), exc)
        return None
    except Exception as exc:  
        logger.error(
            "Unexpected error while preprocessing %s: %s",
            os.path.basename(filepath),
            exc,
            exc_info=True,
        )
        return None

def process_file(filepath: str) -> Optional[str]:
    """Process a complete file and save it as Parquet."""
    try:
        df = preprocess_single_file(filepath)
        if df is None or df.empty:
            logger.warning("File %s was not processed successfully.", os.path.basename(filepath))
            return None

        filename = os.path.basename(filepath).replace(".csv", ".parquet")
        output_path = os.path.join(PROCESSED_DIR, f"clean_{filename}")
        df.to_parquet(output_path, index=False)
        logger.info("File processed and saved: %s", output_path)
        return output_path
    except Exception as exc:
        logger.error("Error processing file %s: %s", filepath, exc, exc_info=True)
        return None

def preprocess_data(parallel: bool = True) -> Optional[List[str]]:
    """Preprocess all CSV files located in data/raw/."""
    if not os.path.isdir(RAW_DIR):
        logger.error("Raw data directory not found: %s", RAW_DIR)
        return None

    csv_files = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if not csv_files:
        logger.warning("No CSV files were found in data/raw/.")
        return None

    logger.info("Files detected: %d", len(csv_files))

    processed_files: List[str] = []
    failed_files: List[str] = []

    def _record_result(source: str, result: Optional[str]) -> None:
        (processed_files if result else failed_files).append(result or source)

    if parallel and len(csv_files) > 1:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(process_file, csv_files))
        for fpath, res in zip(csv_files, results):
            _record_result(fpath, res)
    else:
        for f in csv_files:
            _record_result(f, process_file(f))

    metadata = {
        "processed_files": [p for p in processed_files if p],
        "failed_files": failed_files,
        "total_processed": len(processed_files),
        "total_failed": len(failed_files),
        "date": str(pd.Timestamp.now()),
    }

    metadata_path = os.path.join(PROCESSED_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info("Metadata saved to: %s", metadata_path)
    return processed_files

# ===========================
# MAIN EXECUTION
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized running signal preprocessing.")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel execution.")
    args = parser.parse_args()

    processed = preprocess_data(parallel=not args.no_parallel)
    if processed:
        logger.info("Preprocessing completed successfully.")
    else:
        logger.warning("No files were processed.")
