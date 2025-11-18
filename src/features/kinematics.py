"""
Kinematic feature generation (pipeline stage 2/5):
- Centre accelerations per axis and compute raw/dynamic magnitudes.
- Prepare enriched parquet files consumed by metrics and feature extraction stages.

Input: `data/processed/clean_*.parquet`
Output: `data/enriched/enriched_*.parquet`
Next stage: `python src/data/metrics.py`
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from glob import glob
from typing import Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.kinematics import centre_accelerations, compute_acceleration_magnitudes
from src.utils.schemas import validate_dataframe

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
BASE_DIR = PROJECT_ROOT
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
ENRICHED_DIR = os.path.join(BASE_DIR, "data", "enriched")

# ======================================================================
# CUSTOM TYPES
# ======================================================================
class KinematicsError(Exception):
    """Raised when a kinematic transformation fails."""

@dataclass
class KinematicsStats:
    """Simple structure capturing relevant processing metadata."""

    file: str
    columns_before: int
    columns_after: int
    output_path: str

# ======================================================================
# CORE KINEMATIC FEATURES
# ======================================================================
def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure centred accelerations and magnitude features exist in the frame.
    """
    centre_accelerations(df)
    compute_acceleration_magnitudes(df)
    return df

def _load_processed_session(path: str) -> pd.DataFrame:
    """Load a processed parquet file and validate its schema."""
    df = pd.read_parquet(path)
    validate_dataframe(df, "processed")
    return df

def _write_enriched_session(df: pd.DataFrame, source_path: str, overwrite: bool) -> str:
    """Persist the dataframe to the enriched directory (or overwrite the source)."""
    validate_dataframe(df, "enriched")
    if overwrite:
        output_path = source_path
    else:
        base = os.path.basename(source_path).replace("clean_", "enriched_")
        output_path = os.path.join(ENRICHED_DIR, base)
    df.to_parquet(output_path, index=False)
    return output_path

def process_single_file(path: str, *, overwrite: bool = False) -> Optional[KinematicsStats]:
    """
    Load → enrich → save a single session. Returns stats or None on failure.
    """
    try:
        df = _load_processed_session(path)
        before_cols = len(df.columns)
        df = compute_kinematics(df)
        output_path = _write_enriched_session(df, path, overwrite)
        stats = KinematicsStats(
            file=os.path.basename(path),
            columns_before=before_cols,
            columns_after=len(df.columns),
            output_path=output_path,
        )
        logger.info(
            "Kinematic features written to %s (columns: %d → %d)",
            os.path.basename(output_path),
            stats.columns_before,
            stats.columns_after,
        )
        return stats
    except Exception as exc:
        logger.error("Failed to process %s: %s", os.path.basename(path), exc, exc_info=True)
        return None

# ======================================================================
# BATCH PROCESSING
# ======================================================================
def process_all_kinematics(overwrite: bool = False) -> Optional[int]:
    """
    Compute kinematic features for every file in data/processed/.

    Parameters
    ----------
    overwrite:
        When True, overwrite the processed parquet file. Otherwise, write results
        to data/enriched/enriched_*.
    """
    if not os.path.isdir(PROCESSED_DIR):
        logger.error("Processed directory not found: %s", PROCESSED_DIR)
        return None

    files = sorted(glob(os.path.join(PROCESSED_DIR, "clean_*.parquet")))
    if not files:
        logger.warning("No preprocessed files found in %s", PROCESSED_DIR)
        return 0

    os.makedirs(ENRICHED_DIR, exist_ok=True)

    processed = 0
    for path in files:
        stats = process_single_file(path, overwrite=overwrite)
        if stats:
            processed += 1

    return processed

# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    total = process_all_kinematics()
    if total:
        logger.info("Completed kinematic feature generation for %d files.", total)
