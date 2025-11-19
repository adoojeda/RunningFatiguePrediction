"""
Kinematic feature generation (pipeline stage 2/5):
- Centre accelerations per axis and compute raw/dynamic magnitudes.
- Prepare enriched parquet files consumed by metrics and feature extraction stages.

Input: `data/processed/clean_*.parquet`
Output: `data/enriched/enriched_*.parquet`
Next stage: `python src/data/metrics.py`
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.kinematics import centre_accelerations, compute_acceleration_magnitudes
from src.utils.schemas import validate_dataframe

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
BASE_DIR = PROJECT_ROOT
DEFAULT_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DEFAULT_ENRICHED_DIR = os.path.join(BASE_DIR, "data", "enriched")

# ===========================
# CUSTOM TYPES
# ===========================
class KinematicsError(Exception):
    """Raised when a kinematic transformation fails."""

@dataclass
class KinematicsStats:
    """Simple structure capturing relevant processing metadata."""

    file: str
    columns_before: int
    columns_after: int
    output_path: str

# ===========================
# CORE LOGIC
# ===========================
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

def _write_enriched_session(
    df: pd.DataFrame,
    source_path: str,
    *,
    output_dir: str,
) -> str:
    """Persist the dataframe to the enriched directory."""
    validate_dataframe(df, "enriched")
    base = os.path.basename(source_path).replace("clean_", "enriched_")
    output_path = os.path.join(output_dir, base)
    df.to_parquet(output_path, index=False)
    return output_path

def process_single_file(
    path: str,
    *,
    output_dir: str,
) -> Optional[KinematicsStats]:
    """
    Load → enrich → save a single session. Returns stats or None on failure.
    """
    try:
        df = _load_processed_session(path)
        before_cols = len(df.columns)
        df = compute_kinematics(df)
        output_path = _write_enriched_session(df, path, output_dir=output_dir)
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

# ===========================
# BATCH PROCESSING
# ===========================
def list_processed_files(source_dir: str) -> List[str]:
    """Return sorted list of clean parquet files from the provided directory."""
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Processed directory not found: {source_dir}")
    files = sorted(
        os.path.join(source_dir, fname)
        for fname in os.listdir(source_dir)
        if fname.endswith(".parquet") and fname.startswith("clean_")
    )
    return files


def process_all_kinematics(
    *,
    processed_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    files: Optional[Sequence[str]] = None,
) -> Optional[int]:
    """
    Compute kinematic features for every file in data/processed/.

    Parameters
    ----------
    overwrite:
        When True, overwrite the processed parquet file. Otherwise, write results
        to data/enriched/enriched_*.
    """
    src_dir = processed_dir or DEFAULT_PROCESSED_DIR
    dst_dir = output_dir or DEFAULT_ENRICHED_DIR

    if files is None:
        try:
            files = list_processed_files(src_dir)
        except FileNotFoundError as exc:
            logger.error(exc)
            return None

    if not files:
        logger.warning("No preprocessed files found in %s", src_dir)
        return 0

    os.makedirs(dst_dir, exist_ok=True)

    processed = 0
    for path in files:
        stats = process_single_file(path, output_dir=dst_dir)
        if stats:
            processed += 1

    return processed

# ===========================
# MAIN
# ===========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate centred accelerations and magnitudes.")
    parser.add_argument("--input-dir", help="Directory with clean_*.parquet files (default: data/processed).")
    parser.add_argument("--output-dir", help="Destination directory for enriched files (default: data/enriched).")
    parser.add_argument("--files", nargs="*", help="Optional explicit list of files to process (paths or basenames).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_list: Optional[List[str]] = None
    if args.files:
        base_dir = args.input_dir or DEFAULT_PROCESSED_DIR
        file_list = []
        for item in args.files:
            path = item if item.endswith(".parquet") else os.path.join(base_dir, item)
            file_list.append(path)

    total = process_all_kinematics(
        processed_dir=args.input_dir,
        output_dir=args.output_dir,
        files=file_list,
    )
    if total:
        logger.info("Completed kinematic feature generation for %d files.", total)
    elif total == 0:
        logger.warning("No files were processed.")


if __name__ == "__main__":
    main()
