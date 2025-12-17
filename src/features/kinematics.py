"""
Kinematic feature generation (stage 2/5 of the pipeline):
- Centers per-axis accelerations and computes raw/dynamic magnitudes.
- Produces enriched files consumed by the metrics and feature extraction stages.

Example:
    python src/features/kinematics.py --input-dir data/processed --output-dir data/enriched

Input:  data/processed/clean_*.parquet
Output: data/enriched/enriched_*.parquet
Next:   python src/data/metrics.py
"""

# STANDARD LIBRARIES
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

# PROJECT SETUP
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# PROJECT LIBRARIES
from src.utils.kinematics_utils import centre_accelerations, compute_acceleration_magnitudes
from src.utils.schemas import validate_dataframe

# LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# PATH CONFIGURATION
BASE_DIR = PROJECT_ROOT
DEFAULT_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DEFAULT_ENRICHED_DIR = os.path.join(BASE_DIR, "data", "enriched")

# TYPE DEFINITIONS
class KinematicsError(Exception):
    """Raised when a kinematic transformation fails."""

# DATACLASS DEFINITIONS
@dataclass
class KinematicsStats:
    """Lightweight stats container for logging."""
    file: str
    columns_before: int
    columns_after: int
    output_path: str

# REVISION FUNCTIONS
def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures centred accelerations and associated magnitudes are present."""
    centre_accelerations(df)
    compute_acceleration_magnitudes(df)
    return df

def _load_processed_session(path: str) -> pd.DataFrame:
    """Loads a processed parquet and validates its schema."""
    df = pd.read_parquet(path)
    validate_dataframe(df, "processed")
    return df

# SAVE FUNCTIONS
def _write_enriched_session(
    df: pd.DataFrame,
    source_path: str,
    *,
    output_dir: str,
) -> str:
    """Stores the dataframe in the enriched directory."""
    validate_dataframe(df, "enriched")
    base = os.path.basename(source_path).replace("clean_", "enriched_")
    output_path = os.path.join(output_dir, base)
    df.to_parquet(output_path, index=False)
    return output_path

# SINGLE FILE PROCESSING
def process_single_file(
    path: str,
    *,
    output_dir: str,
) -> Optional[KinematicsStats]:
    """Load → enrich → save a single session. Returns stats or None if it fails."""
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
            "Kinematic features saved to %s (columns: %d → %d)",
            os.path.basename(output_path),
            stats.columns_before,
            stats.columns_after,
        )
        return stats
    except Exception as exc:
        logger.error("Failed to process %s: %s", os.path.basename(path), exc, exc_info=True)
        return None

# BATCH PROCESSING
def list_processed_files(source_dir: str) -> List[str]:
    """Lists all processed files in the given directory."""
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Processed directory not found: {source_dir}")
    files = sorted(
        os.path.join(source_dir, fname)
        for fname in os.listdir(source_dir)
        if fname.endswith(".parquet") and fname.startswith("clean_")
    )
    return files

# PROCESS ALL FILES
def process_all_kinematics(
    *,
    processed_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    files: Optional[Sequence[str]] = None,
) -> Optional[int]:
    """Computes kinematic features for every available file."""
    src_dir = processed_dir or DEFAULT_PROCESSED_DIR
    dst_dir = output_dir or DEFAULT_ENRICHED_DIR

    if files is None:
        try:
            files = list_processed_files(src_dir)
        except FileNotFoundError as exc:
            logger.error(exc)
            return None

    if not files:
        logger.warning("No files found to process in %s.", src_dir)
        return 0

    os.makedirs(dst_dir, exist_ok=True)

    processed = 0
    for path in files:
        stats = process_single_file(path, output_dir=dst_dir)
        if stats:
            processed += 1

    return processed

# COMMAND-LINE INTERFACE
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates centred accelerations and derived magnitudes.")
    parser.add_argument("--input-dir", help="Directory containing clean_*.parquet files (default data/processed).")
    parser.add_argument("--output-dir", help="Destination directory for enriched files (default data/enriched).")
    parser.add_argument("--files", nargs="*", help="Optional list of files to process (path or name).")
    return parser.parse_args()

# MAIN FUNCTION
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
        logger.info("Kinematic generation completed for %d files.", total)
    elif total == 0:
        logger.warning("No files were processed.")

if __name__ == "__main__":
    main()
