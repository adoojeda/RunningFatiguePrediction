"""
Kinematic feature generation (pipeline stage 2/5):
- Centre accelerations per axis and compute raw/dynamic magnitudes.
- Prepare enriched parquet files consumed by metrics and feature extraction stages.

Input: `data/processed/clean_*.parquet`
Output: `data/enriched/enriched_*.parquet`
Next stage: `python src/data/metrics.py`
"""

import logging
import os
import sys
from glob import glob
from typing import Optional

import pandas as pd

# Ensure project root on sys.path when executed as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.kinematics import centre_accelerations, compute_acceleration_magnitudes

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
# CORE KINEMATIC FEATURES
# ======================================================================
def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure centered acceleration and magnitude features are present in the DataFrame.
    """
    try:
        centre_accelerations(df)
        compute_acceleration_magnitudes(df)
        return df

    except Exception as exc:
        logger.error("Error computing kinematic features: %s", exc, exc_info=True)
        return df

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

    processed_count = 0
    for path in files:
        try:
            df = pd.read_parquet(path)
            before_cols = len(df.columns)
            df = compute_kinematics(df)
            after_cols = len(df.columns)

            if overwrite:
                output_path = path
            else:
                base = os.path.basename(path).replace("clean_", "enriched_")
                output_path = os.path.join(ENRICHED_DIR, base)

            df.to_parquet(output_path, index=False)
            processed_count += 1
            logger.info(
                "Kinematic features written to %s (columns: %s → %s)",
                os.path.basename(output_path),
                before_cols,
                after_cols,
            )
        except Exception as exc:
            logger.error("Failed to process %s: %s", path, exc, exc_info=True)

    return processed_count

# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    total = process_all_kinematics()
    if total:
        logger.info("Completed kinematic feature generation for %d files.", total)
