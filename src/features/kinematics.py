"""
Kinematic feature generation for running signals:
- Compute centered accelerations per axis.
- Derive total and dynamic acceleration magnitudes.
- Provide the baseline features required by metrics and feature-extraction modules.
"""

import os
import logging
from glob import glob
from typing import Optional

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
        # Center each acceleration axis to remove bias / gravity components.
        for axis in ["X", "Y", "Z"]:
            col = f"Acc{axis}"
            centered_col = f"{col}_centered"
            if col not in df.columns:
                logger.warning("Column %s not found; skipping centering for this axis.", col)
                continue
            df[centered_col] = df[col] - df[col].mean()

        required_cols = ["AccX", "AccY", "AccZ", "AccX_centered", "AccY_centered", "AccZ_centered"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.warning("Skipping magnitude computation; missing columns: %s", missing)
            return df

        # Magnitude of raw acceleration.
        df["Acc_mag"] = np.sqrt(df["AccX"] ** 2 + df["AccY"] ** 2 + df["AccZ"] ** 2)

        # Magnitude of centered (dynamic) acceleration.
        df["Acc_dyn_mag"] = np.sqrt(
            df["AccX_centered"] ** 2 + df["AccY_centered"] ** 2 + df["AccZ_centered"] ** 2
        )

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
