"""
Running signal data preprocessing:
- Data cleaning (filtering physical and physiological outliers).
- Limited interpolation of HR and SpO2.
- Keep all useful columns (gravity, rotation, orientation) for future biomechanics analysis or models.
- Drop only the absolute timestamp column (`Tiempo`).

Saves the preprocessed data in Parquet format under data/processed/.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, List

# ======================================================================
# LOGGING CONFIGURATION
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ======================================================================
# PATH CONFIGURATION
# ======================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ======================================================================
# GLOBAL PARAMETERS
# ======================================================================
FC_RANGE = (40, 220)
SPO2_RANGE = (70, 100)
ACC_MAX = 50.0
INTERP_MAX_GAP_SEC = 1.0
MAX_WORKERS = max(1, os.cpu_count() // 2)

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================
def _estimate_fs(t: pd.Series) -> float:
    """Estimate sampling frequency using the median Δt."""
    t = pd.to_numeric(t, errors="coerce").dropna().values
    if len(t) < 2:
        return np.nan
    dt = np.diff(t)
    dt_med = np.median(dt)
    return np.nan if dt_med <= 0 or not np.isfinite(dt_med) else 1.0 / dt_med


def _interp_limit_from_seconds(fs_est: float, seconds: float) -> int:
    """Convert a time threshold (s) into the number of consecutive samples to interpolate."""
    if not np.isfinite(fs_est) or fs_est <= 0:
        return 5
    return max(1, int(round(fs_est * seconds)))


# ======================================================================
# MAIN PREPROCESSING
# ======================================================================
def preprocess_single_file(filepath: str) -> Optional[pd.DataFrame]:
    """
    Preprocess a single CSV file.
    """
    try:
        df = pd.read_csv(filepath, header=None)
        if df.empty:
            raise ValueError("The file is empty.")

        if df.shape[1] < 15:
            raise ValueError(
                f"File {os.path.basename(filepath)} does not contain enough columns ({df.shape[1]} detected; expected 15)."
            )

        # Assign expected column names
        df.columns = [
            "Tiempo", "AccX", "AccY", "AccZ",
            "GravX", "GravY", "GravZ",
            "RotX", "RotY", "RotZ",
            "Roll", "Pitch", "Yaw",
            "FC", "SpO2"
        ]

        # Relative time
        df["Tiempo"] = pd.to_numeric(df["Tiempo"], errors="coerce")
        df = df.dropna(subset=["Tiempo"]).reset_index(drop=True)
        df["Tiempo_rel"] = df["Tiempo"] - df["Tiempo"].iloc[0]

        # Drop the absolute timestamp column
        df.drop(columns=["Tiempo"], inplace=True, errors="ignore")

        # Estimate sampling frequency
        fs_est = _estimate_fs(df["Tiempo_rel"])
        interp_limit = _interp_limit_from_seconds(fs_est, INTERP_MAX_GAP_SEC)

        # Ensure numeric types
        for col in ["AccX", "AccY", "AccZ", "GravX", "GravY", "GravZ",
                    "RotX", "RotY", "RotZ", "Roll", "Pitch", "Yaw", "FC", "SpO2"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Replace sentinel values (999) with NaNs
        df["FC"].replace(999, np.nan, inplace=True)
        df["SpO2"].replace(999, np.nan, inplace=True)

        # Physiological filtering
        df.loc[~df["FC"].between(*FC_RANGE, inclusive="both"), "FC"] = np.nan
        df.loc[~df["SpO2"].between(*SPO2_RANGE, inclusive="both"), "SpO2"] = np.nan

        # Limited interpolation
        df["FC"] = df["FC"].interpolate(limit=interp_limit, limit_direction="both")
        df["SpO2"] = df["SpO2"].interpolate(limit=interp_limit, limit_direction="both")

        # Remove acceleration outliers
        mask_acc = (df[["AccX", "AccY", "AccZ"]].abs().max(axis=1) < ACC_MAX)
        df = df[mask_acc].reset_index(drop=True)

        return df

    except Exception as e:
        logging.error(f"Error preprocessing {os.path.basename(filepath)}: {e}", exc_info=True)
        return None


def process_file(filepath: str) -> Optional[str]:
    """Process a complete file and save it as Parquet."""
    try:
        df = preprocess_single_file(filepath)
        if df is None or df.empty:
            logging.warning(f"File {os.path.basename(filepath)} was not processed successfully.")
            return None

        filename = os.path.basename(filepath).replace(".csv", ".parquet")
        output_path = os.path.join(PROCESSED_DIR, f"clean_{filename}")
        df.to_parquet(output_path, index=False)
        logging.info(f"✅ File processed and saved: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error processing file {filepath}: {e}", exc_info=True)
        return None


def preprocess_data(parallel: bool = True) -> Optional[List[str]]:
    """Preprocess all CSV files located in data/raw/."""
    if not os.path.isdir(RAW_DIR):
        logging.error(f"Raw data directory not found: {RAW_DIR}")
        return None

    csv_files = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if not csv_files:
        logging.warning("⚠️ No CSV files were found in data/raw/.")
        return None

    logging.info(f"📂 Files detected: {len(csv_files)}")

    processed_files, failed_files = [], []
    if parallel and len(csv_files) > 1:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(process_file, csv_files))
        for fpath, res in zip(csv_files, results):
            (processed_files if res else failed_files).append(res or fpath)
    else:
        for f in csv_files:
            res = process_file(f)
            (processed_files if res else failed_files).append(res or f)

    metadata = {
        "processed_files": [p for p in processed_files if p],
        "failed_files": failed_files,
        "total_processed": len(processed_files),
        "total_failed": len(failed_files),
        "date": str(pd.Timestamp.now())
    }

    metadata_path = os.path.join(PROCESSED_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"🧾 Metadata saved to: {metadata_path}")
    return processed_files


# ======================================================================
# MAIN EXECUTION
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized running signal preprocessing.")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel execution.")
    args = parser.parse_args()

    processed = preprocess_data(parallel=not args.no_parallel)
    if processed:
        logging.info("✅ Preprocessing completed successfully.")
    else:
        logging.warning("⚠️ No files were processed.")
