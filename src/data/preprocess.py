"""
Running signal data preprocessing (pipeline stage 1/5):
- Clean raw CSV recordings into analysis-ready parquet files.
- Filter physical/physiological outliers and interpolate short gaps in HR/SpO₂.
- Keep gravity/rotation/orientation signals for downstream biomechanical models.
- Drop only the absolute timestamp column; create `relative_time` instead.

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
from typing import List, Optional

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import get_config
from src.utils.kinematics import DEFAULT_FS, estimate_sampling_rate
from src.utils.preprocess_utils import (
    apply_physio_filters,
    derive_relative_time,
    ensure_numeric,
    filter_acc_outliers,
    finalise_dataframe,
    interpolate_channels,
    interp_limit_from_seconds,
    load_raw_file,
    PreprocessError,
    EmptyFileError,
    ColumnCountError,
    PreprocessStats,
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
DEFAULT_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DEFAULT_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(DEFAULT_PROCESSED_DIR, exist_ok=True)

CFG = get_config()
PHYSIO_RANGES = CFG.ranges
INTERP_MAX_GAP_SEC = CFG.interpolation.max_gap_seconds
MAX_WORKERS = CFG.workforce.max_workers

# ===========================
# MAIN PREPROCESSING
# ===========================
def preprocess_single_file(filepath: str) -> Optional[pd.DataFrame]:
    """Preprocess a single CSV file into an analysis-ready dataframe."""
    stats = PreprocessStats()
    try:
        df = load_raw_file(filepath)
        stats.samples_in = len(df)
        ensure_numeric(df, df.columns)
        validate_dataframe(df, "raw")
        df = derive_relative_time(df)

        fs_est = estimate_sampling_rate(df["relative_time"]) or DEFAULT_FS
        interp_limit = interp_limit_from_seconds(fs_est, INTERP_MAX_GAP_SEC)

        apply_physio_filters(df, PHYSIO_RANGES.fc, PHYSIO_RANGES.spo2)
        stats.interpolated_fc, stats.interpolated_spo2 = interpolate_channels(df, interp_limit)
        stats.acc_outliers_removed = filter_acc_outliers(df, PHYSIO_RANGES.acc_max)
        df = finalise_dataframe(df)
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

def process_file(filepath: str, output_dir: str) -> Optional[str]:
    """Process a complete file and save it as Parquet."""
    try:
        df = preprocess_single_file(filepath)
        if df is None or df.empty:
            logger.warning("File %s was not processed successfully.", os.path.basename(filepath))
            return None

        filename = os.path.basename(filepath).replace(".csv", ".parquet")
        output_path = os.path.join(output_dir, f"clean_{filename}")
        df.to_parquet(output_path, index=False)
        logger.info("File processed and saved: %s", output_path)
        return output_path
    except Exception as exc:
        logger.error("Error processing file %s: %s", filepath, exc, exc_info=True)
        return None

def preprocess_data(
    parallel: bool = True,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Optional[List[str]]:
    """Preprocess all CSV files located in data/raw/."""
    src_dir = input_dir or DEFAULT_RAW_DIR
    dst_dir = output_dir or DEFAULT_PROCESSED_DIR

    if not os.path.isdir(src_dir):
        logger.error("Raw data directory not found: %s", src_dir)
        return None

    os.makedirs(dst_dir, exist_ok=True)

    csv_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".csv")]
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
            results = list(executor.map(process_file, csv_files, [dst_dir] * len(csv_files)))
        for fpath, res in zip(csv_files, results):
            _record_result(fpath, res)
    else:
        for f in csv_files:
            _record_result(f, process_file(f, dst_dir))

    metadata = {
        "processed_files": [p for p in processed_files if p],
        "failed_files": failed_files,
        "total_processed": len(processed_files),
        "total_failed": len(failed_files),
        "date": str(pd.Timestamp.now()),
    }

    metadata_path = os.path.join(dst_dir, "metadata.json")
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
    parser.add_argument("--input-dir", help="Custom directory containing raw CSV files.")
    parser.add_argument("--output-dir", help="Destination directory for processed parquet files.")
    args = parser.parse_args()
    processed = preprocess_data(
        parallel=not args.no_parallel,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    if processed:
        logger.info("Preprocessing completed successfully.")
    else:
        logger.warning("No files were processed.")
