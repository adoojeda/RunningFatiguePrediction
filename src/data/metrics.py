"""
Biomechanical/physiological metrics (stage 3/5 of the pipeline):
- High-pass filter + integration of centred accelerations → translational velocity (Vtr).
- Jerk computation (acceleration derivative) and fatigue score estimation.
- Saves the enriched sessions and the consolidated metrics table.

Example:
    python src/data/metrics.py --input-dir data/enriched --output-dir data/results

Input:  data/enriched/enriched_*.parquet
Output: updated enriched files + data/results/all_sessions_metrics.parquet
Next:   python src/features/features_extraction.py
"""

# STANDARD LIBRARIES
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# PROJECT ROOT ADJUSTMENT
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# PROJECT LIBRARIES
from src.config import get_config
from src.utils.kinematics_utils import (
    DEFAULT_FS,
    DEFAULT_HP_CUTOFF,
    centre_accelerations,
    compute_acceleration_magnitudes,
    compute_translational_velocity,
    compute_jerk,
)
from src.utils.metrics_utils import (
    compute_session_metrics,
    derive_fatigue_references,
    compute_fatigue_score,
)
from src.utils.schemas import validate_dataframe

# LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# PATHS AND CONFIG
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
DEFAULT_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DEFAULT_RESULTS_DIR = os.path.join(DATA_DIR, "results")
DEFAULT_OUTPUT_PARQUET = os.path.join(DEFAULT_RESULTS_DIR, "all_sessions_metrics.parquet")

CFG = get_config()
SCORE_SMOOTHING = getattr(CFG.fatigue_weights, "smoothing_window", 0)

# EXCEPTION CLASSES AND DATACLASSES
class MetricsError(Exception):
    """Base exception for the metrics pipeline."""

@dataclass
class SessionResult:
    """Summary of a processed session."""
    file: str
    fatigue_score: float
    metrics: Dict[str, float]

# FILE MANAGEMENT
def list_session_files(source_dir: str, files: Optional[Sequence[str]] = None) -> List[str]:
    """Returns the sorted list of session files to process."""
    directory = os.path.abspath(source_dir)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    if files:
        resolved = []
        for item in files:
            path = item if os.path.isabs(item) else os.path.join(directory, item)
            if not path.endswith(".parquet"):
                path = f"{path}.parquet"
            resolved.append(path)
        return resolved

    enriched = sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("enriched_") and f.endswith(".parquet"))
    if enriched:
        return enriched
    processed = sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("clean_") and f.endswith(".parquet"))
    return processed

def save_metrics_table(results: List[Dict[str, float]], output_path: str) -> Optional[pd.DataFrame]:
    """Stores the aggregated metrics table."""
    if not results:
        return None
    df_all = pd.DataFrame(results)
    output_path = output_path or DEFAULT_OUTPUT_PARQUET
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_all.to_parquet(output_path, index=False)
    logger.info("Global metrics stored at %s (%d sessions)", output_path, len(df_all))
    return df_all

# SESSION PROCESSING
def _load_session(path: str) -> pd.DataFrame:
    """Loads a parquet file validating the appropriate schema."""
    df = pd.read_parquet(path)
    schema_name = "enriched" if os.path.basename(path).startswith("enriched_") else "processed"
    validate_dataframe(df, schema_name)
    return df

def _apply_biomechanics(df: pd.DataFrame) -> pd.DataFrame:
    """Applies biomechanical enrichments prior to scoring."""
    if "relative_time" in df.columns:
        df = (
            df.sort_values("relative_time")
            .drop_duplicates(subset="relative_time", keep="first")
            .reset_index(drop=True)
        )
    df = centre_accelerations(df)
    df = compute_acceleration_magnitudes(df)
    df = compute_translational_velocity(
        df,
        default_fs=DEFAULT_FS,
        cutoff=DEFAULT_HP_CUTOFF,
    )
    df = compute_jerk(df)
    return df

def _save_enriched(df: pd.DataFrame, path: str) -> bool:
    """Saves the enriched dataframe to disk."""
    try:
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_parquet(path, index=False)
        logger.info("Enriched file updated: %s", os.path.basename(path))
        return True
    except Exception as exc:
        logger.error(
            "Error while saving enriched file %s: %s",
            os.path.basename(path),
            exc,
            exc_info=True,
        )
        return False

# MAIN PROCESSING FUNCTIONS
def process_session(path: str, *, allow_save: bool) -> Optional[SessionResult]:
    """Processes a session file and returns the computed metrics."""
    try:
        df = _load_session(path)
        df = _apply_biomechanics(df)

        session_metrics = compute_session_metrics(df)
        references = derive_fatigue_references(df)
        session_metrics = compute_fatigue_score(
            session_metrics,
            context="session",
            references=references,
        )
        if allow_save:
            _save_enriched(df, path)

        fatigue_score = session_metrics.get("fatigue_score", np.nan)
        logger.info(
            "Session %s -> fatigue_score=%.3f",
            os.path.basename(path),
            fatigue_score,
        )
        return SessionResult(
            file=os.path.basename(path),
            fatigue_score=float(fatigue_score) if np.isfinite(fatigue_score) else np.nan,
            metrics=session_metrics,
        )

    except Exception as exc:
        logger.error("Error processing %s: %s", os.path.basename(path), exc, exc_info=True)
        return None

# GLOBAL PROCESSING
def process_all_sessions(
    *,
    input_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    save_enriched: bool = True,
    files: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
    """Processes every session and generates the global summary."""
    source_dir = input_dir or (
        DEFAULT_ENRICHED_DIR if os.path.isdir(DEFAULT_ENRICHED_DIR) else DEFAULT_PROCESSED_DIR
    )

    session_files = list_session_files(source_dir, files)
    if not session_files:
        logger.warning("No files found in %s", source_dir)
        return None

    results: List[Dict[str, float]] = []
    for path in session_files:
        allow_save = save_enriched and os.path.basename(path).startswith("enriched_")
        result = process_session(path, allow_save=allow_save)
        if result:
            results.append(
                {
                    "session_file": result.file,
                    "fatigue_score": result.fatigue_score,
                }
            )

    return save_metrics_table(results, output_path or DEFAULT_OUTPUT_PARQUET)

# COMMAND-LINE INTERFACE
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Computes biomechanical metrics and fatigue scores."
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing enriched_*.parquet files (default: data/enriched).",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where the metrics parquet will be stored (default: data/results).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not overwrite the enriched files with the new scores.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Optional list of specific files (paths or names) to process.",
    )
    return parser.parse_args()

# MAIN ENTRY POINT
def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_RESULTS_DIR
    output_path = os.path.join(output_dir, "all_sessions_metrics.parquet")

    df_metrics = process_all_sessions(
        input_dir=args.input_dir,
        output_path=output_path,
        save_enriched=not args.no_save,
        files=args.files,
    )

    if df_metrics is None:
        logger.warning("No metrics were generated.")

if __name__ == "__main__":
    main()
