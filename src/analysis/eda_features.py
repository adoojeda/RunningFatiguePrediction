"""
Exploratory data analysis (EDA) for the sliding-window dataset.

Generates descriptive statistics and a series of plots linking biomechanical/physiological
metrics with perceived exertion (RPE). Supports filtering by session/metric and exporting
all figures inside a ZIP archive.

Example:
    python src/analysis/eda_features.py --dataset data/results/features_dataset.parquet --output data/results/eda_figures
"""

# STANDARD LIBRARY IMPORTS
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")  

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ROOT PROJECT CONFIGURATION
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# PROJECT IMPORTS
from src.utils.data_loader import load_features_dataset
from src.utils.eda_plots import (
    plot_correlation_heatmap,
    plot_fatigue_levels,
    plot_rpe_relationships,
    plot_runner_facets,
    plot_specialised_relationships,
)

# LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ROOT DIRECTORIES AND CONSTANTS
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
DEFAULT_DATASET = os.path.join(RESULTS_DIR, "features_dataset.parquet")
DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "eda_figures")

RPE_METRIC_CANDIDATES = [
    "hr_mean",
    "fatigue_score",
    "jerk_std",
    "acc_mean",
    "acc_std",
    "acc_mag_mad",
    "vtr_mean",
    "spo2_mean",
    "fc_mean",
]

P_VALUE_PAIRS: List[tuple[str, str]] = [
    ("hr_mean", "fatigue_score"),
    ("hr_mean", "reported_rpe"),
    ("hr_mean", "vtr_mean"),
    ("hr_mean", "jerk_std"),
    ("fatigue_score", "reported_rpe"),
    ("fatigue_score", "vtr_mean"),
    ("fatigue_score", "jerk_std"),
    ("reported_rpe", "vtr_mean"),
    ("reported_rpe", "jerk_std"),
    ("vtr_mean", "jerk_std"),
]

# P-VALUE COMPUTATION
def compute_pvalues(
    df: pd.DataFrame,
    output_dir: str,
    pairs: Optional[List[tuple[str, str]]] = None,
) -> Optional[str]:
    """Computes Pearson correlation p-values for selected pairs."""
    pairs = pairs or P_VALUE_PAIRS
    records: List[Dict[str, float]] = []
    for x, y in pairs:
        if x not in df.columns or y not in df.columns:
            logger.warning("Unable to compute p-value for %s vs %s; missing columns.", x, y)
            continue
        sample = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sample) < 3:
            logger.warning("Not enough samples for %s vs %s; skipping.", x, y)
            continue
        try:
            r, p = stats.pearsonr(sample[x], sample[y])
            records.append(
                {
                    "x": x,
                    "y": y,
                    "n": len(sample),
                    "pearson_r": r,
                    "p_value": p,
                }
            )
        except Exception as exc:
            logger.warning("Error computing p-value for %s vs %s: %s", x, y, exc)

    if not records:
        logger.warning("No p-values were generated; check dataset and columns.")
        return None

    df_p = pd.DataFrame(records)
    path = os.path.join(output_dir, "pvalues.csv")
    df_p.to_csv(path, index=False)
    logger.info("p-value table stored at %s", path)
    return path

def _available_numeric_metrics(df: pd.DataFrame) -> List[str]:
    """Return the subset of metric candidates present in the dataset."""
    present = [col for col in RPE_METRIC_CANDIDATES if col in df.columns]
    if present:
        return present
    return df.select_dtypes(include=[np.number]).columns.tolist()

# MAIN EDA FUNCTION
def run_eda(
    dataset_path: str,
    output_dir: str,
    sessions: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    zip_name: Optional[str] = None,
    clean_after_zip: bool = False,
) -> Optional[str]:
    timestamped_dir = os.path.join(output_dir, datetime.now().strftime("eda_%Y%m%d_%H%M%S"))
    os.makedirs(timestamped_dir, exist_ok=True)
    df = load_features_dataset(path=dataset_path)
    if df is None or df.empty:
        raise ValueError("The features dataset could not be loaded or is empty.")

    if sessions:
        filters = set(str(s) for s in sessions)
        mask = pd.Series(False, index=df.index)
        for col in ("session_id", "file", "source_file", "runner_id"):
            if col in df.columns:
                mask |= df[col].astype(str).isin(filters)
        df = df[mask]
        if df.empty:
            raise ValueError("No rows left after applying the session filters.")
        logger.info("Dataset filtered to %d rows using sessions %s", len(df), sorted(filters))

    metric_list = [m.strip() for m in metrics] if metrics else None
    available_metrics = _available_numeric_metrics(df)
    generated_files: List[str] = []

    generated_files.extend(
        plot_rpe_relationships(
            df,
            timestamped_dir,
            metrics=metric_list,
            available_metrics=available_metrics,
        )
    )
    generated_files.extend(plot_specialised_relationships(df, timestamped_dir))
    heatmap = plot_correlation_heatmap(df, timestamped_dir)
    if heatmap:
        generated_files.append(heatmap)
    generated_files.extend(
        plot_runner_facets(
            df,
            timestamped_dir,
            metrics=available_metrics,
        )
    )
    level_box = plot_fatigue_levels(df, timestamped_dir)
    if level_box:
        generated_files.append(level_box)
    pvalue_file = compute_pvalues(df, timestamped_dir, pairs=P_VALUE_PAIRS)
    if pvalue_file:
        generated_files.append(pvalue_file)

    seen = set()
    unique_files = []
    for path in generated_files:
        if path and os.path.exists(path) and path not in seen:
            seen.add(path)
            unique_files.append(path)

    zip_path: Optional[str] = None
    if zip_name:
        base = zip_name if zip_name != "auto" else f"eda_report_{datetime.now():%Y%m%d_%H%M%S}"
        if not base.endswith(".zip"):
            base = f"{base}.zip"
        zip_path = os.path.join(os.path.dirname(timestamped_dir), base)
        with ZipFile(zip_path, "w") as archive:
            for file_path in unique_files:
                arcname = os.path.relpath(file_path, timestamped_dir) if file_path.startswith(timestamped_dir) else os.path.basename(file_path)
                archive.write(file_path, arcname=arcname)
        logger.info("Figures zipped into %s", zip_path)
        if clean_after_zip:
            for file_path in unique_files:
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            try:
                os.rmdir(timestamped_dir)
            except OSError:
                logger.warning("Not able to remove directory %s", timestamped_dir)
    else:
        zip_path = None

    logger.info("EDA completed. Figures stored in %s", timestamped_dir)
    return zip_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA for the sliding-window features dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Root path to the features dataset (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store generated figures (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--session",
        nargs="+",
        help="Optional list of session IDs/files/runners to filter the dataset.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Optional list of metric columns to highlight (applies to histograms and RPE plots).",
    )
    parser.add_argument(
        "--zip",
        nargs="?",
        const="auto",
        default=None,
        help="Creates a ZIP with the generated figures. A filename can be specified.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Deletes the figures after creating the ZIP (only if --zip is used).",
    )
    return parser.parse_args()

# MAIN ENTRY POINT
if __name__ == "__main__":
    args = parse_args()
    try:
        zip_path = run_eda(
            dataset_path=args.dataset,
            output_dir=args.output,
            sessions=args.session,
            metrics=args.metrics,
            zip_name=args.zip,
            clean_after_zip=args.clean,
        )
        if zip_path:
            logger.info("EDA ZIP created at %s", zip_path)
    except Exception as exc:
        logger.error("EDA process failed: %s", exc)
