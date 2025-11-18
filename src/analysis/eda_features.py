"""
Exploratory data analysis (EDA) for the sliding-window feature dataset.

Generates descriptive statistics and a collection of plots that tie biomechanical
and physiological metrics with perceived exertion (RPE). Supports filtering by
session/metric and exporting the results as a ZIP report.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Optional
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")  # Safe backend for headless environments.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure project root is on sys.path when executed as a script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils.data_loader import load_features_dataset

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
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
DEFAULT_DATASET = os.path.join(RESULTS_DIR, "features_dataset_3s_50olap.parquet")
DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "eda_figures")

# Centralised metric configuration
DISTRIBUTION_METRICS: Dict[str, str] = {
    "vtr_mean": "Translational velocity per window reflects movement intensity.",
    "fc_mean": "Average heart rate summarises physiological response to effort.",
    "fatigue_score": "Composite fatigue index derived from biomechanical and physiological signals.",
}

RPE_RELATIONSHIPS: Dict[str, str] = {
    "vtr_mean": "Translational velocity tends to decline as perceived exertion grows.",
    "fc_mean": "Heart rate rises linearly with RPE, validating physiological response.",
    "jerk_std": "Movement variability rises with RPE, signalling loss of motor control.",
    "fatigue_score": "Fatigue score aligns with RPE, bridging objective and subjective fatigue.",
}

# ======================================================================
# HELPERS
# ======================================================================
def add_comment(text: str, bottom: float = -0.08) -> None:
    """Add a contextual comment below the plot."""
    if not text:
        return
    plt.figtext(
        0.5,
        bottom,
        text,
        wrap=True,
        ha="center",
        fontsize=9,
        color="gray",
        style="italic",
    )

def safe_histogram(data: pd.Series, variable: str, output_dir: str, comment: str) -> Optional[str]:
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        logger.warning("Skipping histogram for %s; no valid data.", variable)
        return None

    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    plt.title(f"Distribution of {variable}")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    add_comment(comment)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(output_dir, f"dist_{variable}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Saved histogram: %s", path)
    return path

def safe_boxplot(df: pd.DataFrame, x: str, y: str, output_dir: str, comment: str) -> Optional[str]:
    if y not in df.columns or x not in df.columns:
        logger.warning("Skipping boxplot for %s vs %s; column not found.", y, x)
        return None

    data = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty or data[x].nunique() == 0:
        logger.warning("Skipping boxplot for %s vs %s; no valid samples.", y, x)
        return None

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=x, y=y, data=data, palette="coolwarm")
    plt.title(f"{y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(alpha=0.3)
    add_comment(comment)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(output_dir, f"box_{y}_vs_{x}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Saved boxplot: %s", path)
    return path

def safe_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str],
    output_dir: str,
    comment: str,
    palette: str = "viridis",
) -> Optional[str]:
    if x not in df.columns or y not in df.columns:
        logger.warning("Skipping scatter %s vs %s; required columns missing.", x, y)
        return None

    data_cols = [x, y]
    if hue and hue in df.columns:
        data_cols.append(hue)
    data = df[data_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        logger.warning("Skipping scatter %s vs %s; no valid samples.", x, y)
        return None

    plt.figure(figsize=(7, 5))
    hue_arg = hue if hue and hue in data.columns else None
    sns.scatterplot(x=x, y=y, hue=hue_arg, data=data, palette=palette, alpha=0.6)
    sns.regplot(x=x, y=y, data=data, scatter=False, color="black", ci=None)
    plt.title(f"{x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(alpha=0.3)
    add_comment(comment)
    if hue_arg:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(output_dir, f"scatter_{x}_vs_{y}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Saved scatter plot: %s", path)
    return path


# ======================================================================
# EDA ROUTINES
# ======================================================================
def plot_distributions(df: pd.DataFrame, output_dir: str, metrics: Optional[Iterable[str]] = None) -> List[str]:
    comments = DISTRIBUTION_METRICS
    selected = list(metrics) if metrics else list(comments)
    paths: List[str] = []
    for variable in selected:
        if variable in df.columns and variable in comments:
            saved = safe_histogram(df[variable], variable, output_dir, comments[variable])
            if saved:
                paths.append(saved)
        elif variable not in df.columns:
            logger.warning("Distribution metric %s not found in dataset.", variable)
    return paths

def plot_rpe_relationships(df: pd.DataFrame, output_dir: str, metrics: Optional[Iterable[str]] = None) -> List[str]:
    comments = RPE_RELATIONSHIPS
    _ensure_columns(df, ["reported_rpe"])
    selected = list(metrics) if metrics else list(comments)
    paths: List[str] = []
    for variable in selected:
        if variable in df.columns and variable in comments:
            saved = safe_boxplot(df, x="reported_rpe", y=variable, output_dir=output_dir, comment=comments[variable])
            if saved:
                paths.append(saved)
        elif variable not in df.columns:
            logger.warning("RPE relationship metric %s not found in dataset.", variable)
    return paths

def plot_specialised_relationships(df: pd.DataFrame, output_dir: str) -> List[str]:
    _ensure_columns(df, ["reported_rpe"])
    paths: List[str] = []
    saved = safe_scatter(
        df,
        x="fc_mean",
        y="reported_rpe",
        hue="session_id" if "session_id" in df.columns else None,
        output_dir=output_dir,
        comment="Heart rate vs RPE shows the physiological-perception linkage.",
    )
    if saved:
        paths.append(saved)
    saved = safe_scatter(
        df,
        x="Vtr_mean",
        y="reported_rpe",
        hue="runner_id" if "runner_id" in df.columns else None,
        output_dir=output_dir,
        comment="Higher RPE typically corresponds to lower translational velocity.",
        palette="coolwarm",
    )
    if saved:
        paths.append(saved)
    saved = safe_boxplot(
        df,
        x="reported_rpe",
        y="jerk_std",
        output_dir=output_dir,
        comment="Jerk variability captures instability with increasing fatigue.",
    )
    if saved:
        paths.append(saved)
    saved = safe_scatter(
        df,
        x="fatigue_score",
        y="reported_rpe",
        hue="session_id" if "session_id" in df.columns else None,
        output_dir=output_dir,
        comment="Assesses agreement between computed fatigue score and subjective exertion.",
        palette="flare",
    )
    if saved:
        paths.append(saved)
    return paths

# ======================================================================
# ORCHESTRATION
# ======================================================================
def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from dataset: {missing}")


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
        raise ValueError("Feature dataset could not be loaded or is empty.")

    if sessions:
        filters = set(str(s) for s in sessions)
        mask = pd.Series(False, index=df.index)
        for col in ("session_id", "file", "source_file", "runner_id"):
            if col in df.columns:
                mask |= df[col].astype(str).isin(filters)
        df = df[mask]
        if df.empty:
            raise ValueError("No rows remain after applying session filters.")
        logger.info("Filtered dataset to %d rows using sessions %s", len(df), sorted(filters))

    metric_list = [m.strip() for m in metrics] if metrics else None
    generated_files: List[str] = []

    generated_files.extend(plot_distributions(df, timestamped_dir, metrics=metric_list))
    generated_files.extend(plot_rpe_relationships(df, timestamped_dir, metrics=metric_list))
    generated_files.extend(plot_specialised_relationships(df, timestamped_dir))

    # Remove duplicates while preserving order
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
        logger.info("Zipped report generated at %s", zip_path)
        if clean_after_zip:
            for file_path in unique_files:
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            try:
                os.rmdir(timestamped_dir)
            except OSError:
                logger.warning("Could not remove directory %s after cleanup.", timestamped_dir)
    else:
        zip_path = None

    logger.info("EDA completed. Figures stored in %s", timestamped_dir)
    return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory data analysis for the sliding-window feature dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Path to the feature dataset (default: {DEFAULT_DATASET}).",
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
        help="Optional session identifiers (session_id/file/source_file) to include in the EDA.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Optional list of metric columns to focus on (applies to distributions and RPE plots).",
    )
    parser.add_argument(
        "--zip",
        nargs="?",
        const="auto",
        default=None,
        help="Create a ZIP archive of generated outputs. Optionally supply the archive name.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove generated figures after creating the ZIP archive (only when --zip is used).",
    )
    return parser.parse_args()

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
            logger.info("🗜️  Compressed report available at %s", zip_path)
    except Exception as exc:
        logger.error("EDA failed: %s", exc, exc_info=True)
