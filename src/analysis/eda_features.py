"""
Exploratory data analysis (EDA) for the sliding-window feature dataset.

Generates descriptive statistics and a collection of plots that tie biomechanical
and physiological metrics with perceived exertion (RPE).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

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
DEFAULT_DATASET = os.path.join(RESULTS_DIR, "features_dataset_5s_50olap.parquet")
DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "eda_figures")

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

def safe_histogram(data: pd.Series, variable: str, output_dir: str, comment: str) -> None:
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        logger.warning("Skipping histogram for %s; no valid data.", variable)
        return

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

def safe_boxplot(df: pd.DataFrame, x: str, y: str, output_dir: str, comment: str) -> None:
    if y not in df.columns or x not in df.columns:
        logger.warning("Skipping boxplot for %s vs %s; column not found.", y, x)
        return

    data = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty or data[x].nunique() == 0:
        logger.warning("Skipping boxplot for %s vs %s; no valid samples.", y, x)
        return

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

def safe_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str],
    output_dir: str,
    comment: str,
    palette: str = "viridis",
) -> None:
    if x not in df.columns or y not in df.columns:
        logger.warning("Skipping scatter %s vs %s; required columns missing.", x, y)
        return

    data_cols = [x, y]
    if hue and hue in df.columns:
        data_cols.append(hue)
    data = df[data_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        logger.warning("Skipping scatter %s vs %s; no valid samples.", x, y)
        return

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


def correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        logger.warning("No numeric columns available for correlation heatmap.")
        return

    corr = numeric.corr().replace([np.inf, -np.inf], np.nan).dropna(how="all").dropna(axis=1, how="all")
    if corr.empty:
        logger.warning("Correlation matrix is empty after cleaning; skipping heatmap.")
        return

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation heatmap (numeric variables)")
    add_comment("Highlights relationships between biomechanical/physiological variables and perceived exertion.")
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(output_dir, "heatmap_correlations.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Saved correlation heatmap: %s", path)

# ======================================================================
# EDA ROUTINES
# ======================================================================
def describe_dataset(df: pd.DataFrame, output_dir: str) -> None:
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    df.describe().T.to_csv(summary_path)
    logger.info("Summary statistics saved to %s", summary_path)

    nan_path = os.path.join(output_dir, "nan_statistics.csv")
    df.isna().mean().sort_values(ascending=False).to_csv(nan_path, header=["nan_fraction"])
    logger.info("NaN statistics saved to %s", nan_path)

    if "reported_rpe" in df.columns:
        rpe_path = os.path.join(output_dir, "rpe_distribution.csv")
        df["reported_rpe"].value_counts(dropna=False).sort_index().to_csv(rpe_path, header=["count"])
        logger.info("Reported RPE distribution saved to %s", rpe_path)


def plot_distributions(df: pd.DataFrame, output_dir: str) -> None:
    comments = {
        "Vtr_mean": "Translational velocity per window reflects movement intensity.",
        "FC_mean": "Average heart rate summarises physiological response to effort.",
        "jerk_std": "Jerk variability increases with neuromuscular fatigue.",
        "Fatigue_Score": "Composite fatigue index derived from biomechanical and physiological signals.",
    }
    for variable in comments:
        if variable in df.columns:
            safe_histogram(df[variable], variable, output_dir, comments[variable])

def plot_rpe_relationships(df: pd.DataFrame, output_dir: str) -> None:
    comments = {
        "Vtr_mean": "Translational velocity tends to decline as perceived exertion grows.",
        "FC_mean": "Heart rate rises linearly with RPE, validating physiological response.",
        "jerk_std": "Movement variability rises with RPE, signalling loss of motor control.",
        "Fatigue_Score": "Fatigue Score aligns with RPE, bridging objective and subjective fatigue.",
    }
    for variable in comments:
        if variable in df.columns and "reported_rpe" in df.columns:
            safe_boxplot(df, x="reported_rpe", y=variable, output_dir=output_dir, comment=comments[variable])

def plot_specialised_relationships(df: pd.DataFrame, output_dir: str) -> None:
    safe_scatter(
        df,
        x="FC_mean",
        y="reported_rpe",
        hue="session_id" if "session_id" in df.columns else None,
        output_dir=output_dir,
        comment="Heart rate vs RPE shows the physiological-perception linkage.",
    )
    safe_scatter(
        df,
        x="Vtr_mean",
        y="reported_rpe",
        hue="runner_id" if "runner_id" in df.columns else None,
        output_dir=output_dir,
        comment="Higher RPE typically corresponds to lower translational velocity.",
        palette="coolwarm",
    )
    safe_boxplot(
        df,
        x="reported_rpe",
        y="jerk_std",
        output_dir=output_dir,
        comment="Jerk variability captures instability with increasing fatigue.",
    )
    if "Acc_mag_std" in df.columns:
        safe_boxplot(
            df,
            x="reported_rpe",
            y="Acc_mag_std",
            output_dir=output_dir,
            comment="Acceleration variability mirrors degraded motor control.",
        )
    safe_scatter(
        df,
        x="Fatigue_Score",
        y="reported_rpe",
        hue="session_id" if "session_id" in df.columns else None,
        output_dir=output_dir,
        comment="Assesses agreement between computed fatigue score and subjective exertion.",
        palette="flare",
    )

# ======================================================================
# ORCHESTRATION
# ======================================================================
def run_eda(dataset_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = load_features_dataset(path=dataset_path)
    if df is None or df.empty:
        raise ValueError("Feature dataset could not be loaded or is empty.")

    describe_dataset(df, output_dir)
    plot_distributions(df, output_dir)
    plot_rpe_relationships(df, output_dir)
    correlation_heatmap(df, output_dir)
    plot_specialised_relationships(df, output_dir)

    logger.info("EDA completed. Figures stored in %s", output_dir)


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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        run_eda(dataset_path=args.dataset, output_dir=args.output)
    except Exception as exc:
        logger.error("EDA failed: %s", exc, exc_info=True)
