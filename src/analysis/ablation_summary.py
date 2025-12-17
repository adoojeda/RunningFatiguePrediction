"""
Ablation results summary.

Example:
    python src/analysis/ablation_summary.py --ablation-dir data/results/modeling/ablation

Scans subdirectories under data/results/modeling/ablation/<tag>,
combines their summary.csv files, and if a baseline exists, computes
metric deltas relative to it.
"""

# STANDARD LIBRARIES
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

# LOGGING CONFIGURATION
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# PROJECT CONFIGURATION
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ABLATION_DIR = PROJECT_ROOT / "data" / "results" / "modeling" / "ablation"

# ARGUMENT PARSING
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates a combined summary of ablation experiments.")
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR, help="Base directory with ablation runs.")
    parser.add_argument("--output", type=Path, default=None, help="Path for the combined CSV (optional).")
    parser.add_argument("--deltas-output", type=Path, default=None, help="Path for the baseline delta CSV (optional).")
    return parser.parse_args()

# DATA COLLECTION
def collect_summary(tag_dir: Path) -> Optional[pd.DataFrame]:
    """Collects the most recent summary.csv from the given tag directory."""
    if not tag_dir.exists():
        return None
    experiment_dirs = sorted([d for d in tag_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for exp_dir in experiment_dirs:
        summary_path = exp_dir / "summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df["experiment"] = exp_dir.name
            return df
    return None

# MAIN FUNCTION
def main() -> None:
    args = parse_args()
    if not args.ablation_dir.exists():
        raise FileNotFoundError(f"Ablation directory not found: {args.ablation_dir}")

    combined: List[pd.DataFrame] = []
    for tag_dir in sorted([d for d in args.ablation_dir.iterdir() if d.is_dir()]):
        df = collect_summary(tag_dir)
        if df is None:
            logger.warning("summary.csv not found in %s", tag_dir)
            continue
        df["block_tag"] = tag_dir.name
        combined.append(df)

    if not combined:
        raise RuntimeError("No ablation results were found.")

    combined_df = pd.concat(combined, ignore_index=True)
    combined_df_path = args.output or (args.ablation_dir / "ablation_summary.csv")
    combined_df.to_csv(combined_df_path, index=False)
    logger.info("Combined summary stored at %s", combined_df_path)

    baseline_df = combined_df[combined_df["block_tag"] == "baseline"]
    if baseline_df.empty:
        logger.warning("Baseline not found; skipping deltas.")
        return

    deltas: List[pd.DataFrame] = []
    baseline_key = ["model", "split"]
    for tag in combined_df["block_tag"].unique():
        if tag == "baseline":
            continue
        df_tag = combined_df[combined_df["block_tag"] == tag]
        merged = pd.merge(
            df_tag,
            baseline_df,
            on=baseline_key,
            suffixes=("", "_baseline"),
            how="inner",
        )
        for metric in ["mae_mean", "rmse_mean", "r2_mean"]:
            merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_baseline"]
        merged["block_tag"] = tag
        deltas.append(merged)

    if deltas:
        deltas_df = pd.concat(deltas, ignore_index=True)
        deltas_path = args.deltas_output or (args.ablation_dir / "ablation_deltas.csv")
        deltas_df.to_csv(deltas_path, index=False)
        logger.info("Baseline deltas stored at %s", deltas_path)
    else:
        logger.warning("Could not compute deltas; check baseline and additional experiments.")

# ENTRY POINT
if __name__ == "__main__":
    main()
