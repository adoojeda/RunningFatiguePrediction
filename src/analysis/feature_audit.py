"""
Feature audit for `features_dataset_3s_50olap.parquet`.

Generates:
    1) coverage/variance summary per column (CSV),
    2) list of highly correlated feature pairs,
    3) quick RandomForest importances as a proxy for usefulness.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure project root on sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.data_loader import load_features_dataset

# =============================
# LOGGING CONFIGURATION
# =============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================
# CONSTANTS
# =============================
META_COLS = {
    "file",
    "source_file",
    "runner_id",
    "session_id",
    "start_s",
    "end_s",
    "duration",
    "n_samples",
}
TARGET_COLS_BASE = {"reported_rpe", "fatigue_level", "fatigue_score"}
TARGET_LEAKAGE_MAP = {
    "fatigue_level": [
        "fatigue_score",
    ],
}

# =============================
# ARGUMENT PARSING
# =============================
DEFAULT_DATASET = BASE_DIR / "data" / "results" / "features_dataset.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature audit for the sliding-window dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help="Path to the window parquet (default: features_dataset.parquet).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="fatigue_score",
        help="Target column to estimate importance.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASE_DIR / "data" / "results" / "feature_audit"),
        help="Directory to store analysis artifacts.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.95,
        help="Threshold to report highly correlated pairs (default 0.95).",
    )
    return parser.parse_args()

# =============================
# FEATURE AUDIT FUNCTIONS
# =============================
def compute_feature_summary(df: pd.DataFrame, target_cols: set[str], output_dir: Path) -> Path:
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_fraction": df.isna().mean(),
            "n_unique": df.nunique(dropna=True),
        }
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary.loc[numeric_cols, "std"] = df[numeric_cols].std()
    summary.loc[numeric_cols, "mean"] = df[numeric_cols].mean()
    summary.loc[numeric_cols, "min"] = df[numeric_cols].min()
    summary.loc[numeric_cols, "max"] = df[numeric_cols].max()

    summary["is_meta"] = summary.index.isin(META_COLS)
    summary["is_target"] = summary.index.isin(target_cols)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "feature_summary.csv"
    summary.sort_index().to_csv(path)
    logger.info("Feature summary saved to %s", path)
    return path

def correlation_report(df: pd.DataFrame, threshold: float, target_cols: set[str], output_dir: Path) -> Path:
    numeric = df.select_dtypes(include=[np.number]).drop(columns=list(target_cols), errors="ignore")
    if numeric.empty:
        raise ValueError("No numeric columns available for correlation analysis.")

    corr = numeric.corr().replace([np.inf, -np.inf], np.nan)
    pairs: List[Tuple[str, str, float]] = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append((cols[i], cols[j], value))

    df_pairs = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "correlation"]).sort_values(
        by="correlation", key=np.abs, ascending=False
    )
    path = output_dir / "high_correlation_pairs.csv"
    df_pairs.to_csv(path, index=False)
    logger.info("High correlations (%d pairs) saved to %s", len(df_pairs), path)
    return path

def feature_importance_report(df: pd.DataFrame, target: str, target_cols: set[str], output_dir: Path) -> Path:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' is not present in the dataset.")

    y = pd.to_numeric(df[target], errors="coerce")
    X = df.drop(columns=list(META_COLS | target_cols), errors="ignore")
    leakage_cols = TARGET_LEAKAGE_MAP.get(target, [])
    if leakage_cols:
        X = X.drop(columns=[c for c in leakage_cols if c in X.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    mask = np.isfinite(y)
    X = X.loc[mask]
    y = y.loc[mask]

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    importances = pipeline.named_steps["model"].feature_importances_
    ranking = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    df_imp = pd.DataFrame(ranking, columns=["feature", "importance"])

    path = output_dir / "feature_importance_rf.csv"
    df_imp.to_csv(path, index=False)
    logger.info("Importances (RandomForest) saved to %s", path)
    payload = [{"feature": feat, "importance": float(score)} for feat, score in ranking[:20]]
    json_path = output_dir / "feature_importance_top20.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path

# =============================
# MAIN ORCHESTRATION
# =============================
def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"No dataset found at {dataset_path}")

    output_dir = Path(args.output_dir)
    df = load_features_dataset(str(dataset_path))
    if df is None or df.empty:
        raise ValueError("Dataset could not be loaded or is empty.")

    logger.info("Audit started over %d rows and %d columns.", len(df), len(df.columns))

    target_cols = TARGET_COLS_BASE | {args.target}

    compute_feature_summary(df, target_cols=target_cols, output_dir=output_dir)
    correlation_report(df, threshold=args.corr_threshold, target_cols=target_cols, output_dir=output_dir)
    feature_importance_report(df, target=args.target, target_cols=target_cols, output_dir=output_dir)

    logger.info("Audit completed. Check artifacts in %s", output_dir)

if __name__ == "__main__":
    main()
