"""
Feature audit for `results/features_dataset.parquet`.

Example:
    python src/analysis/feature_audit.py --dataset data/results/features_dataset.parquet --target fatigue_score

Generates:
    1) per-column coverage/variance summary (CSV),
    2) list of highly correlated feature pairs,
    3) quick RandomForest importances as a utility proxy.
"""

# STANDARD LIBRARIES
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# PROJECT SETUP
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# PROJECT IMPORTS
from src.utils.data_loader import load_features_dataset

# LOGGING CONFIGURATION
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# METADATA AND TARGETS
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

# DEFAULT DATASET PATH
DEFAULT_DATASET = BASE_DIR / "data" / "results" / "features_dataset.parquet"

# ARGPARSE SETUP
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sliding-window dataset audit.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help="Path to the features parquet (default: features_dataset.parquet).",
    )
    parser.add_argument(
        "--target",
        choices=["fatigue_score", "reported_rpe"],
        default="fatigue_score",
        help="Target column for importance estimation (fatigue_score or reported_rpe).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASE_DIR / "data" / "results" / "feature_audit"),
        help="Directory to store audit artefacts.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.95,
        help="Threshold to flag high-correlation pairs (default 0.95).",
    )
    return parser.parse_args()

# FEATURE AUDIT FUNCTIONS
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
    logger.info("Feature summary stored at %s", path)
    return path

def correlation_report(df: pd.DataFrame, threshold: float, target_cols: set[str], output_dir: Path) -> Path:
    numeric = df.select_dtypes(include=[np.number]).drop(columns=list(target_cols), errors="ignore")
    if numeric.empty:
        raise ValueError("No numeric columns available to analyse correlations.")

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
    logger.info("Correlated pairs (%d) stored at %s", len(df_pairs), path)
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
    logger.info("Importances (RandomForest) stored at %s", path)
    payload = [{"feature": feat, "importance": float(score)} for feat, score in ranking[:20]]
    json_path = output_dir / "feature_importance_top20.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path

# MAIN EXECUTION
def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    output_dir = Path(args.output_dir)
    df = load_features_dataset(str(dataset_path))
    if df is None or df.empty:
        raise ValueError("No data loaded from the dataset.")

    logger.info("Loaded dataset with %d rows and %d columns", len(df), len(df.columns))

    target_cols = TARGET_COLS_BASE | {args.target}

    compute_feature_summary(df, target_cols=target_cols, output_dir=output_dir)
    correlation_report(df, threshold=args.corr_threshold, target_cols=target_cols, output_dir=output_dir)
    feature_importance_report(df, target=args.target, target_cols=target_cols, output_dir=output_dir)

    logger.info("Feature audit completed successfully.")

if __name__ == "__main__":
    main()
