"""
Feature audit for `features_dataset_3s_50olap.parquet`.

Generates:
    1) coverage/variance summary per column (CSV),
    2) list of highly correlated feature pairs,
    3) quick RandomForest importances as a proxy for usefulness.

Usage:
    python src/analysis/feature_audit.py \
        --dataset data/results/features_dataset_3s_50olap.parquet \
        --target reported_rpe
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
TARGET_COLS_BASE = {"reported_rpe", "fatigue_level"}
TARGET_LEAKAGE_MAP = {
    "Fatigue_Score": [
        "Fatigue_component_norm_fc",
        "Fatigue_component_norm_acc",
        "Fatigue_component_norm_jerk",
        "Fatigue_component_norm_spo2",
    ],
    "fatigue_level": [
        "Fatigue_Score",
        "Fatigue_component_norm_fc",
        "Fatigue_component_norm_acc",
        "Fatigue_component_norm_jerk",
        "Fatigue_component_norm_spo2",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature audit for the sliding-window dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(BASE_DIR / "data" / "results" / "features_dataset_3s_50olap.parquet"),
        help="Ruta al parquet con las ventanas (default: dataset de 3 s).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="reported_rpe",
        help="Columna objetivo para estimar importancia (default: reported_rpe).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASE_DIR / "data" / "results" / "feature_audit"),
        help="Directorio donde guardar los artefactos del análisis.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.95,
        help="Umbral para reportar pares altamente correlacionados (default 0.95).",
    )
    return parser.parse_args()


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
    logger.info("Feature summary guardado en %s", path)
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
    logger.info("Correlaciones altas (%d pares) guardadas en %s", len(df_pairs), path)
    return path


def feature_importance_report(df: pd.DataFrame, target: str, target_cols: set[str], output_dir: Path) -> Path:
    if target not in df.columns:
        raise KeyError(f"La columna objetivo '{target}' no está presente en el dataset.")

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
    logger.info("Importancias (RandomForest) guardadas en %s", path)
    # Also keep JSON with top 20 for quick viewing
    payload = [{"feature": feat, "importance": float(score)} for feat, score in ranking[:20]]
    json_path = output_dir / "feature_importance_top20.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset: {dataset_path}")

    output_dir = Path(args.output_dir)
    df = load_features_dataset(str(dataset_path))
    if df is None or df.empty:
        raise ValueError("El dataset no pudo cargarse o está vacío.")

    logger.info("Auditoría iniciada sobre %d filas y %d columnas.", len(df), len(df.columns))

    target_cols = TARGET_COLS_BASE | {args.target}

    compute_feature_summary(df, target_cols=target_cols, output_dir=output_dir)
    correlation_report(df, threshold=args.corr_threshold, target_cols=target_cols, output_dir=output_dir)
    feature_importance_report(df, target=args.target, target_cols=target_cols, output_dir=output_dir)

    logger.info("Auditoría completada. Revisa los artefactos en %s", output_dir)


if __name__ == "__main__":
    main()
