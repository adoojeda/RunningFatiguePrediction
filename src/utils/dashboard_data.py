"""Shared data helpers for the Dash dashboard."""

# STANDARD LIBRARIES
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PROJECT LIBRARIES
from src.config import get_config
from src.features.features_extraction import extract_features_from_file
from src.utils.data_loader import load_data

# BASE PATHS AND CONSTANTS
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
ENRICHED_DIR = DATA_DIR / "enriched"
EXPERIMENTS_DIR = DATA_DIR / "results" / "modeling" / "experiments"
EXCLUDED_PREFIXES = ("all_sessions_metrics", "features_dataset")

# COLUMN RENAMING MAPPING
COL_RENAME = {
    "Relative_Time": "relative_time",
    "Tiempo_rel": "relative_time",
    "AccX": "acc_x",
    "AccY": "acc_y",
    "AccZ": "acc_z",
    "GravX": "grav_x",
    "GravY": "grav_y",
    "GravZ": "grav_z",
    "RotX": "rot_x",
    "RotY": "rot_y",
    "RotZ": "rot_z",
    "Roll": "roll",
    "Pitch": "pitch",
    "Yaw": "yaw",
    "FC": "hr",
    "HR": "hr",
    "SpO2": "spo2",
    "Vtr": "vtr",
}

# DEFAULT CONFIGURATION
CFG = get_config()
DEFAULT_MODEL_NAME = "gradient_boosting"

# AVAILABLE FILES AND EXPERIMENTS
def available_files() -> List[Dict[str, str]]:
    """Return the list of available enriched Parquet files."""
    if not ENRICHED_DIR.exists():
        return []
    options = []
    for path in sorted(ENRICHED_DIR.glob("enriched_*.parquet")):
        if any(path.stem.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        options.append({"label": path.name, "value": str(path)})
    return options

def experiment_options(model_name: str = DEFAULT_MODEL_NAME) -> List[Dict[str, str]]:
    """Return experiments that contain the specified trained model."""
    if not EXPERIMENTS_DIR.exists():
        return []
    dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime)
    options: List[Dict[str, str]] = []
    for d in dirs:
        if (d / f"{model_name}_best.joblib").exists():
            options.append({"label": d.name, "value": str(d)})
    return options

# CACHED DATA LOADING
@lru_cache(maxsize=32)
def load_dataset(path_str: str) -> pd.DataFrame:
    """Load a dataset with basic caching and normalized column names."""
    df = load_data(path_str)
    if df is None:
        return pd.DataFrame()
    return df.rename(columns=COL_RENAME)

@lru_cache(maxsize=8)
def load_pipeline(experiment_path: str, model_name: str):
    """Load the trained pipeline and feature list from the experiment directory."""
    exp_dir = Path(experiment_path)
    model_path = exp_dir / f"{model_name}_best.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    pipeline = joblib.load(model_path)
    feature_cols_path = exp_dir / "feature_columns.json"
    feature_columns = json.loads(feature_cols_path.read_text()) if feature_cols_path.exists() else None
    return pipeline, feature_columns

# PREPARE FEATURE MATRIX
def prepare_feature_matrix(df: pd.DataFrame, feature_columns: Optional[List[str]]) -> pd.DataFrame:
    """Select the columns in the order expected by the pipeline (missing columns produce NaN)."""
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in {"file", "source_file", "start_s", "duration", "n_samples"}]
    return df.reindex(columns=feature_columns)

# COMPUTE WINDOW-LEVEL PREDICTIONS
def compute_window_predictions(
    session_path: str,
    experiment_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    window: float = CFG.windows.size_seconds,
    overlap: float = CFG.windows.overlap_ratio,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Extract window-level features from the selected session and apply the trained model."""
    feats = extract_features_from_file(
        session_path,
        window=window,
        overlap=overlap,
        file_id=Path(session_path).name.replace("enriched_", "clean_", 1),
    )
    if not feats:
        raise RuntimeError("Could not generate windows for this session.")
    df_windows = pd.DataFrame(feats).sort_values("start_s").reset_index(drop=True)
    pipeline, feature_columns = load_pipeline(experiment_path, model_name)
    X = prepare_feature_matrix(df_windows, feature_columns)
    df_windows["fatigue_pred"] = pipeline.predict(X)

    metrics: Dict[str, float] = {}
    if "fatigue_score" in df_windows.columns and df_windows["fatigue_score"].notna().any():
        y_true = df_windows["fatigue_score"].to_numpy()
        y_pred = df_windows["fatigue_pred"].to_numpy()
        metrics = {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
            "R2": float(r2_score(y_true, y_pred)),
        }
    return df_windows, metrics

# RELATIVE TIME BOUNDS AND SESSION METADATA
def relative_time_bounds(df: pd.DataFrame) -> Tuple[float, float]:
    """Return min/max interval for ``relative_time``."""
    if "relative_time" not in df.columns:
        return 0.0, 0.0
    return float(df["relative_time"].min()), float(df["relative_time"].max())

def session_metadata(df: pd.DataFrame, source_path: str) -> Dict[str, str]:
    """Return metadata summary for the selected session."""
    duration = df["relative_time"].max() - df["relative_time"].min() if "relative_time" in df.columns else 0
    info = {
        "archivo": Path(source_path).name,
        "filas": f"{len(df):,}",
        "duración": f"{duration:.1f} s" if duration else "N/A",
    }
    for col in ("runner_id", "session_id", "reported_rpe", "age", "sex"):
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            if unique_vals.size == 1:
                info[col] = str(unique_vals[0])
            elif unique_vals.size > 1:
                info[col] = f"Mixed ({unique_vals.size})"
    for col in ("fatigue_score", "fatigue_level"):
        if col in df.columns and df[col].notna().any():
            val = df[col].dropna().iloc[0]
            info[col] = f"{val:.3f}" if pd.api.types.is_numeric_dtype(df[col]) else str(val)
    return info

__all__ = [
    "available_files",
    "experiment_options",
    "load_dataset",
    "load_pipeline",
    "prepare_feature_matrix",
    "compute_window_predictions",
    "relative_time_bounds",
    "session_metadata",
    "DEFAULT_MODEL_NAME",
]
