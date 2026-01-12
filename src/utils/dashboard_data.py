"""Funciones de datos compartidas para el dashboard en Dash."""

# LIBRERÍAS
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib # type: ignore
import pandas as pd # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore

# LIBRERÍAS DEL PROYECTO
from src.config import get_config
from src.features.features_extraction import extract_features_from_file
from src.utils.data_loader import load_data

# CONSTANTES DE COLUMNAS
try:
    from src.models.run_experiments import FEATURE_WHITELIST, META_COLS
except Exception:
    FEATURE_WHITELIST = None
    META_COLS = {"file", "source_file", "runner_id", "session_id", "age", "sex", "start_s", "duration", "n_samples"}
TARGET_COLS = {"reported_rpe", "fatigue_level", "physical_fatigue_index"}

# RUTAS DE DIRECTORIOS
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
ENRICHED_DIR = DATA_DIR / "enriched"
EXPERIMENTS_DIR = DATA_DIR / "results" / "modeling" / "experiments"
EXCLUDED_PREFIXES = ("all_sessions_metrics", "features_dataset")

# MAPEO DE NOMBRES DE COLUMNAS
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

# CONFIGURACIÓN Y MODELO POR DEFECTO
CFG = get_config()
DEFAULT_MODEL_NAME = "gradient_boosting"

# OPCIONES DE ARCHIVOS Y MODELOS
def available_files() -> List[Dict[str, str]]:
    """Devuelve archivos enriquecidos disponibles para selección."""
    if not ENRICHED_DIR.exists():
        return []
    options = []
    for path in sorted(ENRICHED_DIR.glob("enriched_*.parquet")):
        if any(path.stem.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        options.append({"label": path.name, "value": str(path)})
    return options

def experiment_options(model_name: str = DEFAULT_MODEL_NAME) -> List[Dict[str, str]]:
    """Devuelve los experimentos disponibles que contienen el modelo especificado."""
    if not EXPERIMENTS_DIR.exists():
        return []
    dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime)
    options: List[Dict[str, str]] = []
    for d in dirs:
        if (d / f"{model_name}_best.joblib").exists():
            options.append({"label": d.name, "value": str(d)})
    return options

def model_options(experiment_path: str) -> List[Dict[str, str]]:
    """Devuelve los modelos disponibles en un experimento dado."""
    exp_dir = Path(experiment_path)
    if not exp_dir.exists():
        return []
    model_files = sorted(exp_dir.glob("*_best.joblib"))
    options = []
    for model_path in model_files:
        model_name = model_path.name.replace("_best.joblib", "")
        label = model_name.replace("_", " ").title()
        options.append({"label": label, "value": model_name})
    return options

# CARGA DE DATASETS Y PIPELINES
def _file_mtime(path_str: str) -> float:
    """Devuelve la fecha de modificación del archivo (0.0 si no existe)."""
    try:
        return Path(path_str).stat().st_mtime
    except FileNotFoundError:
        return 0.0

@lru_cache(maxsize=32)
def load_dataset(path_str: str, mtime: float) -> pd.DataFrame:
    """Carga un dataset y renombra columnas según el mapeo definido."""
    df = load_data(path_str)
    if df is None:
        return pd.DataFrame()
    return df.rename(columns=COL_RENAME)

@lru_cache(maxsize=8)
def load_pipeline(experiment_path: str, model_name: str):
    """Carga el pipeline entrenado y las columnas de features usadas."""
    exp_dir = Path(experiment_path)
    model_path = exp_dir / f"{model_name}_best.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    pipeline = joblib.load(model_path)
    feature_cols_path = exp_dir / "feature_columns.json"
    feature_columns = json.loads(feature_cols_path.read_text()) if feature_cols_path.exists() else None
    return pipeline, feature_columns

# PREPARACIÓN DE MATRIZ DE FEATURES
def prepare_feature_matrix(df: pd.DataFrame, feature_columns: Optional[List[str]]) -> pd.DataFrame:
    """Prepara la matriz de features para predicción."""
    if feature_columns is None:
        if FEATURE_WHITELIST:
            feature_columns = [c for c in FEATURE_WHITELIST if c in df.columns]
        else:
            excluded = set(META_COLS) | TARGET_COLS
            feature_columns = [c for c in df.columns if c not in excluded]
    return df.reindex(columns=feature_columns)

# CÁLCULO DE PREDICCIONES POR VENTANA
def compute_window_predictions(
    session_path: str,
    experiment_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    window: float = CFG.windows.size_seconds,
    overlap: float = CFG.windows.overlap_ratio,
    target_col: str = "physical_fatigue_index",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Calcula predicciones de cansancio por ventana para una sesión dada."""
    feats = extract_features_from_file(
        session_path,
        window=window,
        overlap=overlap,
        file_id=Path(session_path).name.replace("enriched_", "clean_", 1),
    )
    if not feats:
        raise RuntimeError("No se pudieron generar ventanas para esta sesión.")
    df_windows = pd.DataFrame(feats).sort_values("start_s").reset_index(drop=True)
    pipeline, feature_columns = load_pipeline(experiment_path, model_name)
    X = prepare_feature_matrix(df_windows, feature_columns)
    df_windows["fatigue_pred"] = pipeline.predict(X)

    metrics: Dict[str, float] = {}
    if target_col in df_windows.columns and df_windows[target_col].notna().any():
        y_true = df_windows[target_col].to_numpy()
        y_pred = df_windows["fatigue_pred"].to_numpy()
        metrics = {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
            "R2": float(r2_score(y_true, y_pred)),
            "target": target_col,
        }
    return df_windows, metrics

# INTERVALO DE TIEMPO RELATIVO
def relative_time_bounds(df: pd.DataFrame) -> Tuple[float, float]:
    """Devuelve el tiempo relativo mínimo y máximo en el DataFrame."""
    if "relative_time" not in df.columns:
        return 0.0, 0.0
    return float(df["relative_time"].min()), float(df["relative_time"].max())

# EXTRACCIÓN DE METADATOS DE SESIÓN
def session_metadata(df: pd.DataFrame, source_path: str) -> Dict[str, str]:
    """Extrae metadatos clave de la sesión desde el DataFrame."""
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
                info[col] = f"Mixto ({unique_vals.size})"
    for col in ("physical_fatigue_index", "fatigue_level"):
        if col in df.columns and df[col].notna().any():
            val = df[col].dropna().iloc[0]
            info[col] = f"{val:.3f}" if pd.api.types.is_numeric_dtype(df[col]) else str(val)
    return info

__all__ = [
    "available_files",
    "experiment_options",
    "model_options",
    "load_dataset",
    "_file_mtime",
    "load_pipeline",
    "prepare_feature_matrix",
    "compute_window_predictions",
    "relative_time_bounds",
    "session_metadata",
    "DEFAULT_MODEL_NAME",
]
