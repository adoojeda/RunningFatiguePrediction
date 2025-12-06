"""
Extracción de características con ventanas deslizantes (etapa 4/5 del pipeline).

- Genera ventanas solapadas (por defecto 3 s y 50 % de solape) sobre las sesiones enriquecidas.
- Calcula estadísticas robustas de aceleración, velocidad de traslación, jerk, FC, SpO₂ e índice de fatiga.
- Registra métricas de calidad por ventana (número de muestras, proporción de NaN, duración).
- Cruza las características con el mapeo de RPE (metadata de corredor y sesión).
- Guarda el dataset consolidado bajo `data/results/` (configurable por CLI).

Entrada: `data/enriched/enriched_*.parquet` + `data/raw/rpe_file_mapping.csv`
Salida: `data/results/features_dataset.parquet`
Siguiente etapa: scripts de análisis en `src/analysis/`.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import get_config
from src.utils.metrics_utils import compute_fatigue_score, derive_fatigue_references
from src.utils.schemas import validate_dataframe
from src.utils.window_stats import mad, skewness, kurtosis, safe_stats
from src.utils.windowing import WindowParams, create_window_params, iter_windows, prepare_dataframe

# CONFIGURACIÓN DEL LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# RUTAS Y CONFIGURACIÓN
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MAPPING_PATH = os.path.join(RAW_DIR, "rpe_file_mapping.csv")
DEFAULT_OUTPUT = os.path.join(RESULTS_DIR, "features_dataset.parquet")
os.makedirs(RESULTS_DIR, exist_ok=True)

CFG = get_config()

NUMERIC_COLS = [
    "acc_x_centered",
    "acc_y_centered",
    "acc_z_centered",
    "acc_mag",
    "vtr",
    "jerk_mag",
    "hr",
    "spo2",
    "grav_x",
    "grav_y",
    "grav_z",
    "roll",
    "pitch",
    "yaw",
]

# CÁLCULO DE CARACTERÍSTICAS POR VENTANA
def compute_window_features(
    df_win: pd.DataFrame,
    file_id: str,
    source_file: str,
    fatigue_refs: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Calcula estadísticas para una ventana ya segmentada (df_win).
    Devuelve un diccionario con características y metadatos.
    """
    out: Dict[str, float] = {}

    # Metadatos de la ventana
    t0 = float(df_win["relative_time"].min())
    t1 = float(df_win["relative_time"].max())
    duration = t1 - t0 if np.isfinite(t1) else np.nan

    out["file"] = file_id
    out["source_file"] = source_file
    out["start_s"] = t0
    out["duration"] = max(duration, 0.0) if np.isfinite(duration) else np.nan
    out["n_samples"] = int(len(df_win))

    # Aceleraciones centradas
    for axis in ["x", "y", "z"]:
        col = f"acc_{axis}_centered"
        if col in df_win.columns:
            x = df_win[col].to_numpy(dtype=float)
            out[f"{col}_mean"] = np.nanmean(x)
            out[f"{col}_std"] = np.nanstd(x, ddof=1)
            out[f"{col}_mad"] = mad(x)
            out[f"{col}_skew"] = skewness(x)
            out[f"{col}_kurt"] = kurtosis(x)

    # Magnitud de la aceleración bruta
    if "acc_mag" in df_win.columns:
        x = df_win["acc_mag"].to_numpy(dtype=float)
        out["acc_mean"] = np.nanmean(x)
        out["acc_std"] = np.nanstd(x, ddof=1)
        out["acc_mag_mad"] = mad(x)
        out["acc_mag_skew"] = skewness(x)
        out["acc_mag_kurt"] = kurtosis(x)

    # Magnitud de la velocidad de traslación
    if "vtr" in df_win.columns:
        v = df_win["vtr"].to_numpy(dtype=float)
        out["vtr_mean"] = np.nanmean(v)
        out["vtr_std"] = np.nanstd(v, ddof=1)
        out["vtr_mad"] = mad(v)
        out["vtr_skew"] = skewness(v)
        out["vtr_kurt"] = kurtosis(v)

    # Señales de orientación / balance
    for ori_col in ["roll", "yaw", "grav_x", "grav_y", "grav_z"]:
        if ori_col in df_win.columns:
            val = df_win[ori_col].to_numpy(dtype=float)
            out[f"{ori_col}_mean"] = np.nanmean(val)
            out[f"{ori_col}_std"] = np.nanstd(val, ddof=1)
            out[f"{ori_col}_mad"] = mad(val)
            out[f"{ori_col}_skew"] = skewness(val)
            out[f"{ori_col}_kurt"] = kurtosis(val)

    # Magnitud del jerk
    if "jerk_mag" in df_win.columns:
        j = df_win["jerk_mag"].to_numpy(dtype=float)
        out["jerk_mean"] = np.nanmean(j)
        out["jerk_std"] = np.nanstd(j, ddof=1)
        out["jerk_mad"] = mad(j)
        out["jerk_skew"] = skewness(j)

    # Frecuencia cardiaca
    if "hr" in df_win.columns:
        f = df_win["hr"].to_numpy(dtype=float)
        mean, _, _ = safe_stats(f)
        out["hr_mean"] = mean

    # Saturación de oxígeno (SpO₂)
    if "spo2" in df_win.columns:
        s = df_win["spo2"].to_numpy(dtype=float)
        mean, _, _ = safe_stats(s)
        out["spo2_mean"] = mean
    # Calcula el fatigue_score por ventana a partir de las métricas disponibles
    metrics_payload = {}
    hr_mean = out.get("hr_mean")
    if hr_mean is not None and np.isfinite(hr_mean):
        metrics_payload["hr_mean"] = float(hr_mean)
    spo2_mean = out.get("spo2_mean")
    if spo2_mean is not None and np.isfinite(spo2_mean):
        metrics_payload["spo2_mean"] = float(spo2_mean)
    acc_std = out.get("acc_std")
    if acc_std is not None and np.isfinite(acc_std):
        metrics_payload["acc_std"] = float(acc_std)
    jerk_std = out.get("jerk_std")
    if jerk_std is not None and np.isfinite(jerk_std):
        metrics_payload["jerk_std"] = float(jerk_std)

    if metrics_payload:
        score_dict = compute_fatigue_score(
            metrics_payload.copy(),
            context="window",
            references=fatigue_refs,
        )
        fatigue_score = score_dict.get("fatigue_score")
        if fatigue_score is not None and np.isfinite(fatigue_score):
            out["fatigue_score"] = fatigue_score

    return out

# EXTRACCIÓN A NIVEL DE FICHERO
def extract_features_from_file(
    fpath: str,
    window: float,
    overlap: float,
    file_id: Optional[str] = None,
) -> List[Dict]:
    """
    Desliza ventanas sobre un fichero y calcula las características de cada una.
    """
    try:
        df = pd.read_parquet(fpath)
    except Exception as exc:
        logger.error("Error al leer %s: %s", os.path.basename(fpath), exc, exc_info=True)
        return []

    schema_name = "enriched" if os.path.basename(fpath).startswith("enriched_") else "processed"
    try:
        validate_dataframe(df, schema_name)
    except ValueError as exc:
        logger.error("La validación del esquema falló para %s: %s", os.path.basename(fpath), exc)
        return []

    if "relative_time" not in df.columns:
        logger.warning("%s no contiene 'relative_time'; se omite.", os.path.basename(fpath))
        return []

    df = prepare_dataframe(df, NUMERIC_COLS)
    fatigue_refs = derive_fatigue_references(df)
    params = create_window_params(window, overlap, min_samples=CFG.windows.min_samples)

    feats: List[Dict] = []
    source_file = os.path.basename(fpath)
    file_key = file_id or source_file

    try:
        for _, _, df_win in iter_windows(df, params):
            feats.append(
                compute_window_features(
                    df_win,
                    file_key,
                    source_file,
                    fatigue_refs=fatigue_refs,
                )
            )
    except ValueError as exc:
        logger.warning("%s tiene un rango temporal inválido; se omite. Motivo: %s", source_file, exc)

    return feats

# EJECUCIÓN DEL PIPELINE
def load_rpe_mapping(path: str) -> pd.DataFrame:
    """Carga el fichero de mapeo de RPE con validaciones básicas."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró el mapeo RPE en: {path}")

    df_map = pd.read_csv(path)
    expected_cols = {"file", "runner_id", "session_id", "reported_rpe"}
    missing = expected_cols - set(df_map.columns)
    if missing:
        raise ValueError(f"Faltan columnas en rpe_file_mapping.csv: {missing}")
    return df_map

def collect_source_files(source_dir: Optional[str] = None) -> List[str]:
    """Devuelve la lista de parquets a procesar, priorizando data/enriched."""
    if source_dir:
        directories = [source_dir]
    else:
        directories = [
            ENRICHED_DIR if os.path.isdir(ENRICHED_DIR) else None,
            PROCESSED_DIR,
        ]
        directories = [d for d in directories if d and os.path.isdir(d)]

    for directory in directories:
        files = sorted(
            f.path for f in os.scandir(directory)
            if f.is_file() and f.name.endswith(".parquet") and (f.name.startswith("enriched_") or f.name.startswith("clean_"))
        )
        if files:
            logger.info("Se encontraron %d ficheros en %s", len(files), directory)
            return files

    raise FileNotFoundError("No se encontraron parquets en los directorios configurados.")

def run_feature_extraction(
    window: float,
    overlap: float,
    out_path: str,
    source_dir: Optional[str] = None,
) -> str:
    """
    Ejecuta el pipeline completo de extracción de características.
    """
    df_map = load_rpe_mapping(MAPPING_PATH)
    files = collect_source_files(source_dir=source_dir)

    logger.info("Procesando %d ficheros con ventana=%.2fs solape=%.2f", len(files), window, overlap)
    all_feats: List[Dict] = []

    for fpath in files:
        source_file = os.path.basename(fpath)
        mapping_key = source_file.replace("enriched_", "clean_", 1) if source_file.startswith("enriched_") else source_file
        feats = extract_features_from_file(
            fpath,
            window=window,
            overlap=overlap,
            file_id=mapping_key,
        )
        if feats:
            all_feats.extend(feats)
        else:
            logger.warning("No se generaron características para %s", os.path.basename(fpath))

    if not all_feats:
        raise RuntimeError("No se generaron características. Revisa columnas requeridas y rangos temporales.")

    df_feats = pd.DataFrame(all_feats)
    if "jerk_std" in df_feats.columns:
        threshold = df_feats["jerk_std"].quantile(0.995)
        before = len(df_feats)
        df_feats = df_feats[df_feats["jerk_std"] <= threshold]
        removed = before - len(df_feats)
        if removed > 0:
            logger.info("Filtradas %d ventanas con jerk_std > %.2f", removed, threshold)
    df_out = df_feats.merge(df_map, on="file", how="left")

    missing_rpe = df_out["reported_rpe"].isna().sum() if "reported_rpe" in df_out.columns else len(df_out)
    if missing_rpe:
        logger.warning("Falta información de mapeo para %d ventanas; revisa rpe_file_mapping.csv.", missing_rpe)

    if "reported_rpe" in df_out.columns:
        df_out["fatigue_level"] = pd.cut(
            df_out["reported_rpe"],
            bins=[0, 5, 8, 11],
            labels=["low", "medium", "high"],
            include_lowest=True,
            right=False,
        ).astype(str)

    meta_cols = ["file", "source_file", "runner_id", "session_id", "start_s", "duration", "n_samples"]
    label_cols = ["reported_rpe"]
    other_cols = [c for c in df_out.columns if c not in meta_cols + label_cols]
    df_out = df_out[meta_cols + label_cols + other_cols]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if out_path.lower().endswith(".csv"):
        df_out.to_csv(out_path, index=False)
    else:
        df_out.to_parquet(out_path, index=False)

    logger.info("Características guardadas en %s (%d ventanas)", out_path, len(df_out))
    return out_path

# INTERFAZ CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extracción de características con ventanas deslizantes para analizar la fatiga."
    )
    parser.add_argument(
        "--window",
        type=float,
        default=CFG.windows.size_seconds,
        help=f"Duración de la ventana en segundos (por defecto: {CFG.windows.size_seconds}).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Solape entre ventanas en [0,1) (por defecto: 0.5).",
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Ruta de salida del dataset de características.")
    parser.add_argument("--source", type=str, default=None, help="Directorio opcional desde el que leer los parquets de entrada.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if not (0.0 <= args.overlap < 1.0):
        raise ValueError("El parámetro --overlap debe estar en [0, 1). Valor recomendado: 0.5.")

    run_feature_extraction(
        window=args.window,
        overlap=args.overlap,
        out_path=args.output,
        source_dir=args.source,
    )

if __name__ == "__main__":
    main()
