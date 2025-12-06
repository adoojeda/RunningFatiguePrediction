"""
Herramienta de inferencia casi en tiempo real.

Carga un pipeline entrenado (por ejemplo, gradient boosting) y lo aplica a una
sesión enriquecida. Reutiliza la extracción de ventanas para que las
predicciones sean coherentes con el entrenamiento y puedan reproducirse
ventana a ventana, emulando monitorización online.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.features.features_extraction import extract_features_from_file

# CONFIGURACIÓN DEL LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CFG = get_config()
DEFAULT_MODEL = "gradient_boosting"
DEFAULT_EXPERIMENT = PROJECT_ROOT / "data" / "results" / "modeling" / "experiments"

# FUNCIONES PRINCIPALES
def _load_pipeline(experiment_dir: Path, model_name: str):
    """Carga el pipeline entrenado y la lista de características."""
    model_path = experiment_dir / f"{model_name}_best.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

    pipeline = joblib.load(model_path)
    features_path = experiment_dir / "feature_columns.json"
    if features_path.exists():
        feature_columns = json.loads(features_path.read_text())
    else:
        feature_columns = None
        logger.warning("No se encontró feature_columns.json; se usarán las columnas del dataframe.")

    return pipeline, feature_columns

def _prepare_feature_matrix(df: pd.DataFrame, feature_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, List[str]]:
    """Selecciona y ordena las columnas que espera el pipeline entrenado."""
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in {"file", "source_file", "start_s", "duration", "n_samples"}]
        logger.info("Lista de características inferida con %d columnas.", len(feature_columns))

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        logger.warning("Faltan %d columnas en los datos de entrada: %s", len(missing), missing)
    X = df.reindex(columns=feature_columns)
    return X, feature_columns

def _infer_single_session(
    enriched_path: Path,
    window: float,
    overlap: float,
) -> pd.DataFrame:
    """Genera las features por ventana a partir de una sesión enriquecida."""
    feats = extract_features_from_file(
        str(enriched_path),
        window=window,
        overlap=overlap,
        file_id=enriched_path.name.replace("enriched_", "clean_", 1),
    )
    if not feats:
        raise RuntimeError(f"No se extrajeron ventanas de {enriched_path}.")
    df = pd.DataFrame(feats)
    df.sort_values("start_s", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def _simulate_stream(pred_df: pd.DataFrame, playback_speed: float) -> None:
    """
    Itera sobre las predicciones emulando actualizaciones casi en tiempo real.

    playback_speed = 0   -> sin esperas (instantáneo)
    playback_speed = 1.0 -> duración real
    playback_speed = 4.0 -> 4 veces más rápido
    """
    last_start = None
    for row in pred_df.itertuples(index=False):
        msg = (
            f"[t={row.start_s:6.2f}s] pred={row.fatigue_pred:.3f}"
            + (f" | score={row.fatigue_score:.3f}" if not np.isnan(getattr(row, "fatigue_score", np.nan)) else "")
        )
        logger.info(msg)

        if playback_speed > 0 and last_start is not None:
            delta = max(0.0, row.start_s - last_start)
            if delta > 0:
                time.sleep(delta / playback_speed)
        last_start = row.start_s

def _summarize(pred_df: pd.DataFrame) -> Dict[str, float]:
    """Calcula MAE/RMSE/R2 si existe un fatigue_score de referencia."""
    if "fatigue_score" not in pred_df.columns or pred_df["fatigue_score"].isna().all():
        return {}

    y_true = pred_df["fatigue_score"].to_numpy()
    y_pred = pred_df["fatigue_pred"].to_numpy()
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    return metrics

# FLUJO CLI
def run_inference(args: argparse.Namespace) -> Path:
    enriched_path = Path(args.enriched).resolve()
    experiment_dir = Path(args.experiment).resolve()
    if not enriched_path.exists():
        raise FileNotFoundError(f"No se encontró la sesión: {enriched_path}")
    if not experiment_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio del experimento: {experiment_dir}")

    pipeline, feature_columns = _load_pipeline(experiment_dir, args.model)
    df_windows = _infer_single_session(enriched_path, args.window, args.overlap)
    X, feature_columns = _prepare_feature_matrix(df_windows, feature_columns)

    df_windows["fatigue_pred"] = pipeline.predict(X)

    metrics = _summarize(df_windows)
    if metrics:
        logger.info("Evaluación (fatigue_score por ventana) -> MAE=%.4f RMSE=%.4f R2=%.4f", metrics["mae"], metrics["rmse"], metrics["r2"])
    else:
        logger.info("No hay fatigue_score real; sólo se reportan predicciones.")

    if args.playback_speed is not None and args.playback_speed >= 0:
        _simulate_stream(df_windows, args.playback_speed)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".csv":
            df_windows.to_csv(out_path, index=False)
        else:
            df_windows.to_parquet(out_path, index=False)
        logger.info("Predicciones guardadas en %s", out_path)
        return out_path

    return Path()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica un modelo entrenado a una sesión enriquecida y reproduce las predicciones ventana a ventana."
    )
    parser.add_argument(
        "--enriched",
        required=True,
        help="Ruta a un archivo enriched_*.parquet.",
    )
    parser.add_argument(
        "--experiment",
        default=str(DEFAULT_EXPERIMENT),
        help="Directorio del experimento que contiene {model}_best.joblib y feature_columns.json.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Nombre del modelo a cargar (prefijo del archivo *_best.joblib).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=CFG.windows.size_seconds,
        help="Duración de la ventana en segundos.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=CFG.windows.overlap_ratio,
        help="Solape entre ventanas (0-1).",
    )
    parser.add_argument(
        "--output",
        help="Archivo opcional donde guardar las predicciones (parquet/csv).",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=0.0,
        help="Factor de reproducción temporal (0 = sin espera, 1 = tiempo real, 4 = 4x más rápido).",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    run_inference(args)

if __name__ == "__main__":
    main()
