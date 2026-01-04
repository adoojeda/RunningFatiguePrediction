"""
Utilidad de inferencia casi en tiempo real.

Carga un pipeline entrenado (p. ej., gradient boosting) y lo aplica sobre una sesión
enriquecida. Se reutiliza la extracción de ventanas para alinear las predicciones
con el pipeline de entrenamiento y permitir una reproducción ventana a ventana,
emulando una monitorización en línea.

Ejemplo de uso:
    python src/models/run_inference.py \
        --enriched data/enriched/enriched_<archivo>.parquet \
        --experiment data/results/modeling/experiments/runner_id_YYYYMMDD_HHMMSS \
        --model gradient_boosting \
        --output data/results/modeling/inference/demo_predictions.parquet
"""

# IMPORTACIONES
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore

# AJUSTE DE RUTAS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# IMPORTACIONES LOCALES
from src.config import get_config
from src.features.features_extraction import extract_features_from_file

try:
    from src.models.run_experiments import FEATURE_WHITELIST
except Exception:
    FEATURE_WHITELIST = None

# CONFIGURACIÓN DE LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# CONSTANTES
CFG = get_config()
DEFAULT_MODEL = "gradient_boosting"
DEFAULT_EXPERIMENT = PROJECT_ROOT / "data" / "results" / "modeling" / "experiments"
META_COLS = {
    "file",
    "source_file",
    "runner_id",
    "session_id",
    "age",
    "sex",
    "start_s",
    "duration",
    "n_samples",
}
TARGET_COLS = {"reported_rpe", "fatigue_level", "physical_fatigue_index"}

# FUNCIONES AUXILIARES
def _load_pipeline(experiment_dir: Path, model_name: str):
    """Carga el pipeline entrenado y la lista de variables desde el experimento."""
    model_path = experiment_dir / f"{model_name}_best.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

    pipeline = joblib.load(model_path)
    features_path = experiment_dir / "feature_columns.json"
    if features_path.exists():
        feature_columns = json.loads(features_path.read_text())
    else:
        feature_columns = None
        logger.warning("No se encontró feature_columns.json; se inferirán columnas desde el dataframe.")

    return pipeline, feature_columns

def _prepare_feature_matrix(df: pd.DataFrame, feature_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, List[str]]:
    """Prepara la matriz de características para inferencia, validando columnas."""
    if feature_columns is None:
        if FEATURE_WHITELIST:
            feature_columns = [c for c in FEATURE_WHITELIST if c in df.columns]
            logger.info("Lista de variables inferida desde whitelist (%d columnas).", len(feature_columns))
        else:
            excluded = META_COLS | TARGET_COLS
            feature_columns = [c for c in df.columns if c not in excluded]
            logger.info("Lista de variables inferida desde el dataframe (%d columnas).", len(feature_columns))

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        logger.warning("Faltan %d columnas en los datos de entrada: %s", len(missing), missing)
        feature_columns = [col for col in feature_columns if col in df.columns]
        if not feature_columns:
            raise RuntimeError("No quedan columnas válidas para inferencia tras eliminar las faltantes.")
    X = df.reindex(columns=feature_columns)
    return X, feature_columns

def _infer_single_session(
    enriched_path: Path,
    window: float,
    overlap: float,
) -> pd.DataFrame:
    """Extrae ventanas de una sesión enriquecida y devuelve un DataFrame ordenado."""
    feats = extract_features_from_file(
        str(enriched_path),
        window=window,
        overlap=overlap,
        file_id=enriched_path.name.replace("enriched_", "clean_", 1),
    )
    if not feats:
        raise RuntimeError(f"No se extrajeron ventanas desde {enriched_path}.")
    df = pd.DataFrame(feats)
    df.sort_values("start_s", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def _simulate_stream(pred_df: pd.DataFrame, playback_speed: float) -> None:
    """Simula la reproducción de predicciones en línea según la velocidad indicada."""
    last_start = None
    for row in pred_df.itertuples(index=False):
        msg = (
            f"[t={row.start_s:6.2f}s] pred={row.fatigue_pred:.3f}"
            + (f" | score={row.physical_fatigue_index:.3f}" if not np.isnan(getattr(row, "physical_fatigue_index", np.nan)) else "")
        )
        logger.info(msg)

        if playback_speed > 0 and last_start is not None:
            delta = max(0.0, row.start_s - last_start)
            if delta > 0:
                time.sleep(delta / playback_speed)
        last_start = row.start_s

def _summarize(pred_df: pd.DataFrame) -> Dict[str, float]:
    """Calcula métricas de evaluación si hay referencia disponible."""
    if "physical_fatigue_index" not in pred_df.columns or pred_df["physical_fatigue_index"].isna().all():
        return {}

    y_true = pred_df["physical_fatigue_index"].to_numpy()
    y_pred = pred_df["fatigue_pred"].to_numpy()
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    return metrics

# FUNCION PRINCIPAL DE INFERENCIA
def run_inference(args: argparse.Namespace) -> Path:
    """Ejecuta la inferencia sobre una sesión enriquecida y reproduce predicciones."""
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
        logger.info("Evaluación (physical_fatigue_index por ventana) -> MAE=%.4f RMSE=%.4f R2=%.4f",
                    metrics["mae"], metrics["rmse"], metrics["r2"])
    else:
        logger.info("Sin referencia de physical_fatigue_index; solo predicciones.")

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

# CONFIGURACIÓN DE ARGPARSE
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica un modelo entrenado a una sesión enriquecida y reproduce predicciones por ventana."
    )
    parser.add_argument(
        "--enriched",
        required=True,
        help="Ruta al archivo de sesión enriquecida (parquet/csv).",
    )
    parser.add_argument(
        "--experiment",
        default=str(DEFAULT_EXPERIMENT),
        help="Directorio que contiene el modelo entrenado y la lista de variables."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Nombre del modelo entrenado a cargar (p. ej., gradient_boosting).",
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
        help="Ruta para guardar predicciones (parquet/csv). Si no se indica, no se guardan.",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=0.0,
        help="Factor de reproducción (0 = instantáneo, 1 = tiempo real, 4 = cuatro veces más rápido).",
    )
    return parser.parse_args()

# PUNTO DE ENTRADA
def main() -> None:
    args = parse_args()
    run_inference(args)

if __name__ == "__main__":
    main()
