"""
Métricas biomecánicas/fisiológicas (etapa 3/5 del pipeline):
- Filtro pasa-alto + integración de aceleraciones centradas → velocidad de traslación (Vtr).
- Cálculo del jerk (derivada de la aceleración) y del índice de fatiga.
- Persistencia de las sesiones enriquecidas y de la tabla consolidada de métricas.

Entrada: `data/enriched/enriched_*.parquet`
Salidas: ficheros enriquecidos actualizados + `data/results/all_sessions_metrics.parquet`
Siguiente etapa: `python src/features/features_extraction.py`
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import get_config
from src.utils.kinematics import (
    DEFAULT_FS,
    DEFAULT_HP_CUTOFF,
    centre_accelerations,
    compute_acceleration_magnitudes,
    compute_translational_velocity,
    compute_jerk,
)
from src.utils.metrics_utils import (
    compute_session_metrics,
    derive_fatigue_references,
    compute_fatigue_score,
)
from src.utils.schemas import validate_dataframe

# CONFIGURACIÓN DEL LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# RUTAS Y CONFIGURACIÓN
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
DEFAULT_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DEFAULT_RESULTS_DIR = os.path.join(DATA_DIR, "results")
DEFAULT_OUTPUT_PARQUET = os.path.join(DEFAULT_RESULTS_DIR, "all_sessions_metrics.parquet")

CFG = get_config()
SCORE_SMOOTHING = getattr(CFG.fatigue_weights, "smoothing_window", 0)

# CLASES DE EXCEPCIÓN Y DATACLASSES
class MetricsError(Exception):
    """Excepción base del pipeline de métricas."""

@dataclass
class SessionResult:
    """Resumen del resultado de procesamiento de una sesión."""
    file: str
    fatigue_score: float
    metrics: Dict[str, float]

# FUNCIONES AUXILIARES
def list_session_files(source_dir: str, files: Optional[Sequence[str]] = None) -> List[str]:
    """Devuelve la lista ordenada de sesiones a procesar."""
    directory = os.path.abspath(source_dir)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directorio de origen no encontrado: {directory}")

    if files:
        resolved = []
        for item in files:
            path = item if os.path.isabs(item) else os.path.join(directory, item)
            if not path.endswith(".parquet"):
                path = f"{path}.parquet"
            resolved.append(path)
        return resolved

    enriched = sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("enriched_") and f.endswith(".parquet"))
    if enriched:
        return enriched
    processed = sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("clean_") and f.endswith(".parquet"))
    return processed

def save_metrics_table(results: List[Dict[str, float]], output_path: str) -> Optional[pd.DataFrame]:
    """Guarda la tabla agregada de métricas."""
    if not results:
        return None
    df_all = pd.DataFrame(results)
    output_path = output_path or DEFAULT_OUTPUT_PARQUET
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_all.to_parquet(output_path, index=False)
    logger.info("Métricas globales guardadas en: %s (%d sesiones)", output_path, len(df_all))
    return df_all

# GESTIÓN DE DATAFRAMES
def _load_session(path: str) -> pd.DataFrame:
    """Carga un parquet validando el esquema correspondiente."""
    df = pd.read_parquet(path)
    schema_name = "enriched" if os.path.basename(path).startswith("enriched_") else "processed"
    validate_dataframe(df, schema_name)
    return df

def _apply_biomechanics(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta los enriquecimientos biomecánicos previos al scoring."""
    if "relative_time" in df.columns:
        df = (
            df.sort_values("relative_time")
            .drop_duplicates(subset="relative_time", keep="first")
            .reset_index(drop=True)
        )
    df = centre_accelerations(df)
    df = compute_acceleration_magnitudes(df)
    df = compute_translational_velocity(
        df,
        default_fs=DEFAULT_FS,
        cutoff=DEFAULT_HP_CUTOFF,
    )
    df = compute_jerk(df)
    return df

def _save_enriched(df: pd.DataFrame, path: str) -> bool:
    """Guarda el dataframe enriquecido en disco."""
    try:
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_parquet(path, index=False)
        logger.info("Fichero enriquecido actualizado: %s", os.path.basename(path))
        return True
    except Exception as exc:
        logger.error(
            "Error al guardar el fichero enriquecido %s: %s",
            os.path.basename(path),
            exc,
            exc_info=True,
        )
        return False

# PROCESAMIENTO GLOBAL
def process_session(path: str, *, allow_save: bool) -> Optional[SessionResult]:
    """Procesa un fichero de sesión y devuelve las métricas calculadas."""
    try:
        df = _load_session(path)
        df = _apply_biomechanics(df)

        session_metrics = compute_session_metrics(df)
        references = derive_fatigue_references(df)
        session_metrics = compute_fatigue_score(
            session_metrics,
            context="session",
            references=references,
        )
        if allow_save:
            _save_enriched(df, path)

        fatigue_score = session_metrics.get("fatigue_score", np.nan)
        logger.info(
            "Sesión %s -> fatigue_score=%.3f",
            os.path.basename(path),
            fatigue_score,
        )
        return SessionResult(
            file=os.path.basename(path),
            fatigue_score=float(fatigue_score) if np.isfinite(fatigue_score) else np.nan,
            metrics=session_metrics,
        )

    except Exception as exc:
        logger.error("Error procesando %s: %s", os.path.basename(path), exc, exc_info=True)
        return None

def process_all_sessions(
    *,
    input_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    save_enriched: bool = True,
    files: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
    """Procesa todas las sesiones y genera el resumen global."""
    source_dir = input_dir or (
        DEFAULT_ENRICHED_DIR if os.path.isdir(DEFAULT_ENRICHED_DIR) else DEFAULT_PROCESSED_DIR
    )

    session_files = list_session_files(source_dir, files)
    if not session_files:
        logger.warning("No se encontraron ficheros en %s", source_dir)
        return None

    results: List[Dict[str, float]] = []
    for path in session_files:
        allow_save = save_enriched and os.path.basename(path).startswith("enriched_")
        result = process_session(path, allow_save=allow_save)
        if result:
            results.append(
                {
                    "session_file": result.file,
                    "fatigue_score": result.fatigue_score,
                }
            )

    return save_metrics_table(results, output_path or DEFAULT_OUTPUT_PARQUET)

# INTERFAZ CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula métricas biomecánicas y puntuaciones de fatiga."
    )
    parser.add_argument(
        "--input-dir",
        help="Directorio con ficheros enriched_*.parquet (por defecto: data/enriched).",
    )
    parser.add_argument(
        "--output-dir",
        help="Directorio donde se guardará el parquet de métricas (por defecto: data/results).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No sobrescribir los ficheros enriquecidos con las nuevas puntuaciones.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Lista opcional de ficheros específicos (rutas o nombres) para procesar.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_RESULTS_DIR
    output_path = os.path.join(output_dir, "all_sessions_metrics.parquet")

    df_metrics = process_all_sessions(
        input_dir=args.input_dir,
        output_path=output_path,
        save_enriched=not args.no_save,
        files=args.files,
    )

    if df_metrics is None:
        logger.warning("No se generaron métricas.")

if __name__ == "__main__":
    main()
