"""
Métricas biomecánicas y fisiológicas (etapa 3/5 del pipeline):
- Filtro pasa-alto + integración de aceleraciones centradas → velocidad traslacional (vtr).
- Cálculo de jerk (derivada de la aceleración) y estimación del fatigue_score.
- Actualiza las sesiones enriquecidas para su uso en etapas posteriores.

Ejemplo:
    python src/data/metrics.py --input-dir data/enriched

Entrada:  data/enriched/enriched_*.parquet
Salida:   enriched actualizados
Siguiente: python src/features/features_extraction.py
"""

# LIBRERÍAS ESTÁNDAR
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np # type: ignore
import pandas as pd # type: ignore

# AJUSTE DE RUTA DEL PROYECTO
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# LIBRERÍAS DEL PROYECTO
from src.config import get_config
from src.utils.kinematics_utils import (
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

# CONFIGURACIÓN DE LOGGING
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

CFG = get_config()
SCORE_SMOOTHING = getattr(CFG.fatigue_weights, "smoothing_window", 0)

# EXCEPCIONES Y DATACLASS
class MetricsError(Exception):
    """Excepción base del pipeline de métricas."""

@dataclass
class SessionResult:
    """Resumen de una sesión procesada."""
    file: str
    fatigue_score: float
    metrics: Dict[str, float]

# UTILIDADES
def list_session_files(source_dir: str, files: Optional[Sequence[str]] = None) -> List[str]:
    """Lista los archivos de sesión a procesar."""
    directory = os.path.abspath(source_dir)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"No se encontró el directorio: {directory}")

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

# PROCESAMIENTO DE SESIONES
def _load_session(path: str) -> pd.DataFrame:
    """Carga y valida un archivo de sesión."""
    df = pd.read_parquet(path)
    schema_name = "enriched" if os.path.basename(path).startswith("enriched_") else "processed"
    validate_dataframe(df, schema_name)
    return df

def _apply_biomechanics(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica cálculos biomecánicos al dataframe."""
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
    """Guarda el dataframe enriquecido en el archivo especificado."""
    try:
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_parquet(path, index=False)
        logger.info("Archivo enriched actualizado: %s", os.path.basename(path))
        return True
    except Exception as exc:
        logger.error(
            "Error al guardar archivo enriched %s: %s",
            os.path.basename(path),
            exc,
            exc_info=True,
        )
        return False

# FUNCIONES PRINCIPALES
def process_session(path: str, *, allow_save: bool) -> Optional[SessionResult]:
    """Procesa una sesión individual y calcula sus métricas."""
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
        logger.error("Error al procesar %s: %s", os.path.basename(path), exc, exc_info=True)
        return None

# PROCESAMIENTO GLOBAL
def process_all_sessions(
    *,
    input_dir: Optional[str] = None,
    save_enriched: bool = True,
    files: Optional[Sequence[str]] = None,
) -> Optional[int]:
    """Procesa todas las sesiones en el directorio especificado."""
    source_dir = input_dir or (
        DEFAULT_ENRICHED_DIR if os.path.isdir(DEFAULT_ENRICHED_DIR) else DEFAULT_PROCESSED_DIR
    )

    session_files = list_session_files(source_dir, files)
    if not session_files:
        logger.warning("No se encontraron archivos en %s", source_dir)
        return None

    processed = 0
    for path in session_files:
        allow_save = save_enriched and os.path.basename(path).startswith("enriched_")
        result = process_session(path, allow_save=allow_save)
        if result:
            processed += 1

    return processed

# INTERFAZ DE LÍNEA DE COMANDOS
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula métricas biomecánicas y fatigue_score."
    )
    parser.add_argument(
        "--input-dir",
        help="Directorio con enriched_*.parquet (por defecto: data/enriched).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No sobrescribir archivos enriched con nuevos scores.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Lista opcional de archivos a procesar (rutas o nombres).",
    )
    return parser.parse_args()

# FUNCIÓN MAIN
def main() -> None:
    args = parse_args()
    total = process_all_sessions(
        input_dir=args.input_dir,
        save_enriched=not args.no_save,
        files=args.files,
    )

    if total is None:
        logger.warning("No se generaron métricas.")
    else:
        logger.info("Sesiones procesadas: %d", total)

if __name__ == "__main__":
    main()
