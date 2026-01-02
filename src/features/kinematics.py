"""
Generación de características cinemáticas (etapa 2/5 del pipeline):
- Centra las aceleraciones por eje y calcula magnitudes cruda y dinámica.
- Genera ficheros enriched consumidos por metrics y features_extraction.

Ejemplo:
    python src/features/kinematics.py --input-dir data/processed --output-dir data/enriched

Entrada:  data/processed/clean_*.parquet
Salida:   data/enriched/enriched_*.parquet
Siguiente: python src/data/metrics.py
"""

# LIBRERÍAS ESTÁNDAR
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd # type: ignore

# CONFIGURACIÓN DEL PATH DEL PROYECTO
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# LIBRERÍAS DEL PROYECTO
from src.utils.kinematics_utils import centre_accelerations, compute_acceleration_magnitudes
from src.utils.schemas import validate_dataframe

# CONFIGURACIÓN DE LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# CONFIGURACIÓN DE RUTAS
BASE_DIR = PROJECT_ROOT
DEFAULT_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DEFAULT_ENRICHED_DIR = os.path.join(BASE_DIR, "data", "enriched")

# DEFINICIÓN DE EXCEPCIONES
class KinematicsError(Exception):
    """Se lanza cuando falla una transformación cinemática."""

# DEFINICIONES DE DATACLASS
@dataclass
class KinematicsStats:
    """Estadísticas del procesamiento cinemático de un archivo."""
    file: str
    columns_before: int
    columns_after: int
    output_path: str

# FUNCIONES PRINCIPALES
def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula y añade columnas de características cinemáticas al DataFrame."""
    centre_accelerations(df)
    compute_acceleration_magnitudes(df)
    required = [
        "acc_x_centered",
        "acc_y_centered",
        "acc_z_centered",
        "acc_mag",
        "acc_dyn_mag",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KinematicsError(f"Faltan columnas cinemáticas esperadas: {missing}")
    return df

def _load_processed_session(path: str) -> pd.DataFrame:
    """Carga un archivo procesado y valida su esquema."""
    df = pd.read_parquet(path)
    validate_dataframe(df, "processed")
    return df

# GUARDAR SESION ENRICHED
def _write_enriched_session(
    df: pd.DataFrame,
    source_path: str,
    *,
    output_dir: str,
) -> str:
    """Guarda el DataFrame enriquecido en el directorio de salida."""
    validate_dataframe(df, "enriched")
    base = os.path.basename(source_path).replace("clean_", "enriched_")
    output_path = os.path.join(output_dir, base)
    df.to_parquet(output_path, index=False)
    return output_path

# PROCESAR UN SOLO ARCHIVO
def process_single_file(
    path: str,
    *,
    output_dir: str,
) -> Optional[KinematicsStats]:
    """Procesa un solo archivo para calcular características cinemáticas."""
    try:
        df = _load_processed_session(path)
        before_cols = len(df.columns)
        df = compute_kinematics(df)
        output_path = _write_enriched_session(df, path, output_dir=output_dir)
        stats = KinematicsStats(
            file=os.path.basename(path),
            columns_before=before_cols,
            columns_after=len(df.columns),
            output_path=output_path,
        )
        logger.info(
            "Características cinemáticas guardadas en %s (columnas: %d → %d)",
            os.path.basename(output_path),
            stats.columns_before,
            stats.columns_after,
        )
        return stats
    except Exception as exc:
        logger.error("No se pudo procesar %s: %s", os.path.basename(path), exc, exc_info=True)
        return None

# LISTAR ARCHIVOS PROCESADOS
def list_processed_files(source_dir: str) -> List[str]:
    """Lista todos los archivos procesados en el directorio dado."""
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"No se encontró el directorio de procesados: {source_dir}")
    files = sorted(
        os.path.join(source_dir, fname)
        for fname in os.listdir(source_dir)
        if fname.endswith(".parquet") and fname.startswith("clean_")
    )
    return files

# PROCESAR TODOS LOS ARCHIVOS
def process_all_kinematics(
    *,
    processed_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    files: Optional[Sequence[str]] = None,
) -> Optional[int]:
    """Procesa todos los archivos para calcular características cinemáticas."""
    src_dir = processed_dir or DEFAULT_PROCESSED_DIR
    dst_dir = output_dir or DEFAULT_ENRICHED_DIR

    if files is None:
        try:
            files = list_processed_files(src_dir)
        except FileNotFoundError as exc:
            logger.error(exc)
            return None

    if not files:
        logger.warning("No se encontraron archivos para procesar en %s.", src_dir)
        return 0

    os.makedirs(dst_dir, exist_ok=True)

    processed = 0
    processed_files: List[str] = []
    failed_files: List[str] = []
    for path in files:
        stats = process_single_file(path, output_dir=dst_dir)
        if stats:
            processed += 1
            processed_files.append(stats.output_path)
        else:
            failed_files.append(path)

    metadata = {
        "processed_files": processed_files,
        "failed_files": failed_files,
        "total_processed": processed,
        "total_failed": len(failed_files),
        "date": str(pd.Timestamp.now()),
    }
    metadata_path = os.path.join(dst_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Metadatos guardados en: %s", metadata_path)

    return processed

# PARSEO DE ARGUMENTOS
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera aceleraciones centradas y magnitudes derivadas.")
    parser.add_argument("--input-dir", help="Directorio con clean_*.parquet (por defecto data/processed).")
    parser.add_argument("--output-dir", help="Directorio destino para enriched (por defecto data/enriched).")
    parser.add_argument("--files", nargs="*", help="Lista opcional de archivos a procesar (ruta o nombre).")
    return parser.parse_args()

# FUNCIÓN PRINCIPAL
def main() -> None:
    args = parse_args()
    file_list: Optional[List[str]] = None
    if args.files:
        base_dir = args.input_dir or DEFAULT_PROCESSED_DIR
        file_list = []
        for item in args.files:
            path = item if item.endswith(".parquet") else os.path.join(base_dir, item)
            file_list.append(path)

    total = process_all_kinematics(
        processed_dir=args.input_dir,
        output_dir=args.output_dir,
        files=file_list,
    )
    if total:
        logger.info("Generación de características cinemáticas completada. Archivos procesados: %d", total)
    elif total == 0:
        logger.warning("No se procesó ningún archivo.")

if __name__ == "__main__":
    main()
