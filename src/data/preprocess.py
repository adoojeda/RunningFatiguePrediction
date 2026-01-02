"""
Preprocesamiento de señales (etapa 1/5 del pipeline).

- Limpia los CSV brutos y genera archivos Parquet listos para el análisis.
- Filtra valores fisiológicos imposibles e interpola huecos cortos en las señales fisiológicas (hr/spo2).
- Conserva señales de gravedad/rotación/orientación para el modelado biomecánico posterior.
- Elimina únicamente la marca temporal absoluta y crea `relative_time`.
- Guarda un `metadata.json` con el resumen de archivos procesados y fallidos.
- Usa paralelización por defecto si hay varios archivos.

Ejemplo:
    python src/data/preprocess.py --input-dir data/raw --output-dir data/processed

Salida: data/processed/clean_*.parquet
Siguiente paso: python src/features/kinematics.py
"""

# LIBRERÍAS 
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional
import pandas as pd  # type: ignore

# RUTA DEL PROYECTO
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# IMPORTACIONES DEL PROYECTO
from src.config import get_config
from src.utils.kinematics_utils import DEFAULT_FS, estimate_sampling_rate
from src.utils.schemas import validate_dataframe
from src.utils.preprocess_utils import (
    apply_physio_filters,
    derive_relative_time,
    ensure_numeric,
    filter_acc_outliers,
    finalise_dataframe,
    interpolate_channels,
    interp_limit_from_seconds,
    load_raw_file,
    PreprocessError,
    EmptyFileError,
    PreprocessStats,
)

# CONFIGURACIÓN DE LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# DIRECTORIOS POR DEFECTO
DEFAULT_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DEFAULT_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(DEFAULT_PROCESSED_DIR, exist_ok=True)

# CONFIGURACIÓN GLOBAL
CFG = get_config()
PHYSIO_RANGES = CFG.ranges
INTERP_MAX_GAP_SEC = CFG.interpolation.max_gap_seconds
MAX_WORKERS = CFG.workforce.max_workers

# PREPROCESAMIENTO PRINCIPAL
def preprocess_single_file(filepath: str) -> Optional[pd.DataFrame]:
    """Procesa un CSV y devuelve un DataFrame listo para análisis."""
    stats = PreprocessStats()
    try:
        df = load_raw_file(filepath)
        if df.empty:
            raise EmptyFileError("El archivo está vacío.")
        stats.samples_in = len(df)
        ensure_numeric(df, df.columns)
        validate_dataframe(df, "raw")
        df = derive_relative_time(df)

        fs_est = estimate_sampling_rate(df["relative_time"]) or DEFAULT_FS
        interp_limit = interp_limit_from_seconds(fs_est, INTERP_MAX_GAP_SEC)

        apply_physio_filters(df, PHYSIO_RANGES.hr, PHYSIO_RANGES.spo2)
        stats.interpolated_hr, stats.interpolated_spo2 = interpolate_channels(df, interp_limit)
        stats.acc_outliers_removed = filter_acc_outliers(df, PHYSIO_RANGES.acc_max)
        df = finalise_dataframe(df)
        stats.samples_out = len(df)

        logger.info(
            "Preprocesado %s | fs=%.2f Hz | stats=%s",
            os.path.basename(filepath),
            fs_est,
            stats.as_dict(),
        )
        return df

    except PreprocessError as exc:
        logger.error("Error de preprocesamiento en %s: %s", os.path.basename(filepath), exc)
        return None
    except Exception as exc:  
        logger.error(
            "Error inesperado durante el preprocesamiento de %s: %s",
            os.path.basename(filepath),
            exc,
            exc_info=True,
        )
        return None

def process_file(filepath: str, output_dir: str) -> Optional[str]:
    """Procesa un archivo completo y lo guarda en Parquet."""
    try:
        df = preprocess_single_file(filepath)
        if df is None or df.empty:
            logger.warning("El archivo %s no se procesó correctamente.", os.path.basename(filepath))
            return None

        filename = os.path.basename(filepath).replace(".csv", ".parquet")
        output_path = os.path.join(output_dir, f"clean_{filename}")
        df.to_parquet(output_path, index=False)
        logger.info("Archivo procesado y guardado: %s", output_path)
        return output_path
    except Exception as exc:
        logger.error("Error al procesar %s: %s", filepath, exc, exc_info=True)
        return None

def preprocess_data(
    parallel: bool = True,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    files: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """Preprocesa todos los CSV encontrados en data/raw/."""
    src_dir = input_dir or DEFAULT_RAW_DIR
    dst_dir = output_dir or DEFAULT_PROCESSED_DIR

    if not os.path.isdir(src_dir):
        logger.error("No se encontró el directorio de datos brutos: %s", src_dir)
        return None

    os.makedirs(dst_dir, exist_ok=True)

    if files:
        csv_files: List[str] = []
        for name in files:
            candidate = name if os.path.isabs(name) else os.path.join(src_dir, name)
            if os.path.isfile(candidate):
                csv_files.append(candidate)
            else:
                logger.warning("Archivo solicitado no encontrado; se omitirá: %s", candidate)
    else:
        csv_files = sorted(
            os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".csv")
        )

    if not csv_files:
        logger.warning("No se encontraron CSV para preprocesar.")
        return None

    logger.info("Archivos detectados: %d (directorio de entrada: %s)", len(csv_files), src_dir)

    processed_files: List[str] = []
    failed_files: List[str] = []

    def _record_result(source: str, result: Optional[str]) -> None:
        (processed_files if result else failed_files).append(result or source)

    if parallel and len(csv_files) > 1:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(process_file, csv_files, [dst_dir] * len(csv_files)))
        for fpath, res in zip(csv_files, results):
            _record_result(fpath, res)
    else:
        for f in csv_files:
            _record_result(f, process_file(f, dst_dir))

    metadata = {
        "processed_files": [p for p in processed_files if p],
        "failed_files": failed_files,
        "total_processed": len(processed_files),
        "total_failed": len(failed_files),
        "date": str(pd.Timestamp.now()),
    }

    metadata_path = os.path.join(dst_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info("Metadatos guardados en: %s", metadata_path)
    return processed_files

# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Etapa de preprocesamiento optimizada para señales de running.")
    parser.add_argument("--no-parallel", action="store_true", help="Desactivar ejecución en paralelo.")
    parser.add_argument("--input-dir", help="Directorio con los CSV brutos.")
    parser.add_argument("--output-dir", help="Directorio destino para los Parquet procesados.")
    parser.add_argument(
        "--files",
        nargs="+",
        help="Lista opcional de CSV a procesar (relativos a --input-dir salvo ruta absoluta).",
    )
    args = parser.parse_args()
    processed = preprocess_data(
        parallel=not args.no_parallel,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        files=args.files,
    )
    if processed:
        logger.info("Preprocesamiento completado correctamente.")
    else:
        logger.warning("No se procesó ningún archivo.")
