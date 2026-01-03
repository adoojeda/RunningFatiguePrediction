"""Funciones para cargar sesiones procesadas/enriquecidas y datasets de características."""

# LIBRERÍAS
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Optional

import numpy as np # type: ignore
import pandas as pd # type: ignore

# CONFIGURACIÓN DE LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# CONFIGURACIÓN DE RUTAS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
DEFAULT_FEATURES_PATH = os.path.join(RESULTS_DIR, "features_dataset.parquet")

# GENERADOR DE RUTAS CANDIDATAS
def _candidate_paths(name: str, directory: str, prefixes: Optional[List[str]] = None) -> List[str]:
    """Genera rutas candidatas para un archivo dado en un directorio con prefijos opcionales."""
    if os.path.isabs(name):
        return [name]

    filename = name if name.endswith(".parquet") else f"{name}.parquet"
    candidates = [os.path.join(directory, filename)]
    for prefix in prefixes or []:
        prefixed = filename if filename.startswith(prefix) else f"{prefix}{filename}"
        candidates.append(os.path.join(directory, prefixed))

    seen = set()
    unique_candidates = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique_candidates.append(path)
    return unique_candidates

# CARGA DE SESIONES INDIVIDUALES
@lru_cache(maxsize=64)
def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Carga un archivo Parquet o CSV y añade una columna 'second' basada en tiempo relativo."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".parquet", ".parq"):
            df = pd.read_parquet(file_path)
        elif ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Formato no soportado: {ext}")

        if df.empty:
            raise ValueError(f"El archivo {os.path.basename(file_path)} está vacío.")

        time_col = "relative_time" if "relative_time" in df.columns else "time"
        if time_col not in df.columns:
            raise KeyError("No se encontró la columna temporal ('relative_time' o 'time').")

        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df.dropna(subset=[time_col], inplace=True)
        df["second"] = np.floor(df[time_col]).astype(int)

        logger.info(
            "Cargado %s (%d filas, %d columnas)",
            os.path.basename(file_path),
            len(df),
            len(df.columns),
        )
        return df

    except Exception as exc:
        logger.error("Error al cargar %s: %s", file_path, exc, exc_info=True)
        return None

# CARGA DE SESIÓN ENRIQUECIDA 
@lru_cache(maxsize=64)
def load_enriched_session(name: str, *, fallback_to_processed: bool = True) -> Optional[pd.DataFrame]:
    """Carga una sesión enriquecida por nombre, con opción de fallback a procesada."""
    for candidate in _candidate_paths(name, ENRICHED_DIR, ["enriched_"]):
        if os.path.exists(candidate):
            return load_data(candidate)

    if not fallback_to_processed:
        logger.warning("Sesión enriquecida %s no encontrada en %s", name, ENRICHED_DIR)
        return None

    processed_base = os.path.splitext(os.path.basename(name))[0]
    if processed_base.startswith("enriched_"):
        processed_base = processed_base[len("enriched_"):]

    for candidate in _candidate_paths(processed_base, PROCESSED_DIR, ["clean_"]):
        if os.path.exists(candidate):
            logger.warning(
                "Falta enriched %s; se usa el archivo procesado %s",
                name,
                os.path.basename(candidate),
            )
            return load_data(candidate)

    logger.error("Sesión %s no encontrada en directorios enriched ni processed.", name)
    return None

# LISTA DE SESIONES ENRIQUECIDAS
def list_enriched_sessions() -> List[str]:
    """Lista los nombres de archivos de sesiones enriquecidas disponibles."""
    if not os.path.isdir(ENRICHED_DIR):
        return []
    return sorted(
        f for f in os.listdir(ENRICHED_DIR)
        if f.endswith(".parquet") and f.startswith("enriched_")
    )

# PROMEDIO POR SEGUNDO
def average_per_second(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Calcula el promedio de columnas numéricas por segundo."""
    try:
        if df is None or df.empty:
            raise ValueError("El DataFrame está vacío o no se cargó correctamente.")

        if "second" not in df.columns and "Second" in df.columns:
            df = df.rename(columns={"Second": "second"})
        if "second" not in df.columns:
            raise KeyError("No se encontró la columna 'second' en el DataFrame.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if columns is not None:
            numeric_cols = [col for col in numeric_cols if col in columns]
        numeric_cols = [col for col in numeric_cols if col != "second"]

        if not numeric_cols:
            raise ValueError("No hay columnas numéricas para promediar.")

        df_avg = (
            df.groupby("second")[numeric_cols]
            .mean()
            .reset_index()
            .sort_values(by="second")
            .reset_index(drop=True)
        )

        logger.info("Promediadas %d filas (1 por segundo).", len(df_avg))
        return df_avg

    except Exception as exc:
        logger.error("Error al promediar por segundo: %s", exc, exc_info=True)
        return None

# CARGA DE MÚLTIPLES SESIONES
@lru_cache(maxsize=8)
def load_all_sessions(limit: Optional[int] = None, prefer_enriched: bool = True) -> Optional[pd.DataFrame]:
    """Carga y concatena todas las sesiones procesadas/enriquecidas disponibles."""
    directories: List[str] = []
    if prefer_enriched and os.path.isdir(ENRICHED_DIR):
        directories.append(ENRICHED_DIR)
    if os.path.isdir(PROCESSED_DIR):
        directories.append(PROCESSED_DIR)

    if not directories:
        logger.warning("No se encontraron directorios válidos para cargar sesiones.")
        return None

    for directory in directories:
        files = [
            f for f in os.listdir(directory)
            if f.endswith(".parquet") and (f.startswith("enriched_") or f.startswith("clean_"))
        ]
        if not files:
            continue

        if limit:
            files = files[:limit]

        dfs = []
        for fname in files:
            path = os.path.join(directory, fname)
            df = load_data(path)
            if df is not None and not df.empty:
                df["file_id"] = os.path.splitext(fname)[0]
                dfs.append(df)

        if dfs:
            df_all = pd.concat(dfs, ignore_index=True)
            logger.info(
                "Cargados y concatenados %d archivos (%d filas) desde %s",
                len(dfs),
                len(df_all),
                directory,
            )
            return df_all

    logger.warning("No se pudieron cargar archivos Parquet desde los directorios disponibles.")
    return None

# CARGA DEL DATASET DE CARACTERÍSTICAS
@lru_cache(maxsize=8)
def load_features_dataset(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Carga el dataset de características desde la ruta especificada o la ruta por defecto."""
    try:
        dataset_path = path or DEFAULT_FEATURES_PATH
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"No se encontró el dataset de características en: {dataset_path}")

        df = pd.read_parquet(dataset_path)
        logger.info("Dataset de características cargado (%d ventanas, %d columnas).", len(df), len(df.columns))
        return df

    except Exception as exc:
        logger.error("Error al cargar el dataset de características: %s", exc, exc_info=True)
        return None

# PRUEBA RÁPIDA DE CARGA
if __name__ == "__main__":
    print("Prueba rápida de carga:")
    sessions = list_enriched_sessions()
    if sessions:
        print("Primera sesión enriquecida:", sessions[0])
        df_session = load_enriched_session(sessions[0])
        if df_session is not None:
            print(df_session.head(3))

    df_all = load_all_sessions(limit=1)
    if df_all is not None:
        print(df_all.head(3))
        df_avg = average_per_second(df_all)
        if df_avg is not None:
            print("\nPromedio por segundo:")
            print(df_avg.head(3))

    df_features = load_features_dataset()
    if df_features is not None:
        print("\nFeatures dataset:")
        print(df_features.head(3))

__all__ = [
    "load_data",
    "load_enriched_session",
    "list_enriched_sessions",
    "average_per_second",
    "load_all_sessions",
    "load_features_dataset",
]
