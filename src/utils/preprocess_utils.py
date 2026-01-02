"""
Utilidades auxiliares para la etapa de preprocesamiento.

Incluye:
- carga de CSV y validación del esquema esperado,
- conversión de umbrales temporales a límites de interpolación,
- filtrado fisiológico de HR y SpO₂,
- interpolación de huecos cortos,
- eliminación de muestras de aceleración imposibles.
"""

# LIBRERÍAS
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np # type: ignore
import pandas as pd # type: ignore

# IMPORTS DEL PROYECTO
from src.utils.schemas import validate_dataframe

# EXCEPCIONES PERSONALIZADAS
class PreprocessError(Exception):
    """Error base para incidencias de preprocesamiento."""

class EmptyFileError(PreprocessError):
    """Se lanza cuando un CSV no contiene datos."""

class ColumnCountError(PreprocessError):
    """Se lanza cuando el CSV no contiene el número de columnas esperado."""

# RESUMEN DE PREPROCESAMIENTO
@dataclass
class PreprocessStats:
    """Resumen por archivo para enriquecer los logs."""

    samples_in: int = 0
    samples_out: int = 0
    interpolated_hr: int = 0
    interpolated_spo2: int = 0
    acc_outliers_removed: int = 0

    def as_dict(self) -> dict:
        return {
            "samples_in": self.samples_in,
            "samples_out": self.samples_out,
            "interpolated_hr": self.interpolated_hr,
            "interpolated_spo2": self.interpolated_spo2,
            "acc_outliers_removed": self.acc_outliers_removed,
        }
    
# CARGA Y VALIDACIÓN DE CSV
def load_raw_file(filepath: str, expected_columns: int = 15) -> pd.DataFrame:
    """Lee un CSV bruto y valida el esquema esperado."""
    df = pd.read_csv(filepath, header=None)
    if df.empty:
        raise EmptyFileError("El archivo está vacío.")
    if df.shape[1] < expected_columns:
        raise ColumnCountError(
            f"Se esperaban {expected_columns} columnas, se encontraron {df.shape[1]}."
        )

    df.columns = [
        "time",
        "acc_x",
        "acc_y",
        "acc_z",
        "grav_x",
        "grav_y",
        "grav_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "roll",
        "pitch",
        "yaw",
        "hr",
        "spo2",
    ]
    return df

# UTILIDADES DE PREPROCESAMIENTO
def ensure_numeric(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> None:
    """Convierte a numérico las columnas indicadas (o todas) en el propio DataFrame."""
    if columns is None:
        columns = df.columns
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def derive_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `relative_time` a partir de la marca temporal absoluta."""
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    if df.empty:
        raise PreprocessError("Todas las marcas temporales son NaN.")
    df["relative_time"] = df["time"] - df["time"].iloc[0]
    df.drop(columns=["time"], inplace=True, errors="ignore")
    return df

def finalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena columnas y valida el esquema del dataframe procesado."""
    ordered_columns = ["relative_time"] + [col for col in df.columns if col != "relative_time"]
    df = df[ordered_columns]
    validate_dataframe(df, "processed")
    return df

# UTILIDADES DE INTERPOLACIÓN
def interp_limit_from_seconds(fs_est: float, seconds: float, fallback: int = 5) -> int:
    """Convierte un intervalo temporal en número de muestras consecutivas a interpolar."""
    if not np.isfinite(fs_est) or fs_est <= 0:
        return fallback
    return max(1, int(round(fs_est * seconds)))

# FILTRADO DE DATOS FISIOLÓGICOS
def apply_physio_filters(
    df: pd.DataFrame,
    hr_range: tuple[float, float],
    spo2_range: tuple[float, float],
) -> None:
    """Sustituye sentinelas y anula valores fisiológicos fuera de rango."""
    df["hr"] = df["hr"].replace(999, np.nan)
    df["spo2"] = df["spo2"].replace(999, np.nan)
    df.loc[~df["hr"].between(*hr_range, inclusive="both"), "hr"] = np.nan
    df.loc[~df["spo2"].between(*spo2_range, inclusive="both"), "spo2"] = np.nan

# INTERPOLACIÓN DE CANALES
def interpolate_channels(df: pd.DataFrame, limit: int) -> tuple[int, int]:
    """Interpola HR y SpO₂ y devuelve el número de muestras recuperadas por canal."""
    hr_before = df["hr"].isna().sum()
    spo2_before = df["spo2"].isna().sum()

    df["hr"] = df["hr"].interpolate(limit=limit, limit_direction="both")
    df["spo2"] = df["spo2"].interpolate(limit=limit, limit_direction="both")

    hr_after = df["hr"].isna().sum()
    spo2_after = df["spo2"].isna().sum()

    return max(hr_before - hr_after, 0), max(spo2_before - spo2_after, 0)

# FILTRADO DE MUESTRAS DE ACELERACIÓN
def filter_acc_outliers(df: pd.DataFrame, acc_max: float) -> int:
    """Elimina filas con aceleraciones imposibles y devuelve cuántas se eliminaron."""
    initial = len(df)
    mask_acc = df[["acc_x", "acc_y", "acc_z"]].abs().max(axis=1) < acc_max

    df.drop(index=df.index[~mask_acc], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return initial - len(df)
