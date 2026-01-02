"""
Definiciones de esquemas y validadores ligeros compartidos en el pipeline.

Cada etapa valida sus entradas para detectar columnas ausentes o tipos
incorrectos antes de ejecutar cálculos costosos.
"""

# LIBRERÍAS ESTÁNDAR
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd # type: ignore
from pandas.api import types as ptypes # type: ignore

# DEFINICIÓN DE ESQUEMA
@dataclass(frozen=True)
class DataSchema:
    """Definición mínima de esquema para validar dataframes."""
    name: str
    required: Iterable[str]
    numeric: Iterable[str]
    description: str

# DEFINICIONES DE COLUMNAS
RAW_COLUMNS: List[str] = [
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

# COLUMNAS DERIVADAS
PROCESSED_COLUMNS: List[str] = [
    "relative_time",
    *RAW_COLUMNS[1:],  
]

# COLUMNAS ENRIQUECIDAS
ENRICHED_COLUMNS: List[str] = [
    *PROCESSED_COLUMNS,
    "acc_x_centered",
    "acc_y_centered",
    "acc_z_centered",
    "acc_mag",
    "acc_dyn_mag",
]

# ESQUEMAS DEFINIDOS
SCHEMAS: Dict[str, DataSchema] = {
    "raw": DataSchema(
        name="raw",
        required=RAW_COLUMNS,
        numeric=RAW_COLUMNS,
        description="CSV bruto registrado por el Apple Watch.",
    ),
    "processed": DataSchema(
        name="processed",
        required=PROCESSED_COLUMNS,
        numeric=PROCESSED_COLUMNS,
        description="Parquet limpio producido por src/data/preprocess.py.",
    ),
    "enriched": DataSchema(
        name="enriched",
        required=ENRICHED_COLUMNS,
        numeric=ENRICHED_COLUMNS,
        description="Parquet enriquecido con aceleraciones centradas y magnitudes.",
    ),
}

# FUNCIONES DE VALIDACIÓN
def validate_dataframe(
    df: pd.DataFrame,
    schema_name: str,
    *,
    raise_on_error: bool = True,
) -> bool:
    """Valida si un dataframe cumple con el esquema solicitado."""
    if schema_name not in SCHEMAS:
        raise KeyError(f"Esquema desconocido: '{schema_name}'.")

    schema = SCHEMAS[schema_name]

    missing = [col for col in schema.required if col not in df.columns]
    numeric_mismatches = [
        col for col in schema.numeric if col in df.columns and not ptypes.is_numeric_dtype(df[col])
    ]

    if missing or numeric_mismatches:
        message = (
            f"El dataframe no cumple con el esquema '{schema.name}'. "
            f"Columnas faltantes: {missing or 'ninguna'}; "
            f"columnas no numéricas: {numeric_mismatches or 'ninguna'}."
        )
        if raise_on_error:
            raise ValueError(message)
        return False

    return True

__all__ = ["validate_dataframe", "SCHEMAS"]
