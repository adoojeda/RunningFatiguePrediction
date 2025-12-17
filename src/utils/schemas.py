"""
Schema definitions and lightweight validators shared across the pipeline.

Each stage validates its inputs to catch missing columns or wrong dtypes
before running expensive computations.
"""

# STANDARD LIBRARIES
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd
from pandas.api import types as ptypes

# DATA SCHEMA
@dataclass(frozen=True)
class DataSchema:
    """Minimal schema definition used for dataframe validation."""
    name: str
    required: Iterable[str]
    numeric: Iterable[str]
    description: str

# COLUMN DEFINITIONS (RAW → ENRICHED)
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

PROCESSED_COLUMNS: List[str] = [
    "relative_time",
    *RAW_COLUMNS[1:],  
]

ENRICHED_COLUMNS: List[str] = [
    *PROCESSED_COLUMNS,
    "acc_x_centered",
    "acc_y_centered",
    "acc_z_centered",
    "acc_mag",
    "acc_dyn_mag",
]

# SCHEMA REGISTRY
SCHEMAS: Dict[str, DataSchema] = {
    "raw": DataSchema(
        name="raw",
        required=RAW_COLUMNS,
        numeric=RAW_COLUMNS,
        description="Raw CSV recorded by the Apple Watch.",
    ),
    "processed": DataSchema(
        name="processed",
        required=PROCESSED_COLUMNS,
        numeric=PROCESSED_COLUMNS,
        description="Clean parquet produced by src/data/preprocess.py.",
    ),
    "enriched": DataSchema(
        name="enriched",
        required=ENRICHED_COLUMNS,
        numeric=ENRICHED_COLUMNS,
        description="Parquet enriched with centred accelerations and magnitudes.",
    ),
}

# VALIDATION UTILITY
def validate_dataframe(
    df: pd.DataFrame,
    schema_name: str,
    *,
    raise_on_error: bool = True,
) -> bool:
    """Validates whether a dataframe complies with the requested schema."""
    if schema_name not in SCHEMAS:
        raise KeyError(f"Unknown schema: '{schema_name}'.")

    schema = SCHEMAS[schema_name]

    missing = [col for col in schema.required if col not in df.columns]
    numeric_mismatches = [
        col for col in schema.numeric if col in df.columns and not ptypes.is_numeric_dtype(df[col])
    ]

    if missing or numeric_mismatches:
        message = (
            f"The dataframe does not comply with schema '{schema.name}'. "
            f"Missing columns: {missing or 'none'}; "
            f"non-numeric columns: {numeric_mismatches or 'none'}."
        )
        if raise_on_error:
            raise ValueError(message)
        return False

    return True

__all__ = ["validate_dataframe", "SCHEMAS"]
