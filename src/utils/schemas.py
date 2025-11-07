"""
Data schema definitions and lightweight validators shared across the pipeline.

Each stage validates its key inputs to catch missing columns or wrong dtypes
before executing expensive computations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd
from pandas.api import types as ptypes


@dataclass(frozen=True)
class DataSchema:
    """Minimal schema definition used for dataframe validation."""

    name: str
    required: Iterable[str]
    numeric: Iterable[str]
    description: str


RAW_COLUMNS: List[str] = [
    "Time",
    "AccX",
    "AccY",
    "AccZ",
    "GravX",
    "GravY",
    "GravZ",
    "RotX",
    "RotY",
    "RotZ",
    "Roll",
    "Pitch",
    "Yaw",
    "FC",
    "SpO2",
]

PROCESSED_COLUMNS: List[str] = [
    "Relative_Time",
    *RAW_COLUMNS[1:],  
]

ENRICHED_COLUMNS: List[str] = [
    *PROCESSED_COLUMNS,
    "AccX_centered",
    "AccY_centered",
    "AccZ_centered",
    "Acc_mag",
    "Acc_dyn_mag",
]


SCHEMAS: Dict[str, DataSchema] = {
    "raw": DataSchema(
        name="raw",
        required=RAW_COLUMNS,
        numeric=RAW_COLUMNS,
        description="Raw CSV recorded by Apple Watch (90 s window).",
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


def validate_dataframe(
    df: pd.DataFrame,
    schema_name: str,
    *,
    raise_on_error: bool = True,
) -> bool:
    """
    Validate that a dataframe adheres to the requested schema.

    Parameters
    ----------
    df:
        Dataframe to validate.
    schema_name:
        Schema key ('raw', 'processed', 'enriched').
    raise_on_error:
        When True a ValueError is raised on violations.
    """
    if schema_name not in SCHEMAS:
        raise KeyError(f"Schema '{schema_name}' is not registered.")

    schema = SCHEMAS[schema_name]
    missing = [col for col in schema.required if col not in df.columns]
    numeric_mismatches = [
        col for col in schema.numeric if col in df.columns and not ptypes.is_numeric_dtype(df[col])
    ]

    if missing or numeric_mismatches:
        message = (
            f"Dataframe failed '{schema.name}' schema validation. "
            f"Missing columns: {missing or 'none'}; non-numeric columns: {numeric_mismatches or 'none'}."
        )
        if raise_on_error:
            raise ValueError(message)
        return False

    return True


__all__ = ["validate_dataframe", "SCHEMAS"]
