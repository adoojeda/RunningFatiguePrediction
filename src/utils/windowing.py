"""
Utilidades para el cálculo de características con ventanas deslizantes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class WindowParams:
    """Configuración usada en el proceso de ventanas deslizantes."""
    size: float
    step: float
    min_samples: int

def create_window_params(window: float, overlap: float, *, min_samples: int) -> WindowParams:
    """Construye los parámetros de ventana garantizando un paso válido."""
    step = window * (1.0 - overlap)
    if step <= 0:
        raise ValueError("El paso calculado es <= 0. Revisa la configuración de ventana/solape.")
    return WindowParams(size=window, step=step, min_samples=min_samples)

def prepare_dataframe(df: pd.DataFrame, numeric_cols: Sequence[str]) -> pd.DataFrame:
    """Ordena temporalmente y convierte a numérico antes de ventanear."""
    df = df.sort_values("relative_time").reset_index(drop=True)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def iter_windows(df: pd.DataFrame, params: WindowParams) -> Iterator[Tuple[float, float, pd.DataFrame]]:
    """Genera tuplas (inicio, fin, ventana) a lo largo del dataframe."""
    t_start = float(df["relative_time"].min())
    t_end = float(df["relative_time"].max())
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        raise ValueError("Rango de relative_time inválido.")

    current = t_start
    while current + params.size <= t_end + 1e-9:
        w_end = current + params.size
        mask = (df["relative_time"] >= current) & (df["relative_time"] < w_end)
        df_win = df.loc[mask]
        if len(df_win) >= params.min_samples:
            yield current, w_end, df_win
        current += params.step
