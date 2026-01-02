"""
Funciones compartidas para el calculo de características cinemáticas del pipeline.

Incluye funciones para estimar la frecuencia de muestreo, centrar aceleraciones,
calcular magnitudes, aplicar filtros pasa-alto, integrar para obtener velocidad
traslacional y calcular jerk.
"""

# LIBRERÍAS 
from __future__ import annotations

import logging
from typing import Iterable, Sequence
import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy import integrate # type: ignore
from scipy.signal import butter, filtfilt # type: ignore

# IMPORTACIONES DEL PROYECTO
from src.config import get_config

# CONFIGURACIÓN DE LOGGING
logger = logging.getLogger(__name__)

# PARÁMETROS POR DEFECTO
CFG = get_config()
DEFAULT_FS: float = CFG.sampling.default_fs
DEFAULT_HP_CUTOFF: float = CFG.sampling.highpass_cutoff
DEFAULT_VTR_SMOOTHING: int = CFG.sampling.vtr_smoothing

# FUNCIONES AUXILIARES
def estimate_sampling_rate(time: Sequence[float]) -> float:
    """Estima la frecuencia de muestreo a partir de una secuencia temporal."""
    array = pd.to_numeric(time, errors="coerce").dropna().to_numpy() if isinstance(time, pd.Series) else np.asarray(time)
    if array.size < 2:
        return np.nan
    diffs = np.diff(array)
    median_dt = np.median(diffs)
    if median_dt <= 0 or not np.isfinite(median_dt):
        return np.nan
    return 1.0 / median_dt

def centre_accelerations(df: pd.DataFrame, axes: Iterable[str] = ("x", "y", "z"), *, prefix: str = "acc_",
                         suffix: str = "_centered") -> pd.DataFrame:
    """Centra las columnas de aceleración restando la media."""
    for axis in axes:
        col = f"{prefix}{axis}"
        centred = f"{col}{suffix}"
        if col not in df.columns:
            logger.warning("No se encontro la columna de aceleracion %s; se omite el centrado.", col)
            continue
        df[centred] = df[col] - df[col].mean()
    return df

def compute_acceleration_magnitudes(
    df: pd.DataFrame,
    axes: Iterable[str] = ("x", "y", "z"),
    *,
    prefix: str = "acc_",
    centred_suffix: str = "_centered",
    raw_mag_col: str = "acc_mag",
    dyn_mag_col: str = "acc_dyn_mag",
) -> pd.DataFrame:
    """Calcula las magnitudes de aceleración cruda y centrada."""
    required_raw = [f"{prefix}{axis}" for axis in axes]
    required_centred = [f"{prefix}{axis}{centred_suffix}" for axis in axes]

    if all(col in df.columns for col in required_raw):
        df[raw_mag_col] = np.sqrt(sum(df[col] ** 2 for col in required_raw))
    else:
        missing = [col for col in required_raw if col not in df.columns]
        logger.warning("Faltan columnas de aceleracion cruda (%s); no se genera acc_mag.", missing)

    if all(col in df.columns for col in required_centred):
        df[dyn_mag_col] = np.sqrt(sum(df[col] ** 2 for col in required_centred))
    else:
        missing = [col for col in required_centred if col not in df.columns]
        logger.warning(
            "Faltan columnas de aceleracion centrada (%s); no se genera acc_dyn_mag.",
            missing,
        )

    return df

# FUNCIONES PRINCIPALES
def highpass_filter(
    data: np.ndarray,
    fs: float,
    cutoff: float = DEFAULT_HP_CUTOFF,
    order: int = 3,
) -> np.ndarray:
    """Aplica un filtro pasa-alto Butterworth a los datos."""
    array = np.asarray(data, dtype=float)
    if not np.isfinite(fs) or fs <= 0 or array.size < 10:
        return array

    if np.isnan(array).any():
        idx = np.arange(array.size)
        array = np.interp(idx, idx[~np.isnan(array)], array[~np.isnan(array)])

    nyquist = 0.5 * fs
    normal_cutoff = min(cutoff / nyquist, 0.99)
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    try:
        return filtfilt(b, a, array, method="gust")
    except Exception:  
        return filtfilt(b, a, array)

def compute_translational_velocity(
    df: pd.DataFrame,
    *,
    time_col: str = "relative_time",
    accel_prefix: str = "acc_",
    centred_suffix: str = "_centered",
    axes: Iterable[str] = ("x", "y", "z"),
    velocity_prefix: str = "v",
    velocity_mag_col: str = "vtr",
    default_fs: float = DEFAULT_FS,
    cutoff: float = DEFAULT_HP_CUTOFF,
) -> pd.DataFrame:
    """Calcula la velocidad traslacional integrando aceleraciones centradas filtradas."""
    required = [f"{accel_prefix}{axis}{centred_suffix}" for axis in axes]
    if time_col not in df.columns or any(col not in df.columns for col in required):
        missing = [col for col in [time_col, *required] if col not in df.columns]
        logger.warning("Faltan columnas para calcular velocidad: %s", missing)
        return df

    t = df[time_col].to_numpy(dtype=float)
    if t.size < 2:
        logger.warning("No hay suficientes muestras para integrar la velocidad.")
        return df

    fs = estimate_sampling_rate(t) or default_fs

    filtered = {}
    for axis in axes:
        col = f"{accel_prefix}{axis}{centred_suffix}"
        filt = highpass_filter(df[col].to_numpy(dtype=float), fs, cutoff=cutoff)
        filtered[axis] = filt - np.mean(filt)

    velocities = {}
    for axis in axes:
        velocities[axis] = integrate.cumtrapz(filtered[axis], x=t, initial=0.0)
        velocities[axis] -= np.mean(velocities[axis])
        df[f"{velocity_prefix}{axis}"] = velocities[axis]

    mag = np.sqrt(sum(velocities[axis] ** 2 for axis in axes))
    df[velocity_mag_col] = mag
    return df

def compute_jerk(
    df: pd.DataFrame,
    *,
    time_col: str = "relative_time",
    accel_prefix: str = "acc_",
    centred_suffix: str = "_centered",
    axes: Iterable[str] = ("x", "y", "z"),
    jerk_prefix: str = "jerk_",
    jerk_mag_col: str = "jerk_mag",
) -> pd.DataFrame:
    """Calcula el jerk (derivada de la aceleración) por eje y su magnitud."""
    if time_col not in df.columns:
        logger.warning("La columna %s es obligatoria para calcular jerk.", time_col)
        return df

    t = df[time_col].to_numpy(dtype=float)
    if t.size < 2:
        logger.warning("No hay suficientes muestras para calcular jerk.")
        return df

    if np.any(~np.isfinite(t)):
        logger.warning("El vector temporal contiene valores no finitos; jerk se fija a NaN.")
        for axis in axes:
            df[f"{jerk_prefix}{axis}"] = np.nan
        df[jerk_mag_col] = np.nan
        return df

    diffs = np.diff(t)
    if np.any(diffs <= 0):
        logger.warning("El vector temporal no es estrictamente creciente; jerk se fija a NaN.")
        for axis in axes:
            df[f"{jerk_prefix}{axis}"] = np.nan
        df[jerk_mag_col] = np.nan
        return df

    jerks = {}
    for axis in axes:
        col = f"{accel_prefix}{axis}{centred_suffix}"
        if col not in df.columns:
            logger.warning("Se omite el jerk para %s (columna ausente).", col)
            df[f"{jerk_prefix}{axis}"] = np.nan
            continue
        values = df[col].to_numpy(dtype=float)
        if np.isnan(values).all():
            logger.warning("Todos los valores son NaN para %s; jerk se fija a NaN.", col)
            df[f"{jerk_prefix}{axis}"] = np.nan
            continue

        if np.isnan(values).any():
            idx = np.arange(values.size)
            valid = ~np.isnan(values)
            values = np.interp(idx, idx[valid], values[valid])

        edge_order = 2 if values.size >= 3 else 1
        jerks[axis] = np.gradient(values, t, edge_order=edge_order)
        df[f"{jerk_prefix}{axis}"] = jerks[axis]

    valid_axes = [axis for axis in axes if axis in jerks]
    if valid_axes:
        df[jerk_mag_col] = np.sqrt(sum(jerks[axis] ** 2 for axis in valid_axes))
    else:
        df[jerk_mag_col] = np.nan
    return df

__all__ = [
    "DEFAULT_FS",
    "DEFAULT_HP_CUTOFF",
    "DEFAULT_VTR_SMOOTHING",
    "centre_accelerations",
    "compute_acceleration_magnitudes",
    "compute_translational_velocity",
    "compute_jerk",
    "estimate_sampling_rate",
    "highpass_filter",
]
