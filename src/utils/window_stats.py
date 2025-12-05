"""
Funciones estadísticas compartidas para el cálculo de características por ventana.
"""

from __future__ import annotations

import numpy as np

# DESVIACIÓN ABSOLUTA MEDIANA
def mad(x: np.ndarray) -> float:
    """Desviación absoluta mediana; robusta ante valores atípicos."""
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

# ASIMETRÍA
def skewness(x: np.ndarray) -> float:
    """Asimetría muestral con manejo de NaN."""
    x = x[~np.isnan(x)]
    if x.size < 3:
        return np.nan
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))

# CURTOSIS
def kurtosis(x: np.ndarray) -> float:
    """Curtosis en exceso (Fisher); devuelve NaN si no hay muestras suficientes."""
    x = x[~np.isnan(x)]
    if x.size < 4:
        return np.nan
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    if s == 0:
        return -3.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)

# DEVIACIÓN ESTÁNDAR, MEDIA Y MEDIANA 
def safe_stats(x: np.ndarray) -> tuple[float, float, float]:
    """Devuelve (media, desviación estándar, mediana) tolerando segmentos vacíos o con NaN."""
    cleaned = x[~np.isnan(x)]
    if cleaned.size == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(cleaned))
    std = float(np.std(cleaned, ddof=1)) if cleaned.size > 1 else 0.0
    median = float(np.median(cleaned))

    return mean, std, median
