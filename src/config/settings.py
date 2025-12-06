"""
Configuración centralizada para el pipeline Running Fatigue Prediction.

Todas las etapas (preprocesamiento, cinemática, métricas, extracción de ventanas y modelado)
consumen estos ajustes para mantener un comportamiento consistente. 
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

# AYUDAS PARA VARIABLES DE ENTORNO
def _get_float(name: str, default: float) -> float:
    """Lee un float del entorno con retroceso seguro."""
    value = os.getenv(name)
    try:
        return float(value) if value is not None else float(default)
    except (TypeError, ValueError):
        return float(default)

def _get_int(name: str, default: int) -> int:
    """Lee un int del entorno con retroceso seguro."""
    value = os.getenv(name)
    try:
        return int(value) if value is not None else int(default)
    except (TypeError, ValueError):
        return int(default)
    
# DATACLASSES DE CONFIGURACIÓN
@dataclass(frozen=True)
class SamplingConfig:
    default_fs: float = _get_float("RFP_DEFAULT_FS", 50.0)
    highpass_cutoff: float = _get_float("RFP_HP_CUTOFF", 0.3)
    vtr_smoothing: int = _get_int("RFP_VTR_SMOOTHING", 10)

@dataclass(frozen=True)
class PhysiologicalRanges:
    hr: Tuple[float, float] = (
        _get_float("RFP_FC_MIN", 40.0),
        _get_float("RFP_FC_MAX", 220.0),
    )
    spo2: Tuple[float, float] = (
        _get_float("RFP_SPO2_MIN", 70.0),
        _get_float("RFP_SPO2_MAX", 100.0),
    )
    acc_max: float = _get_float("RFP_ACC_MAX", 50.0)

@dataclass(frozen=True)
class InterpolationConfig:
    max_gap_seconds: float = _get_float("RFP_INTERP_MAX_GAP_SEC", 1.0)

@dataclass(frozen=True)
class WindowingConfig:
    size_seconds: float = _get_float("RFP_WINDOW_SECONDS", 3.0)
    overlap_ratio: float = _get_float("RFP_WINDOW_OVERLAP", 0.75)
    min_samples: int = _get_int("RFP_WINDOW_MIN_SAMPLES", 5)

@dataclass(frozen=True)
class FatigueWeights:
    weights: Dict[str, float] = None 

    def __post_init__(self) -> None:
        default_weights = {
            "jerk": _get_float("RFP_FATIGUE_WEIGHT_JERK", 0.5745),
            "acc": _get_float("RFP_FATIGUE_WEIGHT_ACC", 0.1242),
            "hr": _get_float("RFP_FATIGUE_WEIGHT_FC", 0.2329),
            "spo2": _get_float("RFP_FATIGUE_WEIGHT_SPO2", 0.0684),
        }
        object.__setattr__(self, "weights", default_weights)

@dataclass(frozen=True)
class FatigueReferences:
    references: Dict[str, float] = None  
    def __post_init__(self) -> None:
        defaults = {
            "hr_max": _get_float("RFP_FATIGUE_FC_MAX", 200.0),
            "spo2_min": _get_float("RFP_FATIGUE_SPO2_MIN", 90.0),
            "acc_std_ref": _get_float("RFP_FATIGUE_ACC_STD_REF", 5.0),
            "jerk_std_ref": _get_float("RFP_FATIGUE_JERK_STD_REF", 50.0),
        }
        object.__setattr__(self, "references", defaults)

@dataclass(frozen=True)
class WorkforceConfig:
    max_workers: int = max(1, os.cpu_count() // 2) if os.cpu_count() else 1

@dataclass(frozen=True)
class PipelineConfig:
    sampling: SamplingConfig = SamplingConfig()
    ranges: PhysiologicalRanges = PhysiologicalRanges()
    interpolation: InterpolationConfig = InterpolationConfig()
    windows: WindowingConfig = WindowingConfig()
    fatigue_weights: FatigueWeights = FatigueWeights()
    fatigue_refs: FatigueReferences = FatigueReferences()
    workforce: WorkforceConfig = WorkforceConfig()

# ACCESO A CONFIGURACIÓN CACHEADA
@lru_cache(maxsize=1)
def get_config() -> PipelineConfig:
    """Devuelve una instancia cacheada de la configuración."""
    return PipelineConfig()

__all__ = ["PipelineConfig", "get_config"]
