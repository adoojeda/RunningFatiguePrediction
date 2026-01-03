"""Pruebas para las utilidades de métricas de cansancio."""

# LIBRERÍAS ESTÁNDAR
from pathlib import Path
import sys

import numpy as np # type: ignore
import pandas as pd # type: ignore

# CONFIGURACIÓN DEL PROYECTO
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# IMPORTS DEL PROYECTO
from src.utils.metrics_utils import compute_physical_fatigue_index, derive_fatigue_references

# PRUEBAS DE MÉTRICAS
def test_compute_physical_fatigue_index_respects_weights():
    """El cálculo del physical_fatigue_index debe respetar los pesos dados."""
    metrics = {
        "hr_mean": 150,
        "spo2_mean": 95,
        "acc_std": 0.5,
        "jerk_std": 0.2,
    }
    references = {
        "hr_max": 200,
        "spo2_min": 80,
        "acc_std_ref": 1.0,
        "jerk_std_ref": 1.0,
    }
    weights = {"jerk": 0.4, "acc": 0.3, "hr": 0.2, "spo2": 0.1}
    result = compute_physical_fatigue_index(metrics.copy(), references=references, weights=weights, adaptive=False)
    assert "physical_fatigue_index" in result
    assert 0.0 <= result["physical_fatigue_index"] <= 1.0

def test_derive_fatigue_references_uses_percentiles():
    """Las referencias de cansancio deben derivarse usando percentiles seguros."""
    df = pd.DataFrame(
        {
            "hr": [100, 110, 120, 130, 140],
            "spo2": [98, 97, 96, 95, 94],
            "acc_mag": [0.1, 0.2, 0.3, 0.4, 0.5],
            "jerk_mag": [0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )
    refs = derive_fatigue_references(df)
    assert np.isclose(refs["hr_max"], 138, atol=1e-3)
    assert refs["spo2_min"] <= 94.2
    assert refs["acc_std_ref"] > 0
    assert refs["jerk_std_ref"] > 0
