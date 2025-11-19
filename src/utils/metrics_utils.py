"""
Utility functions for biomechanical/physiological metrics and fatigue scoring.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import get_config

CFG = get_config()
WEIGHTS = dict(CFG.fatigue_weights.weights)
DEFAULT_FATIGUE_REFERENCES: Dict[str, float] = dict(CFG.fatigue_refs.references)

def compute_session_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute descriptive statistics for biomechanical and physiological signals."""
    metrics: Dict[str, float] = {}
    if df is None or df.empty:
        return metrics

    def safe_stat(col: str) -> tuple[float, float]:
        series = pd.to_numeric(df[col], errors="coerce")
        return series.mean(), series.std()

    if "acc_mag" in df.columns:
        metrics["acc_mean"], metrics["acc_std"] = safe_stat("acc_mag")
    if "acc_dyn_mag" in df.columns:
        metrics["acc_dyn_mean"], metrics["acc_dyn_std"] = safe_stat("acc_dyn_mag")
    if "vtr" in df.columns:
        metrics["vtr_mean"], metrics["vtr_std"] = safe_stat("vtr")
    if "fc" in df.columns:
        metrics["fc_mean"], metrics["fc_std"] = safe_stat("fc")
    if "spo2" in df.columns:
        metrics["spo2_mean"], metrics["spo2_std"] = safe_stat("spo2")
    if "jerk_mag" in df.columns:
        metrics["jerk_mean"], metrics["jerk_std"] = safe_stat("jerk_mag")
    return metrics

def _safe_percentile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return np.nan
    return float(np.nanpercentile(values.to_numpy(dtype=float), q))

def _safe_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() < 2:
        return np.nan
    return float(np.nanstd(values.to_numpy(dtype=float), ddof=1))

def derive_fatigue_references(df: pd.DataFrame) -> Dict[str, float]:
    """Estimate fatigue normalisation references from a session dataframe."""
    refs = DEFAULT_FATIGUE_REFERENCES.copy()

    if "fc" in df.columns:
        fc_95 = _safe_percentile(df["fc"], 95)
        if np.isfinite(fc_95):
            refs["fc_max"] = max(fc_95, 1e-6)

    if "spo2" in df.columns:
        spo2_05 = _safe_percentile(df["spo2"], 5)
        if np.isfinite(spo2_05):
            refs["spo2_min"] = min(spo2_05, refs["spo2_min"])

    if "acc_mag" in df.columns:
        acc_std = _safe_std(df["acc_mag"])
        if np.isfinite(acc_std):
            refs["acc_std_ref"] = max(acc_std, 1e-6)

    if "jerk_mag" in df.columns:
        jerk_std = _safe_std(df["jerk_mag"])
        if np.isfinite(jerk_std):
            refs["jerk_std_ref"] = max(jerk_std, 1e-6)

    return refs

def compute_fatigue_score(
    metrics: Dict[str, float],
    *,
    context: str = "session",
    references: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    adaptive: bool = True,
) -> Dict[str, float]:
    """Compute the composite fatigue score with configurable weights and references."""
    if not metrics:
        return {"fatigue_score": np.nan}

    if context not in {"session", "window"}:
        raise ValueError(f"Invalid context '{context}'. Expected 'session' or 'window'.")

    params = DEFAULT_FATIGUE_REFERENCES.copy()
    if references:
        params.update({k: v for k, v in references.items() if v is not None and np.isfinite(v)})

    weight_cfg = WEIGHTS.copy()
    if weights:
        weight_cfg.update({k: v for k, v in weights.items() if k in weight_cfg})

    fc_denominator = max(params["fc_max"], 1e-6)
    norm_fc = np.clip(metrics.get("fc_mean", 0.0) / fc_denominator, 0.0, 1.0)

    spo2_denominator = max(100.0 - params["spo2_min"], 1e-6)
    norm_spo2 = np.clip(
        1.0 - ((metrics.get("spo2_mean", 100.0) - params["spo2_min"]) / spo2_denominator),
        0.0,
        1.0,
    )

    acc_std = metrics.get("acc_std", np.nan)
    jerk_std = metrics.get("jerk_std", np.nan)

    if context == "window":
        acc_ref = max(params["acc_std_ref"], 1e-6)
        jerk_ref = max(params["jerk_std_ref"], 1e-6)
        norm_acc = np.clip(acc_std / acc_ref, 0.0, 1.0) if np.isfinite(acc_std) else 0.0
        norm_jerk = np.clip(jerk_std / jerk_ref, 0.0, 1.0) if np.isfinite(jerk_std) else 0.0
    else:
        acc_std_max = params["acc_std_ref"]
        jerk_std_max = params["jerk_std_ref"]
        if adaptive and np.isfinite(acc_std):
            acc_std_max = max(acc_std * 1.2, acc_std_max)
        if adaptive and np.isfinite(jerk_std):
            jerk_std_max = max(jerk_std * 1.2, jerk_std_max)
        norm_acc = (
            np.clip(acc_std / max(acc_std_max, 1e-6), 0.0, 1.0) if np.isfinite(acc_std) else 0.0
        )
        norm_jerk = (
            np.clip(jerk_std / max(jerk_std_max, 1e-6), 0.0, 1.0) if np.isfinite(jerk_std) else 0.0
        )

    fatigue = (
        weight_cfg["jerk"] * norm_jerk
        + weight_cfg["acc"] * norm_acc
        + weight_cfg["fc"] * norm_fc
        + weight_cfg["spo2"] * norm_spo2
    )

    metrics["fatigue_score"] = round(float(fatigue), 3)
    return metrics

__all__ = [
    "compute_session_metrics",
    "derive_fatigue_references",
    "compute_fatigue_score",
]
