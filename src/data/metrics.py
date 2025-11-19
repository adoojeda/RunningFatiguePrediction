"""
Biomechanical/physiological metrics (pipeline stage 3/5):
- High-pass filter + integration of centred accelerations → translational velocity (Vtr).
- Jerk computation (derivative of acceleration) and fatigue scoring.
- Persist enriched sessions and a consolidated metrics table.

Input: `data/enriched/enriched_*.parquet`
Outputs: updated enriched parquet files + `data/results/all_sessions_metrics.parquet`
Next stage: `python src/features/features_extraction.py`
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from glob import glob
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import get_config
from src.utils.kinematics import (
    DEFAULT_FS,
    DEFAULT_HP_CUTOFF,
    centre_accelerations,
    compute_acceleration_magnitudes,
    compute_translational_velocity,
    compute_jerk,
)
from src.utils.schemas import validate_dataframe

# ===========================
# LOGGING
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ===========================
# PATHS & CONFIG
# ===========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

OUTPUT_PARQUET = os.path.join(RESULTS_DIR, "all_sessions_metrics.parquet")

CFG = get_config()
WEIGHTS = dict(CFG.fatigue_weights.weights)
DEFAULT_FATIGUE_REFERENCES: Dict[str, float] = dict(CFG.fatigue_refs.references)

# ===========================
# DATA CLASSES
# ===========================
class MetricsError(Exception):
    """Base exception for the metrics pipeline."""

@dataclass
class SessionResult:
    """Summary of the processing outcome for a single session."""
    file: str
    fatigue_score: float
    metrics: Dict[str, float]

# ===========================
# SESSION METRICS
# ===========================
def compute_session_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute descriptive statistics for biomechanical and physiological signals."""
    metrics: Dict[str, float] = {}
    if df is None or df.empty:
        return metrics

    def safe_stat(col: str) -> tuple[float, float]:
        series = pd.to_numeric(df[col], errors="coerce")
        return series.mean(), series.std()

    try:
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
    except Exception as exc:
        logger.error("Error computing session metrics: %s", exc, exc_info=True)

    return metrics

# ===========================
# FATIGUE REFERENCES
# ===========================
def _safe_percentile(series: pd.Series, q: float) -> float:
    """Percentile helper resilient to NaNs and empty inputs."""
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return np.nan
    return float(np.nanpercentile(values.to_numpy(dtype=float), q))

def _safe_std(series: pd.Series) -> float:
    """Standard deviation helper resilient to NaNs and empty inputs."""
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() < 2:
        return np.nan
    return float(np.nanstd(values.to_numpy(dtype=float), ddof=1))

def derive_fatigue_references(df: pd.DataFrame) -> Dict[str, float]:
    """
    Estimate fatigue normalisation references from a session dataframe.
    """
    refs = DEFAULT_FATIGUE_REFERENCES.copy()

    try:
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

    except Exception as exc:
        logger.warning("Failed to derive fatigue references: %s", exc, exc_info=True)

    return refs

# ===========================
# FATIGUE SCORING
# ===========================
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
    metrics["fatigue_components"] = {
        "norm_fc": round(norm_fc, 3),
        "norm_spo2": round(norm_spo2, 3),
        "norm_acc": round(norm_acc, 3),
        "norm_jerk": round(norm_jerk, 3),
    }
    metrics["fatigue_references"] = {
        "fc_max": round(params["fc_max"], 3),
        "spo2_min": round(params["spo2_min"], 3),
        "acc_std_ref": round(params["acc_std_ref"], 3),
        "jerk_std_ref": round(params["jerk_std_ref"], 3),
    }
    return metrics

# ===========================
# DATAFRAME HANDLING
# ===========================
def _load_session(path: str) -> pd.DataFrame:
    """Load a parquet file and ensure the appropriate schema is satisfied."""
    df = pd.read_parquet(path)
    schema_name = "enriched" if os.path.basename(path).startswith("enriched_") else "processed"
    validate_dataframe(df, schema_name)
    return df

def _apply_biomechanics(df: pd.DataFrame) -> pd.DataFrame:
    """Run the biomechanical enrichments required before scoring."""
    if "relative_time" in df.columns:
        df = (
            df.sort_values("relative_time")
            .drop_duplicates(subset="relative_time", keep="first")
            .reset_index(drop=True)
        )
    df = centre_accelerations(df)
    df = compute_acceleration_magnitudes(df)
    df = compute_translational_velocity(
        df,
        default_fs=DEFAULT_FS,
        cutoff=DEFAULT_HP_CUTOFF,
    )
    df = compute_jerk(df)
    return df

def _annotate_fatigue(df: pd.DataFrame, session_metrics: Dict[str, float]) -> None:
    """Attach fatigue score and components to the dataframe in place."""
    fatigue_score = session_metrics.get("fatigue_score")
    if fatigue_score is None or not np.isfinite(fatigue_score):
        return

    df["fatigue_score"] = fatigue_score

def _save_enriched(df: pd.DataFrame, path: str) -> bool:
    """Persist enriched dataframe back to disk."""
    try:
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_parquet(path, index=False)
        logger.info("Enriched file updated: %s", os.path.basename(path))
        return True
    except Exception as exc:
        logger.error("Error saving enriched file %s: %s", os.path.basename(path), exc, exc_info=True)
        return False

# ===========================
# GLOBAL PROCESSING
# ===========================
def process_session(path: str, *, allow_save: bool) -> Optional[SessionResult]:
    """Process a single session file and return the computed metrics."""
    try:
        df = _load_session(path)
        df = _apply_biomechanics(df)

        session_metrics = compute_session_metrics(df)
        references = derive_fatigue_references(df)
        session_metrics = compute_fatigue_score(
            session_metrics,
            context="session",
            references=references,
        )
        _annotate_fatigue(df, session_metrics)

        if allow_save:
            _save_enriched(df, path)

        fatigue_score = session_metrics.get("Fatigue_Score", np.nan)
        return SessionResult(
            file=os.path.basename(path),
            fatigue_score=float(fatigue_score) if np.isfinite(fatigue_score) else np.nan,
            metrics=session_metrics,
        )

    except Exception as exc:
        logger.error("Error processing %s: %s", os.path.basename(path), exc, exc_info=True)
        return None

def process_all_sessions(
    save_enriched: bool = True,
    source_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Process every enriched parquet file and produce the global summary."""
    data_dir = source_dir or (ENRICHED_DIR if os.path.isdir(ENRICHED_DIR) else PROCESSED_DIR)
    if not os.path.isdir(data_dir):
        logger.error("Processed/enriched directory not found: %s", data_dir)
        return None

    pattern = "enriched_*.parquet" if data_dir == ENRICHED_DIR else "clean_*.parquet"
    files = sorted(glob(os.path.join(data_dir, pattern)))
    if not files:
        logger.warning("No input files found in %s", data_dir)
        return None

    results: list[Dict[str, float]] = []
    for path in files:
        result = process_session(path, allow_save=save_enriched and data_dir == ENRICHED_DIR)
        if result:
            payload = result.metrics.copy()
            payload["session_file"] = result.file
            results.append(payload)

    if not results:
        return None

    df_all = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_all.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Global metrics saved to: %s (%d sessions)", OUTPUT_PARQUET, len(df_all))
    return df_all

# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    print("Starting advanced metrics computation...")
    print("=" * 50)

    df_metrics = process_all_sessions(save_enriched=True)

    if df_metrics is not None:
        print("\nCombined metrics generated:")
        print(df_metrics.head())
        print(f"\nProcessed files: {len(df_metrics)}")
        print(f"Global metrics stored in: {RESULTS_DIR}")
    else:
        print("No metrics were generated.")
