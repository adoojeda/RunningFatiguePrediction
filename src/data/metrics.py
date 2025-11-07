"""
Biomechanical/physiological metrics (pipeline stage 3/5):
- High-pass filter + integration of centred accelerations → translational velocity (Vtr).
- Jerk computation (derivative of acceleration) and fatigue scoring.
- Persist enriched sessions and a consolidated metrics table.

Input: `data/enriched/enriched_*.parquet`
Outputs: updated enriched parquet files + `data/results/all_sessions_metrics.parquet`
Next stage: `python src/features/features_extraction.py`
"""

import os
import sys
import logging
from glob import glob
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Ensure project root on sys.path when executed directly
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

# ======================================================================
# LOGGING CONFIGURATION
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================================================
# PATH CONFIGURATION
# ======================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

OUTPUT_PARQUET = os.path.join(RESULTS_DIR, "all_sessions_metrics.parquet")

# ======================================================================
CFG = get_config()

# ======================================================================
# FATIGUE SCORE PARAMETERS
# ======================================================================
WEIGHTS = dict(CFG.fatigue_weights.weights)

DEFAULT_FATIGUE_REFERENCES: Dict[str, float] = dict(CFG.fatigue_refs.references)


# ======================================================================
# SESSION METRICS
# ======================================================================
def compute_session_metrics(df: pd.DataFrame) -> dict:
    """Compute basic metrics for acceleration, jerk, HR, and SpO₂."""
    metrics = {}
    if df is None or df.empty:
        return metrics

    try:
        def safe_stat(col: str) -> tuple[float, float]:
            return df[col].mean(), df[col].std()

        if "Acc_mag" in df.columns:
            metrics["Acc_mean"], metrics["Acc_std"] = safe_stat("Acc_mag")
        if "Acc_dyn_mag" in df.columns:
            metrics["Acc_dyn_mean"], metrics["Acc_dyn_std"] = safe_stat("Acc_dyn_mag")
        if "Vtr" in df.columns:
            metrics["Vtr_mean"], metrics["Vtr_std"] = safe_stat("Vtr")
        if "FC" in df.columns:
            metrics["FC_mean"], metrics["FC_std"] = safe_stat("FC")
        if "SpO2" in df.columns:
            metrics["SpO2_mean"], metrics["SpO2_std"] = safe_stat("SpO2")
        if "jerk_mag" in df.columns:
            metrics["jerk_mean"], metrics["jerk_std"] = safe_stat("jerk_mag")

        return metrics

    except Exception as exc:
        logger.error("❌ Error computing session metrics: %s", exc, exc_info=True)
        return metrics


# ======================================================================
# FATIGUE REFERENCES
# ======================================================================
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

    These references contextualise window-level scores with session-specific
    baselines instead of relying solely on global defaults.
    """
    refs = DEFAULT_FATIGUE_REFERENCES.copy()

    try:
        if "FC" in df.columns:
            fc_95 = _safe_percentile(df["FC"], 95)
            if np.isfinite(fc_95):
                refs["fc_max"] = max(fc_95, 1e-6)

        if "SpO2" in df.columns:
            spo2_05 = _safe_percentile(df["SpO2"], 5)
            if np.isfinite(spo2_05):
                refs["spo2_min"] = min(spo2_05, refs["spo2_min"])

        if "Acc_mag" in df.columns:
            acc_std = _safe_std(df["Acc_mag"])
            if np.isfinite(acc_std):
                refs["acc_std_ref"] = max(acc_std, 1e-6)

        if "jerk_mag" in df.columns:
            jerk_std = _safe_std(df["jerk_mag"])
            if np.isfinite(jerk_std):
                refs["jerk_std_ref"] = max(jerk_std, 1e-6)

    except Exception as exc:
        logger.warning("⚠️ Failed to derive fatigue references: %s", exc, exc_info=True)

    return refs

# ======================================================================
# FATIGUE SCORE
# ======================================================================
def compute_fatigue_score(
    metrics: dict,
    *,
    context: str = "session",
    references: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    adaptive: bool = True,
) -> dict:
    """Compute the composite fatigue score with configurable weights and references."""
    if not metrics:
        return {"Fatigue_Score": np.nan}

    try:
        if context not in {"session", "window"}:
            raise ValueError(f"Invalid context '{context}'. Expected 'session' or 'window'.")

        params = DEFAULT_FATIGUE_REFERENCES.copy()
        if references:
            params.update({k: v for k, v in references.items() if v is not None and np.isfinite(v)})

        weight_cfg = WEIGHTS.copy()
        if weights:
            weight_cfg.update({k: v for k, v in weights.items() if k in weight_cfg})

        fc_denominator = max(params["fc_max"], 1e-6)
        norm_fc = np.clip(metrics.get("FC_mean", 0.0) / fc_denominator, 0.0, 1.0)

        spo2_denominator = max(100.0 - params["spo2_min"], 1e-6)
        norm_spo2 = np.clip(
            1.0 - ((metrics.get("SpO2_mean", 100.0) - params["spo2_min"]) / spo2_denominator),
            0.0,
            1.0,
        )

        acc_std = metrics.get("Acc_std", np.nan)
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

        metrics["Fatigue_Score"] = round(float(fatigue), 3)
        metrics["Fatigue_components"] = {
            "norm_fc": round(norm_fc, 3),
            "norm_spo2": round(norm_spo2, 3),
            "norm_acc": round(norm_acc, 3),
            "norm_jerk": round(norm_jerk, 3),
        }
        metrics["Fatigue_references"] = {
            "fc_max": round(params["fc_max"], 3),
            "spo2_min": round(params["spo2_min"], 3),
            "acc_std_ref": round(params["acc_std_ref"], 3),
            "jerk_std_ref": round(params["jerk_std_ref"], 3),
        }
        return metrics

    except Exception as exc:
        logger.error("❌ Error computing Fatigue Score: %s", exc, exc_info=True)
        return {"Fatigue_Score": np.nan}

# ======================================================================
# SAVE ENRICHED DATA
# ======================================================================
def save_enriched_data(df: pd.DataFrame, original_path: str) -> bool:
    """Persist the DataFrame enriched with the newly computed metrics."""
    try:
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_parquet(original_path, index=False)
        logger.info("💾 Enriched file updated: %s", original_path)
        return True

    except Exception as exc:
        logger.error("❌ Error saving enriched file %s: %s", original_path, exc, exc_info=True)
        return False

# ======================================================================
# GLOBAL PROCESSING
# ======================================================================
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
        logger.warning("⚠️ No input files found in %s", data_dir)
        return None

    all_metrics = []
    for path in files:
        try:
            df = pd.read_parquet(path)
            schema_name = "enriched" if os.path.basename(path).startswith("enriched_") else "processed"
            validate_dataframe(df, schema_name)
            df = centre_accelerations(df)
            df = compute_acceleration_magnitudes(df)
            df = compute_translational_velocity(
                df,
                default_fs=DEFAULT_FS,
                cutoff=DEFAULT_HP_CUTOFF,
            )
            df = compute_jerk(df)

            session_metrics = compute_session_metrics(df)
            references = derive_fatigue_references(df)
            session_metrics = compute_fatigue_score(
                session_metrics,
                context="session",
                references=references,
            )

            fatigue_score = session_metrics.get("Fatigue_Score")
            if fatigue_score is not None and np.isfinite(fatigue_score):
                df["Fatigue_Score"] = fatigue_score
                df["Fatigue_Score_session"] = fatigue_score
                components = session_metrics.get("Fatigue_components", {})
                for key, value in components.items():
                    if value is None or not np.isfinite(value):
                        continue
                    df[f"Fatigue_component_{key}"] = value

            if save_enriched and data_dir == ENRICHED_DIR:
                save_enriched_data(df, path)

            session_metrics["session_file"] = os.path.basename(path)
            all_metrics.append(session_metrics)

            logger.info("✅ Metrics processed: %s", os.path.basename(path))

        except Exception as exc:
            logger.error("❌ Error processing %s: %s", path, exc, exc_info=True)

    if not all_metrics:
        return None

    df_all = pd.DataFrame(all_metrics)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_all.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("📊 Global metrics saved to: %s", OUTPUT_PARQUET)
    return df_all

# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    print("🚀 Starting advanced metrics computation...")
    print("=" * 50)

    df_metrics = process_all_sessions(save_enriched=True)

    if df_metrics is not None:
        print("\n✅ Combined metrics generated:")
        print(df_metrics.head())
        print(f"\n📁 Processed files: {len(df_metrics)}")
        print(f"💾 Global metrics stored in: {RESULTS_DIR}")
    else:
        print("⚠️ No metrics were generated.")
