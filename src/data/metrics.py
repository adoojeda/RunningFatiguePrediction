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
from typing import Optional

import numpy as np
import pandas as pd

# Ensure project root on sys.path when executed directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.kinematics import (
    DEFAULT_FS,
    DEFAULT_HP_CUTOFF,
    centre_accelerations,
    compute_acceleration_magnitudes,
    compute_translational_velocity,
    compute_jerk,
)

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
# FATIGUE SCORE PARAMETERS
# ======================================================================
WEIGHTS = {
    "jerk": 0.35,
    "acc": 0.25,
    "fc": 0.25,
    "spo2": 0.15,
}


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
# FATIGUE SCORE
# ======================================================================
def compute_fatigue_score(
    metrics: dict,
    fc_max: float = 200.0,
    spo2_min: float = 90.0,
    acc_std_ref: float = 5.0,
    jerk_std_ref: float = 50.0,
    adaptive: bool = True,
) -> dict:
    """Compute the composite fatigue score with configurable weights."""
    if not metrics:
        return {"Fatigue_Score": np.nan}

    try:
        norm_fc = np.clip(metrics.get("FC_mean", 0) / fc_max, 0, 1)
        norm_spo2 = np.clip(
            1 - ((metrics.get("SpO2_mean", 100) - spo2_min) / (100 - spo2_min)), 0, 1
        )
        acc_std = metrics.get("Acc_std", 0)
        jerk_std = metrics.get("jerk_std", 0)

        if adaptive:
            acc_std_max = max(acc_std * 1.2, acc_std_ref)
            jerk_std_max = max(jerk_std * 1.2, jerk_std_ref)
        else:
            acc_std_max, jerk_std_max = acc_std_ref, jerk_std_ref

        norm_acc = np.clip(acc_std / acc_std_max, 0, 1)
        norm_jerk = np.clip(jerk_std / jerk_std_max, 0, 1)

        fatigue = (
            WEIGHTS["jerk"] * norm_jerk
            + WEIGHTS["acc"] * norm_acc
            + WEIGHTS["fc"] * norm_fc
            + WEIGHTS["spo2"] * norm_spo2
        )

        metrics["Fatigue_Score"] = round(float(fatigue), 3)
        metrics["Fatigue_components"] = {
            "norm_fc": round(norm_fc, 3),
            "norm_spo2": round(norm_spo2, 3),
            "norm_acc": round(norm_acc, 3),
            "norm_jerk": round(norm_jerk, 3),
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
            df = centre_accelerations(df)
            df = compute_acceleration_magnitudes(df)
            df = compute_translational_velocity(
                df,
                default_fs=DEFAULT_FS,
                cutoff=DEFAULT_HP_CUTOFF,
            )
            df = compute_jerk(df)

            session_metrics = compute_session_metrics(df)
            session_metrics = compute_fatigue_score(session_metrics)

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
