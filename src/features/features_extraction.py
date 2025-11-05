"""
Sliding-window feature extraction (pipeline stage 4/5).

- Generates overlapping windows (default 5 s, 50% overlap) over enriched sensor data.
- Computes robust statistics for acceleration, translational velocity, jerk, HR, SpO₂ and fatigue score.
- Tracks per-window quality metrics (sample count, NaN ratios, duration).
- Joins the resulting features with the RPE mapping (runner/session metadata).
- Saves the consolidated dataset under `data/results/` (configurable via CLI).

Input: `data/enriched/enriched_*.parquet` + `data/raw/rpe_file_mapping.csv`
Output: `data/results/features_dataset_5s_50olap.parquet`
Next stage: analysis scripts under `src/analysis/`.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root on sys.path when executed directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.kinematics import DEFAULT_VTR_SMOOTHING
from src.data.metrics import compute_fatigue_score

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
RAW_DIR = os.path.join(DATA_DIR, "raw")
MAPPING_PATH = os.path.join(RAW_DIR, "rpe_file_mapping.csv")
DEFAULT_OUTPUT = os.path.join(RESULTS_DIR, "features_dataset_5s_50olap.parquet")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ======================================================================
# STATISTICAL UTILITIES
# ======================================================================
def _mad(x: np.ndarray) -> float:
    "Median absolute deviation; high values indicate abrupt movement."
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def _iqr(x: np.ndarray) -> float:
    "Interquartile range; higher values reflect greater variability."
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)

def _rms(x: np.ndarray) -> float:
    "Root-mean-square magnitude summarising overall signal energy."
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(x**2)))

def _skew(x: np.ndarray) -> float:
    "Distribution skewness; deviations may reveal asymmetries in running form."
    x = x[~np.isnan(x)]
    if x.size < 3:
        return np.nan
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))

def _kurtosis(x: np.ndarray) -> float:
    "Excess kurtosis; high values can signal sharp peaks or impacts."
    x = x[~np.isnan(x)]
    if x.size < 4:
        return np.nan
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    if s == 0:
        return -3.0  # Fisher definition: normal distribution -> 0 (converted to -3 here).
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _nanmin(x: np.ndarray) -> float:
    "Safe minimum that returns NaN if no valid samples exist."
    x = x[~np.isnan(x)]
    return float(x.min()) if x.size else np.nan


def _nanmax(x: np.ndarray) -> float:
    "Safe maximum that returns NaN if no valid samples exist."
    x = x[~np.isnan(x)]
    return float(x.max()) if x.size else np.nan

# ======================================================================
# WINDOW-LEVEL FEATURE COMPUTATION
# ======================================================================
def compute_window_features(df_win: pd.DataFrame, file_id: str, source_file: str) -> Dict:
    """
    Compute statistics for an already segmented window (df_win).
    Returns a dictionary containing features and metadata.
    """
    out: Dict[str, float] = {}

    # Window metadata
    t0 = float(df_win["Relative_Time"].min())
    t1 = float(df_win["Relative_Time"].max())
    duration = t1 - t0 if np.isfinite(t1) else np.nan

    out["file"] = file_id
    out["source_file"] = source_file
    out["start_s"] = t0
    out["end_s"] = t1
    out["duration"] = max(duration, 0.0) if np.isfinite(duration) else np.nan
    out["n_samples"] = int(len(df_win))

    def _nan_fraction(col: str) -> float:
        return float(df_win[col].isna().mean()) if col in df_win.columns else np.nan

    # Centered accelerations
    for axis in ["X", "Y", "Z"]:
        col = f"Acc{axis}_centered"
        if col in df_win.columns:
            x = df_win[col].to_numpy(dtype=float)
            out[f"{col}_mean"] = np.nanmean(x)
            out[f"{col}_std"] = np.nanstd(x, ddof=1)
            out[f"{col}_mad"] = _mad(x)
            out[f"{col}_min"] = _nanmin(x)
            out[f"{col}_p25"] = np.nanpercentile(x, 25)
            out[f"{col}_median"] = np.nanmedian(x)
            out[f"{col}_p75"] = np.nanpercentile(x, 75)
            out[f"{col}_max"] = _nanmax(x)
            out[f"{col}_iqr"] = _iqr(x)
            out[f"{col}_rms"] = _rms(x)
            out[f"{col}_skew"] = _skew(x)
            out[f"{col}_kurt"] = _kurtosis(x)
            out[f"{col}_nan_frac"] = _nan_fraction(col)

    # Raw acceleration magnitude
    if "Acc_mag" in df_win.columns:
        x = df_win["Acc_mag"].to_numpy(dtype=float)
        out["Acc_mag_mean"] = np.nanmean(x)
        out["Acc_mag_std"] = np.nanstd(x, ddof=1)
        out["Acc_mag_mad"] = _mad(x)
        out["Acc_mag_min"] = _nanmin(x)
        out["Acc_mag_median"] = np.nanmedian(x)
        out["Acc_mag_max"] = _nanmax(x)
        out["Acc_mag_iqr"] = _iqr(x)
        out["Acc_mag_rms"] = _rms(x)
        out["Acc_mag_skew"] = _skew(x)
        out["Acc_mag_kurt"] = _kurtosis(x)
        out["Acc_mag_nan_frac"] = _nan_fraction("Acc_mag")
        out["Acc_mean"] = out.get("Acc_mag_mean")
        out["Acc_std"] = out.get("Acc_mag_std")

    # Translational velocity magnitude
    if "Vtr" in df_win.columns:
        v = df_win["Vtr"].to_numpy(dtype=float)
        out["Vtr_mean"] = np.nanmean(v)
        out["Vtr_std"] = np.nanstd(v, ddof=1)
        out["Vtr_mad"] = _mad(v)
        out["Vtr_min"] = _nanmin(v)
        out["Vtr_p25"] = np.nanpercentile(v, 25)
        out["Vtr_median"] = np.nanmedian(v)
        out["Vtr_p75"] = np.nanpercentile(v, 75)
        out["Vtr_max"] = _nanmax(v)
        out["Vtr_iqr"] = _iqr(v)
        out["Vtr_rms"] = _rms(v)
        out["Vtr_skew"] = _skew(v)
        out["Vtr_kurt"] = _kurtosis(v)
        out["Vtr_nan_frac"] = _nan_fraction("Vtr")

    # Jerk magnitude
    if "jerk_mag" in df_win.columns:
        j = df_win["jerk_mag"].to_numpy(dtype=float)
        out["jerk_mean"] = np.nanmean(j)
        out["jerk_std"] = np.nanstd(j, ddof=1)
        out["jerk_mad"] = _mad(j)
        out["jerk_min"] = _nanmin(j)
        out["jerk_p25"] = np.nanpercentile(j, 25)
        out["jerk_median"] = np.nanmedian(j)
        out["jerk_p75"] = np.nanpercentile(j, 75)
        out["jerk_max"] = _nanmax(j)
        out["jerk_iqr"] = _iqr(j)
        out["jerk_rms"] = _rms(j)
        out["jerk_skew"] = _skew(j)
        out["jerk_kurt"] = _kurtosis(j)
        out["jerk_nan_frac"] = _nan_fraction("jerk_mag")

    # Heart rate (FC)
    if "FC" in df_win.columns:
        f = df_win["FC"].to_numpy(dtype=float)
        out["FC_mean"] = np.nanmean(f)
        out["FC_std"] = np.nanstd(f, ddof=1)
        out["FC_min"] = _nanmin(f)
        out["FC_median"] = np.nanmedian(f)
        out["FC_max"] = _nanmax(f)
        out["FC_nan_frac"] = _nan_fraction("FC")

    # Oxygen saturation (SpO₂)
    if "SpO2" in df_win.columns:
        s = df_win["SpO2"].to_numpy(dtype=float)
        out["SpO2_mean"] = np.nanmean(s)
        out["SpO2_std"] = np.nanstd(s, ddof=1)
        out["SpO2_min"] = _nanmin(s)
        out["SpO2_median"] = np.nanmedian(s)
        out["SpO2_max"] = _nanmax(s)
        out["SpO2_nan_frac"] = _nan_fraction("SpO2")

    # Fatigue score
    if "Fatigue_Score" in df_win.columns:
        fs = df_win["Fatigue_Score"].to_numpy(dtype=float)
        out["Fatigue_mean"] = np.nanmean(fs)
        out["Fatigue_std"] = np.nanstd(fs, ddof=1)
        out["Fatigue_nan_frac"] = _nan_fraction("Fatigue_Score")

    # Compute fatigue score per window using available metrics
    metrics_payload = {}
    fc_mean = out.get("FC_mean")
    if fc_mean is not None and np.isfinite(fc_mean):
        metrics_payload["FC_mean"] = float(fc_mean)
    spo2_mean = out.get("SpO2_mean")
    if spo2_mean is not None and np.isfinite(spo2_mean):
        metrics_payload["SpO2_mean"] = float(spo2_mean)
    acc_std = out.get("Acc_std")
    if acc_std is not None and np.isfinite(acc_std):
        metrics_payload["Acc_std"] = float(acc_std)
    jerk_std = out.get("jerk_std")
    if jerk_std is not None and np.isfinite(jerk_std):
        metrics_payload["jerk_std"] = float(jerk_std)

    if metrics_payload:
        score_dict = compute_fatigue_score(metrics_payload.copy())
        fatigue_score = score_dict.get("Fatigue_Score")
        if fatigue_score is not None and np.isfinite(fatigue_score):
            out["Fatigue_Score_window"] = fatigue_score
            # Backwards-compatible column name expected by downstream scripts/EDA
            out["Fatigue_Score"] = fatigue_score
            components = score_dict.get("Fatigue_components", {})
            for key, value in components.items():
                if value is None or not np.isfinite(value):
                    continue
                out[f"Fatigue_component_{key}_window"] = value
                out[f"Fatigue_component_{key}"] = value

    return out

# ======================================================================
# FILE-LEVEL EXTRACTION
# ======================================================================
def extract_features_from_file(
    fpath: str,
    window: float,
    overlap: float,
    min_samples: int = 5,
    file_id: Optional[str] = None,
) -> List[Dict]:
    """
    Slide windows over a single file and compute features per window.
    """
    try:
        df = pd.read_parquet(fpath)
    except Exception as exc:
        logger.error("Error reading %s: %s", os.path.basename(fpath), exc, exc_info=True)
        return []

    if "Relative_Time" not in df.columns:
        logger.warning("%s does not contain 'Relative_Time'; skipping.", os.path.basename(fpath))
        return []

    df = df.sort_values("Relative_Time").reset_index(drop=True)
    numeric_cols = [
        "AccX_centered", "AccY_centered", "AccZ_centered",
        "Acc_mag", "Vtr", "jerk_mag", "FC", "SpO2", "Fatigue_Score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    t_start = float(df["Relative_Time"].min())
    t_end = float(df["Relative_Time"].max())
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        logger.warning("%s has an invalid time range; skipping.", os.path.basename(fpath))
        return []

    step = window * (1.0 - overlap)
    if step <= 0:
        logger.error("Computed window step is <= 0. Check window/overlap configuration.")
        return []

    feats: List[Dict] = []
    w_start = t_start
    source_file = os.path.basename(fpath)
    file_key = file_id or source_file

    while w_start + window <= t_end + 1e-9:
        w_end = w_start + window
        mask = (df["Relative_Time"] >= w_start) & (df["Relative_Time"] < w_end)
        df_win = df.loc[mask]

        if len(df_win) >= min_samples:
            feats.append(compute_window_features(df_win, file_key, source_file))

        w_start += step

    return feats

# ======================================================================
# PIPELINE UTILS
# ======================================================================
def load_rpe_mapping(path: str) -> pd.DataFrame:
    """Load the RPE mapping file with basic validation."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"RPE mapping not found at: {path}")

    df_map = pd.read_csv(path)
    expected_cols = {"file", "runner_id", "session_id", "reported_rpe", "estimated_rpe"}
    missing = expected_cols - set(df_map.columns)
    if missing:
        raise ValueError(f"Missing columns in rpe_file_mapping.csv: {missing}")
    return df_map

def collect_source_files(source_dir: Optional[str] = None) -> List[str]:
    """Return the list of parquet files to process, preferring data/enriched."""
    # Determine directory priority: explicit -> enriched -> processed
    if source_dir:
        directories = [source_dir]
    else:
        directories = [
            ENRICHED_DIR if os.path.isdir(ENRICHED_DIR) else None,
            PROCESSED_DIR,
        ]
        directories = [d for d in directories if d and os.path.isdir(d)]

    for directory in directories:
        files = sorted(
            f.path for f in os.scandir(directory)
            if f.is_file() and f.name.endswith(".parquet") and (f.name.startswith("enriched_") or f.name.startswith("clean_"))
        )
        if files:
            logger.info("Found %d files in %s", len(files), directory)
            return files

    raise FileNotFoundError("No parquet files found in the configured source directories.")

def run_feature_extraction(
    window: float,
    overlap: float,
    out_path: str,
    source_dir: Optional[str] = None,
) -> str:
    """
    Run the end-to-end feature extraction pipeline.
    """
    df_map = load_rpe_mapping(MAPPING_PATH)
    files = collect_source_files(source_dir=source_dir)

    logger.info("Processing %d files with window=%.2fs overlap=%.2f", len(files), window, overlap)
    all_feats: List[Dict] = []

    for fpath in files:
        source_file = os.path.basename(fpath)
        mapping_key = source_file.replace("enriched_", "clean_", 1) if source_file.startswith("enriched_") else source_file
        feats = extract_features_from_file(
            fpath,
            window=window,
            overlap=overlap,
            file_id=mapping_key,
        )
        if feats:
            all_feats.extend(feats)
        else:
            logger.warning("No features generated for %s", os.path.basename(fpath))

    if not all_feats:
        raise RuntimeError("No features were generated. Check required columns and time ranges.")

    df_feats = pd.DataFrame(all_feats)
    df_out = df_feats.merge(df_map, on="file", how="left")

    missing_rpe = df_out["reported_rpe"].isna().sum() if "reported_rpe" in df_out.columns else len(df_out)
    if missing_rpe:
        logger.warning("Mapping data missing for %d windows; check rpe_file_mapping.csv.", missing_rpe)

    if "reported_rpe" in df_out.columns:
        df_out["fatigue_level"] = pd.cut(
            df_out["reported_rpe"],
            bins=[0, 5, 8, 11],
            labels=["low", "medium", "high"],
            include_lowest=True,
            right=False,
        ).astype(str)

    meta_cols = ["file", "source_file", "runner_id", "session_id", "start_s", "end_s", "duration", "n_samples"]
    label_cols = ["reported_rpe", "estimated_rpe"]
    other_cols = [c for c in df_out.columns if c not in meta_cols + label_cols]
    df_out = df_out[meta_cols + label_cols + other_cols]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if out_path.lower().endswith(".csv"):
        df_out.to_csv(out_path, index=False)
    else:
        df_out.to_parquet(out_path, index=False)

    logger.info("✅ Features saved to %s (%d windows)", out_path, len(df_out))
    return out_path

# ======================================================================
# CLI
# ======================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sliding-window feature extraction for running fatigue analysis."
    )
    parser.add_argument("--window", type=float, default=5.0, help="Window size in seconds (default: 5.0).")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap ratio [0, 1) (default: 0.5).")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output path for the feature dataset.")
    parser.add_argument("--source", type=str, default=None, help="Optional directory to read input parquet files from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.overlap < 1.0):
        raise ValueError("The --overlap parameter must be within [0, 1). Recommended value: 0.5.")

    run_feature_extraction(
        window=args.window,
        overlap=args.overlap,
        out_path=args.output,
        source_dir=args.source,
    )

if __name__ == "__main__":
    main()
