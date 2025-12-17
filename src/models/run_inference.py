"""
Near real-time inference utility.

Loads a trained pipeline (e.g., gradient boosting) and applies it to an enriched
session. Window extraction is reused so the predictions are aligned with the
training pipeline and can be replayed window by window, emulating online
monitoring.

Example usage:
    python src/models/run_inference.py \
        --enriched data/enriched/enriched_<file>.parquet \
        --experiment data/results/modeling/experiments/runner_id_YYYYMMDD_HHMMSS \
        --model gradient_boosting \
        --output data/results/modeling/inference/demo_predictions.parquet
"""

# STANDARD LIBRARIES
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PROJECT SETUP
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PROJECT LIBRARIES
from src.config import get_config
from src.features.features_extraction import extract_features_from_file

# LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# GLOBAL VARIABLES
CFG = get_config()
DEFAULT_MODEL = "gradient_boosting"
DEFAULT_EXPERIMENT = PROJECT_ROOT / "data" / "results" / "modeling" / "experiments"

# HELPER FUNCTIONS
def _load_pipeline(experiment_dir: Path, model_name: str):
    """Load the trained pipeline and feature list."""
    model_path = experiment_dir / f"{model_name}_best.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    pipeline = joblib.load(model_path)
    features_path = experiment_dir / "feature_columns.json"
    if features_path.exists():
        feature_columns = json.loads(features_path.read_text())
    else:
        feature_columns = None
        logger.warning("feature_columns.json not found; inferring columns from dataframe.")

    return pipeline, feature_columns

def _prepare_feature_matrix(df: pd.DataFrame, feature_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, List[str]]:
    """Select and order the columns expected by the trained pipeline."""
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in {"file", "source_file", "start_s", "duration", "n_samples"}]
        logger.info("Inferred feature list with %d columns.", len(feature_columns))

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        logger.warning("Missing %d columns in the input data: %s", len(missing), missing)
    X = df.reindex(columns=feature_columns)
    return X, feature_columns

def _infer_single_session(
    enriched_path: Path,
    window: float,
    overlap: float,
) -> pd.DataFrame:
    """Generate window-level features from an enriched session."""
    feats = extract_features_from_file(
        str(enriched_path),
        window=window,
        overlap=overlap,
        file_id=enriched_path.name.replace("enriched_", "clean_", 1),
    )
    if not feats:
        raise RuntimeError(f"No windows extracted from {enriched_path}.")
    df = pd.DataFrame(feats)
    df.sort_values("start_s", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def _simulate_stream(pred_df: pd.DataFrame, playback_speed: float) -> None:
    """
    Iterate through predictions emulating near real-time updates.

    playback_speed = 0   -> no wait (instantaneous)
    playback_speed = 1.0 -> real duration
    playback_speed = 4.0 -> four times faster
    """
    last_start = None
    for row in pred_df.itertuples(index=False):
        msg = (
            f"[t={row.start_s:6.2f}s] pred={row.fatigue_pred:.3f}"
            + (f" | score={row.fatigue_score:.3f}" if not np.isnan(getattr(row, "fatigue_score", np.nan)) else "")
        )
        logger.info(msg)

        if playback_speed > 0 and last_start is not None:
            delta = max(0.0, row.start_s - last_start)
            if delta > 0:
                time.sleep(delta / playback_speed)
        last_start = row.start_s

def _summarize(pred_df: pd.DataFrame) -> Dict[str, float]:
    """Compute MAE/RMSE/R2 if a reference fatigue_score exists."""
    if "fatigue_score" not in pred_df.columns or pred_df["fatigue_score"].isna().all():
        return {}

    y_true = pred_df["fatigue_score"].to_numpy()
    y_pred = pred_df["fatigue_pred"].to_numpy()
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    return metrics

# MAIN INFERENCE FUNCTION
def run_inference(args: argparse.Namespace) -> Path:
    enriched_path = Path(args.enriched).resolve()
    experiment_dir = Path(args.experiment).resolve()
    if not enriched_path.exists():
        raise FileNotFoundError(f"Session not found: {enriched_path}")
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    pipeline, feature_columns = _load_pipeline(experiment_dir, args.model)
    df_windows = _infer_single_session(enriched_path, args.window, args.overlap)
    X, feature_columns = _prepare_feature_matrix(df_windows, feature_columns)

    df_windows["fatigue_pred"] = pipeline.predict(X)

    metrics = _summarize(df_windows)
    if metrics:
        logger.info("Evaluation (window-level fatigue_score) -> MAE=%.4f RMSE=%.4f R2=%.4f",
                    metrics["mae"], metrics["rmse"], metrics["r2"])
    else:
        logger.info("No reference fatigue_score; predictions only.")

    if args.playback_speed is not None and args.playback_speed >= 0:
        _simulate_stream(df_windows, args.playback_speed)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".csv":
            df_windows.to_csv(out_path, index=False)
        else:
            df_windows.to_parquet(out_path, index=False)
        logger.info("Predictions saved to %s", out_path)
        return out_path

    return Path()

# ARGPARSE SETUP
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a trained model to an enriched session and replay predictions window by window."
    )
    parser.add_argument(
        "--enriched",
        required=True,
        help="Path to the enriched session file (parquet/csv).",
    )
    parser.add_argument(
        "--experiment",
        default=str(DEFAULT_EXPERIMENT),
        help="Directory containing the trained model and feature list."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Name of the trained model to load (e.g., gradient_boosting).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=CFG.windows.size_seconds,
        help="Window duration in seconds.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=CFG.windows.overlap_ratio,
        help="Window overlap ratio (0-1).",
    )
    parser.add_argument(
        "--output",
        help="Path to save the predictions (parquet/csv). If not provided, predictions are not saved.",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=0.0,
        help="Playback factor (0 = instant, 1 = real-time, 4 = four times faster).",
    )
    return parser.parse_args()

# MAIN ENTRY POINT
def main() -> None:
    args = parse_args()
    run_inference(args)

if __name__ == "__main__":
    main()
