"""
Optuna-based search for fatigue score weights using the window-level feature dataset.

Strategy:
    - Recompute `fatigue_score` per window with candidate weights over jerk/acc/fc/spo2.
    - Train a lightweight model (HistGradientBoostingRegressor, fast grid) with GroupKFold CV.
    - Evaluate R² on a grouped hold-out split (runner_id) and maximise it.
    - Report best weights and save study results.

Usage:
    python src/models/optimize_fatigue_weights.py \
        --dataset data/results/features_dataset.parquet \
        --trials 20 \
        --output-dir data/results/modeling/weight_search
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd
import sys
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import r2_score

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.utils.metrics_utils import compute_fatigue_score

# ===================
# LOGGING SETUP
# ===================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===================
# HELPER FUNCTIONS
# ===================
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_parquet(path)
    logger.info("Dataset loaded: %s (%d rows, %d cols)", path, df.shape[0], df.shape[1])
    return df

def recompute_fatigue(df: pd.DataFrame, weights: Dict[str, float], refs: Dict[str, float]) -> pd.Series:
    """Recompute fatigue_score per row using the provided weights and references."""
    needed = ["fc_mean", "spo2_mean", "acc_std", "jerk_std"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for fatigue recomputation: {missing}")
    scores = []
    for _, row in df[needed].iterrows():
        metrics_payload = {
            "fc_mean": float(row["fc_mean"]) if np.isfinite(row["fc_mean"]) else 0.0,
            "spo2_mean": float(row["spo2_mean"]) if np.isfinite(row["spo2_mean"]) else 0.0,
            "acc_std": float(row["acc_std"]) if np.isfinite(row["acc_std"]) else 0.0,
            "jerk_std": float(row["jerk_std"]) if np.isfinite(row["jerk_std"]) else 0.0,
        }
        score_dict = compute_fatigue_score(
            metrics_payload,
            context="window",
            references=refs,
            weights=weights,
        )
        scores.append(score_dict.get("fatigue_score", np.nan))
    return pd.to_numeric(pd.Series(scores), errors="coerce")

def train_eval(df: pd.DataFrame, target_col: str, group_col: str, seed: int) -> float:
    """Train a small HistGB model and return hold-out R²."""
    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    groups = df[group_col]
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups.iloc[train_idx]

    model = HistGradientBoostingRegressor(random_state=seed, learning_rate=0.05, max_depth=None, max_leaf_nodes=31)
    gkf = GroupKFold(n_splits=min(3, groups_train.nunique()))
    r2_cv = []
    for train_cv, val_cv in gkf.split(X_train, y_train, groups_train):
        model.fit(X_train.iloc[train_cv], y_train[train_cv])
        preds = model.predict(X_train.iloc[val_cv])
        r2_cv.append(r2_score(y_train[val_cv], preds))
    r2_cv_mean = float(np.mean(r2_cv))

    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    r2_test = r2_score(y_test, preds_test)
    logger.info("R² CV=%.3f | R² test=%.3f", r2_cv_mean, r2_test)
    return r2_test

def objective_factory(df: pd.DataFrame, refs: Dict[str, float], seed: int = 42, group_col: str = "runner_id"):
    def objective(trial: optuna.Trial) -> float:
        w_jerk = trial.suggest_float("w_jerk", 0.1, 0.6)
        w_acc = trial.suggest_float("w_acc", 0.1, 0.6)
        w_fc = trial.suggest_float("w_fc", 0.1, 0.6)
        w_spo2 = trial.suggest_float("w_spo2", 0.05, 0.4)
        total = w_jerk + w_acc + w_fc + w_spo2
        weights = {
            "jerk": w_jerk / total,
            "acc": w_acc / total,
            "fc": w_fc / total,
            "spo2": w_spo2 / total,
        }

        df_copy = df.copy()
        df_target = df_copy.copy()
        df_target["fatigue_score"] = recompute_fatigue(df_target, weights, refs)
        df_target = df_target.dropna(subset=["fatigue_score"])
        r2 = train_eval(df_target, target_col="fatigue_score", group_col=group_col, seed=seed)
        
        return -r2

    return objective

# ===================
# PARSE ARGS
# ===================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna search for fatigue score weights.")
    parser.add_argument("--dataset", type=Path, default=Path("data/results/features_dataset_3s_50olap.parquet"))
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/modeling/weight_search"))
    return parser.parse_args()

# ===================
# MAIN ENTRYPOINT
# ===================
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    # Keep group column and numeric features for speed
    if "runner_id" not in df.columns:
        raise KeyError("runner_id column is required for grouped evaluation.")
    group_series = df["runner_id"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]
    df["runner_id"] = group_series

    cfg = get_config()
    refs = dict(cfg.fatigue_refs.references)
    objective = objective_factory(df, refs, seed=args.seed)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = study.best_params
    total = best["w_jerk"] + best["w_acc"] + best["w_fc"] + best["w_spo2"]
    best_weights = {
        "jerk": best["w_jerk"] / total,
        "acc": best["w_acc"] / total,
        "fc": best["w_fc"] / total,
        "spo2": best["w_spo2"] / total,
    }
    logger.info("Best R² (neg) %.4f with weights: %s", study.best_value, best_weights)

    results_path = args.output_dir / f"weight_search_{args.trials}_trials.json"
    payload = {
        "best_value": study.best_value,
        "best_weights": best_weights,
        "trials": args.trials,
    }
    results_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved best weights to %s", results_path)

if __name__ == "__main__":
    main()
