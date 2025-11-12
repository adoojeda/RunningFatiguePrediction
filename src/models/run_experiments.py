"""
Experiment runner for fatigue/RPE modeling.

Responsibilities:
    * Load the curated feature dataset (3 s windows).
    * Apply the feature whitelist.
    * Train/evaluate multiple models (GBDT, RF, ElasticNet, XGBoost/LightGBM/CatBoost if available).
    * Perform hyperparameter search with GroupKFold (via GridSearchCV or Optuna-like loops).
    * Persist metrics, predictions, and serialized models under data/results/modeling/experiments/.

Usage:
    python src/models/run_experiments.py \
        --dataset data/results/features_dataset_3s_50olap.parquet \
        --group runner_id \
        --models gradient_boosting random_forest elasticnet xgboost \
        --output-dir data/results/modeling/experiments \
        --save-predictions \
        --save-models
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBRegressor
except ImportError:  
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:  
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:  
    CatBoostRegressor = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "results" / "features_dataset_3s_50olap.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "modeling" / "experiments"
FEATURE_WHITELIST = [
    "FC_mean",
    "SpO2_mean",
    "Fatigue_Score",
    "Fatigue_component_norm_fc",
    "Fatigue_component_norm_acc",
    "Fatigue_component_norm_jerk",
    "Acc_mean",
    "Acc_std",
    "Acc_mag_mad",
    "Acc_mag_skew",
    "Acc_mag_kurt",
    "AccX_centered_mean",
    "AccX_centered_std",
    "AccX_centered_mad",
    "AccX_centered_skew",
    "AccX_centered_kurt",
    "AccY_centered_mean",
    "AccY_centered_std",
    "AccY_centered_mad",
    "AccY_centered_skew",
    "AccY_centered_kurt",
    "AccZ_centered_mean",
    "AccZ_centered_std",
    "AccZ_centered_mad",
    "AccZ_centered_skew",
    "AccZ_centered_kurt",
    "Vtr_mean",
    "Vtr_std",
    "Vtr_mad",
    "Vtr_skew",
    "Vtr_kurt",
    "jerk_mean",
    "jerk_std",
    "jerk_mad",
    "jerk_skew",
]

META_COLS = {
    "file",
    "source_file",
    "runner_id",
    "session_id",
    "start_s",
    "duration",
    "n_samples",
}
TARGET_SIBLINGS = {"reported_rpe", "fatigue_level"}

@dataclass
class ExperimentConfig:
    dataset: Path
    target: str
    group: str
    test_size: float
    seed: int
    output_dir: Path
    models: List[str]
    save_predictions: bool
    save_models: bool
    whitelist: bool

@dataclass
class FoldResult:
    model: str
    split: str
    fold: Optional[int]
    mae: float
    rmse: float
    r2: float
    samples: int

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment runner for fatigue modeling.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to the feature parquet.")
    parser.add_argument("--target", type=str, default="reported_rpe", help="Target column name.")
    parser.add_argument("--group", type=str, default="runner_id", help="Grouping column for splits.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gradient_boosting", "random_forest", "elasticnet"],
        help="Model types to train (options: gradient_boosting, random_forest, elasticnet, xgboost, lightgbm, catboost).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Results output directory.")
    parser.add_argument("--save-predictions", action="store_true", help="Save test set predictions.")
    parser.add_argument("--save-models", action="store_true", help="Save trained models.")
    parser.add_argument("--no-whitelist", action="store_true", help="Do not use feature whitelist; use all numeric features.")
    return parser.parse_args()

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_parquet(path)
    logger.info("Dataset loaded from %s with %d rows and %d columns.", path, df.shape[0], df.shape[1])
    return df

def prepare_features(df: pd.DataFrame, target: str, use_whitelist: bool) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not present in dataset.")
    df = df.copy()
    y = pd.to_numeric(df[target], errors="coerce")
    df.drop(columns=[target], inplace=True, errors="ignore")
    df.drop(columns=[c for c in META_COLS if c in df.columns], inplace=True, errors="ignore")
    df.drop(columns=[c for c in TARGET_SIBLINGS if c in df.columns], inplace=True, errors="ignore")

    if use_whitelist:
        available = [col for col in FEATURE_WHITELIST if col in df.columns]
        missing = [col for col in FEATURE_WHITELIST if col not in df.columns]
        if missing:
            logger.warning("Missing whitelist columns: %s", missing)
        if available:
            df = df[available]
        else:
            logger.warning("Whitelist columns missing; falling back to all numeric columns.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns remain after filtering.")
    X = df[numeric_cols]
    mask = np.isfinite(y)
    X, y = X.loc[mask], y.loc[mask]
    return X, y

def split_data(X, y, groups, *, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    logger.info(
        "Data split: %d train samples (%d groups), %d test samples (%d groups).",
        len(train_idx),
        groups.iloc[train_idx].nunique(),
        len(test_idx),
        groups.iloc[test_idx].nunique(),
    )
    return train_idx, test_idx

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, split: str, fold: Optional[int]) -> FoldResult:
    return FoldResult(
        model=model_name,
        split=split,
        fold=fold,
        mae=mean_absolute_error(y_true, y_pred),
        rmse=mean_squared_error(y_true, y_pred, squared=False),
        r2=r2_score(y_true, y_pred),
        samples=len(y_true),
    )

def make_numeric_pipeline(model) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return Pipeline(
        steps=[
            ("features", ColumnTransformer([("num", numeric, slice(0, None))], remainder="drop")),
            ("model", model),
        ]
    )

def build_model_grid(name: str, seed: int):
    if name == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=seed)
        param_grid = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 4],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
        }
    elif name == "random_forest":
        model = RandomForestRegressor(random_state=seed, n_jobs=-1)
        param_grid = {"model__n_estimators": [200, 400], "model__max_depth": [None, 10], "model__min_samples_leaf": [2, 4]}
    elif name == "elasticnet":
        model = ElasticNet(random_state=seed, max_iter=10000)
        param_grid = {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]}
    elif name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed.")
        model = XGBRegressor(
            random_state=seed,
            n_estimators=400,
            tree_method="hist",
            n_jobs=-1,
        )
        param_grid = {"model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1], "model__subsample": [0.8, 1.0]}
    elif name == "lightgbm":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed.")
        model = LGBMRegressor(random_state=seed)
        param_grid = {"model__n_estimators": [400, 800], "model__learning_rate": [0.05, 0.1], "model__num_leaves": [31, 63]}
    elif name == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed.")
        model = CatBoostRegressor(
            random_state=seed,
            verbose=False,
        )
        param_grid = {"model__depth": [6, 8], "model__learning_rate": [0.03, 0.1], "model__iterations": [300, 600]}
    else:
        raise ValueError(f"Model '{name}' no soportado.")
    pipeline = make_numeric_pipeline(model)
    return pipeline, param_grid

def run_experiment(cfg: ExperimentConfig) -> None:
    df = load_dataset(cfg.dataset)
    groups = df[cfg.group].astype(str)
    X, y = prepare_features(df, cfg.target, cfg.whitelist)
    if len(X) != len(groups):
        groups = groups.loc[X.index]

    train_idx, test_idx = split_data(X, y, groups, test_size=cfg.test_size, seed=cfg.seed)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = cfg.output_dir / f"experiment_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    feature_list = X.columns.tolist()
    with open(exp_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_list, f, indent=2)

    all_results: List[FoldResult] = []
    predictions_store: Dict[str, pd.DataFrame] = {}

    for model_name in cfg.models:
        logger.info("=== Model %s ===", model_name)
        pipeline, param_grid = build_model_grid(model_name, cfg.seed)
        gkf = GroupKFold(n_splits=min(5, groups_train.nunique()))
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=gkf,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train, groups=groups_train)
        logger.info("%s -> best hyperparameters: %s", model_name, search.best_params_)

        for fold, (train_idx_cv, val_idx_cv) in enumerate(gkf.split(X_train, y_train, groups_train), start=1):
            pipeline_best = search.best_estimator_
            pipeline_best.fit(X_train.iloc[train_idx_cv], y_train.iloc[train_idx_cv])
            val_pred = pipeline_best.predict(X_train.iloc[val_idx_cv])
            all_results.append(
                evaluate(y_train.iloc[val_idx_cv], val_pred, model_name=model_name, split="cv", fold=fold)
            )

        # Test metrics
        best_pipeline = search.best_estimator_
        best_pipeline.fit(X_train, y_train)
        test_pred = best_pipeline.predict(X_test)
        all_results.append(evaluate(y_test, test_pred, model_name=model_name, split="test", fold=None))

        if cfg.save_predictions:
            pred_df = pd.DataFrame(
                {
                    "file": df.loc[X_test.index, "file"] if "file" in df.columns else X_test.index,
                    "group": groups.loc[X_test.index],
                    "y_true": y_test,
                    "y_pred": test_pred,
                    "model": model_name,
                }
            )
            predictions_store[model_name] = pred_df

        if cfg.save_models:
            model_path = exp_dir / f"{model_name}_best.joblib"
            joblib.dump(best_pipeline, model_path)
            logger.info("Model saved to %s", model_path)

    results_df = pd.DataFrame([asdict(res) for res in all_results])
    results_path = exp_dir / "metrics.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Metrics saved to %s", results_path)

    summary = (
        results_df.groupby(["model", "split"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            samples_total=("samples", "sum"),
        )
        .reset_index()
    )
    summary_path = exp_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Summary saved to %s", summary_path)

    cfg_payload = asdict(cfg)
    cfg_payload["models"] = cfg.models
    cfg_payload["timestamp"] = timestamp
    for key, value in list(cfg_payload.items()):
        if isinstance(value, Path):
            cfg_payload[key] = str(value)
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_payload, f, indent=2, ensure_ascii=False)

    if cfg.save_predictions and predictions_store:
        pred_concat = pd.concat(predictions_store.values(), ignore_index=True)
        pred_path = exp_dir / "predictions.parquet"
        pred_concat.to_parquet(pred_path, index=False)
        logger.info("Predictions saved to %s", pred_path)

def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        dataset=args.dataset,
        target=args.target,
        group=args.group,
        test_size=args.test_size,
        seed=args.seed,
        output_dir=args.output_dir,
        models=args.models,
        save_predictions=args.save_predictions,
        save_models=args.save_models,
        whitelist=not args.no_whitelist,
    )
    run_experiment(cfg)

if __name__ == "__main__":
    main()
