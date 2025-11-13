"""
Baseline training and evaluation for fatigue/RPE prediction.

Typical usage:
    python src/models/train_baselines.py \
        --dataset data/results/features_dataset_3s_50olap.parquet \
        --target reported_rpe \
        --group session_id

The script:
    - loads the 3 s window dataset generated in stage 4,
    - performs grouped train/test splits (session/runner),
    - trains a handful of simple models,
    - reports MAE, RMSE and R² for cross-validation and hold-out,
    - saves a summary under data/results/modeling/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "results" / "features_dataset_3s_50olap.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "modeling"

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
TARGET_LEAKAGE_MAP = {
    "Fatigue_Score": [
        "Fatigue_component_norm_fc",
        "Fatigue_component_norm_acc",
        "Fatigue_component_norm_jerk",
    ],
    "fatigue_level": [
        "Fatigue_Score",
        "Fatigue_component_norm_fc",
        "Fatigue_component_norm_acc",
        "Fatigue_component_norm_jerk",
    ],
}

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

@dataclass
class MetricsReport:
    model: str
    fold: Optional[int]
    split: str
    mae: float
    rmse: float
    r2: float
    samples: int

@dataclass(frozen=True)
class TrainingConfig:
    dataset: Path
    target: str
    group: str
    test_size: float
    cv_folds: int
    seed: int
    output_dir: Path
    models: Optional[List[str]] = None
    save_predictions: bool = False
    use_whitelist: bool = True

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline training and evaluation for fatigue/RPE prediction.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to the feature parquet (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="reported_rpe",
        help="Target column to predict.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="session_id",
        help="Column to use for grouped splitting (e.g., session_id or runner_id).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out test proportion.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of GroupKFold splits (use 1 to disable CV).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save results (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional list of models to run (default: all). Supported: gradient_boosting, random_forest, linear_regression.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save test set predictions for each model.",
    )
    parser.add_argument(
        "--no-feature-whitelist",
        action="store_true",
        help="Use all numeric columns (skip the curated feature whitelist).",
    )
    return parser.parse_args()

def build_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        dataset=args.dataset,
        target=args.target,
        group=args.group,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        seed=args.seed,
        output_dir=args.output_dir,
        models=[m.strip() for m in args.models] if args.models else None,
        save_predictions=args.save_predictions,
        use_whitelist=not args.no_feature_whitelist,
    )

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_parquet(path)
    logger.info("Dataset loaded from %s with shape %s", path, df.shape)
    return df

def prepare_features(
    df: pd.DataFrame,
    target: str,
    drop_meta: bool = True,
    feature_whitelist: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not present in dataset.")

    df = df.copy()
    y = pd.to_numeric(df[target], errors="coerce")
    df = df.drop(columns=[target], errors="ignore")

    if drop_meta:
        df = df.drop(columns=[c for c in META_COLS if c in df.columns], errors="ignore")

    exclude = TARGET_SIBLINGS.difference({target})
    df = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")

    leakage_cols = TARGET_LEAKAGE_MAP.get(target, [])
    if leakage_cols:
        df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore")

    if feature_whitelist:
        available = [col for col in feature_whitelist if col in df.columns]
        missing = [col for col in feature_whitelist if col not in df.columns]
        if missing:
            logger.warning("Whitelist columns missing from dataset and will be skipped: %s", missing)
        if available:
            df = df[available]
        else:
            logger.warning("Whitelist yielded no valid columns; falling back to all numeric columns.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns available for training.")

    X = df[numeric_cols]

    mask = np.isfinite(y)
    X = X.loc[mask]
    y = y.loc[mask]

    logger.info("Final shape: %d numeric columns, %d rows after cleaning.", X.shape[1], X.shape[0])
    return X, y

def select_groups(df: pd.DataFrame, group_col: str, fallback: str = "file") -> pd.Series:
    if group_col in df.columns:
        groups = df[group_col].astype(str)
    elif fallback in df.columns:
        logger.warning("Column '%s' not found; falling back to '%s' for grouping.", group_col, fallback)
        groups = df[fallback].astype(str)
    else:
        raise KeyError(f"No suitable grouping column found ('{group_col}' or '{fallback}').")
    return groups

def build_models(random_state: int) -> Dict[str, Pipeline]:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    def make_pipeline(model) -> Pipeline:
        return Pipeline(
            steps=[
                ("features", ColumnTransformer(transformers=[("num", numeric_transformer, slice(0, None))], remainder="drop")),
                ("model", model),
            ]
        )

    models = {
        "gradient_boosting": make_pipeline(
            GradientBoostingRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                min_samples_leaf=3,
                random_state=random_state,
            )
        ),
        "random_forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=random_state,
            )
        ),
        "linear_regression": make_pipeline(
            LinearRegression()
        ),
    }
    return models

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model_name: str,
    split: str,
    fold: Optional[int],
) -> MetricsReport:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return MetricsReport(model=model_name, fold=fold, split=split, mae=mae, rmse=rmse, r2=r2, samples=len(y_true))

def group_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    logger.info(
        "Group split: %d train samples (%d groups), %d test samples (%d groups).",
        len(train_idx),
        groups.iloc[train_idx].nunique(),
        len(test_idx),
        groups.iloc[test_idx].nunique(),
    )
    return train_idx, test_idx

def compute_cv_reports(
    model_name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    folds: int,
) -> List[MetricsReport]:
    if folds <= 1:
        return []
    unique_groups = groups.nunique()
    if unique_groups < folds:
        logger.warning(
            "Number of groups (%d) is smaller than n_splits (%d). Reducing to %d.",
            unique_groups,
            folds,
            unique_groups,
        )
        folds = unique_groups
        if folds <= 1:
            logger.warning("Not enough groups for CV; skipping cross-validation.")
            return []
    reports: List[MetricsReport] = []
    gkf = GroupKFold(n_splits=folds)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        val_pred = pipeline.predict(X.iloc[val_idx])
        report = evaluate(
            y_true=y.iloc[val_idx],
            y_pred=val_pred,
            model_name=model_name,
            split="cv",
            fold=fold,
        )
        reports.append(report)
        logger.info(
            "[%s][Fold %d/%d] CV -> MAE: %.3f | RMSE: %.3f | R²: %.3f",
            model_name,
            fold,
            folds,
            report.mae,
            report.rmse,
            report.r2,
        )

    return reports

def feature_importance(pipeline: Pipeline, feature_names: List[str]) -> Optional[List[Tuple[str, float]]]:
    model = pipeline.named_steps.get("model")
    if model is None:
        return None

    attrs: Optional[np.ndarray] = None
    if hasattr(model, "feature_importances_"):
        attrs = model.feature_importances_
    elif hasattr(model, "coef_"):
        coeffs = model.coef_
        attrs = np.abs(np.asarray(coeffs)) if coeffs is not None else None

    if attrs is None:
        return None

    if attrs.shape[0] != len(feature_names):
        logger.debug("Importance vector length (%d) differs from #features (%d).", attrs.shape[0], len(feature_names))
        return None

    top = sorted(zip(feature_names, attrs), key=lambda x: x[1], reverse=True)
    return top[:20]

def save_reports(
    reports: List[MetricsReport],
    feature_ranking: Dict[str, List[Tuple[str, float]]],
    output_dir: Path,
    summary_df: Optional[pd.DataFrame] = None,
) -> Path:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"baseline_results_{timestamp}.json"

    payload = {
        "generated_at": timestamp,
        "reports": [asdict(rep) for rep in reports],
        "feature_importance": {name: [(feat, float(score)) for feat, score in ranking] for name, ranking in feature_ranking.items()},
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if summary_df is not None:
        summary_path = output_dir / f"baseline_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Summary saved in %s", summary_path)

    logger.info("Results saved in %s", json_path)
    return json_path

def summarise_reports(reports: List[MetricsReport]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(rep) for rep in reports])
    if df.empty:
        return df
    summary = (
        df.groupby(["model", "split"])
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
    summary.fillna(0.0, inplace=True)
    return summary

def save_predictions(
    predictions: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> Optional[Path]:
    if not predictions:
        return None
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"baseline_predictions_{timestamp}.parquet"
    df_all = pd.concat(predictions.values(), ignore_index=True)
    df_all.to_parquet(path, index=False)
    logger.info("Predictions saved in %s", path)
    return path

def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    df = load_dataset(cfg.dataset)
    groups = select_groups(df, cfg.group)
    X, y = prepare_features(
        df,
        target=cfg.target,
        feature_whitelist=FEATURE_WHITELIST if cfg.use_whitelist else None,
    )

    if len(df) != len(X):
        groups = groups.loc[X.index]
    feature_names = X.columns.tolist()

    train_idx, test_idx = group_split(X, y, groups, test_size=cfg.test_size, seed=cfg.seed)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    models = build_models(random_state=cfg.seed)
    if cfg.models:
        missing = [m for m in cfg.models if m not in models]
        if missing:
            raise ValueError(f"Models not recognized: {missing}. Supported models: {list(models.keys())}")
        models = {name: models[name] for name in cfg.models}

    reports: List[MetricsReport] = []
    feature_rankings: Dict[str, List[Tuple[str, float]]] = {}
    predictions_store: Dict[str, pd.DataFrame] = {}

    for name, pipeline in models.items():
        logger.info("=== Model: %s ===", name)
        cv_reports = compute_cv_reports(name, pipeline, X_train, y_train, groups_train, cfg.cv_folds)
        reports.extend(cv_reports)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_report = evaluate(y_true=y_test, y_pred=y_pred, model_name=name, split="test", fold=None)
        reports.append(test_report)
        logger.info("[ %s ] Test -> MAE: %.3f | RMSE: %.3f | R²: %.3f", name, test_report.mae, test_report.rmse, test_report.r2)

        ranking = feature_importance(pipeline, feature_names)
        if ranking:
            feature_rankings[name] = ranking
            logger.info("Top features (%s): %s", name, ", ".join(f"{feat} ({score:.3f})" for feat, score in ranking[:5]))

        if cfg.save_predictions:
            pred_df = pd.DataFrame(
                {
                    "index": X_test.index,
                    "y_true": y_test.values,
                    "y_pred": y_pred,
                    "model": name,
                }
            )
            predictions_store[name] = pred_df

    summary = summarise_reports(reports)
    save_reports(reports, feature_rankings, cfg.output_dir, summary_df=summary)
    if cfg.save_predictions:
        save_predictions(predictions_store, cfg.output_dir)

if __name__ == "__main__":
    main()
