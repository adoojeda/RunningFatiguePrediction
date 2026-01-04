"""
Ejecutor de experimentos para el modelado de cansancio físico y RPE.

Responsabilidades:
    * Cargar el dataset de características (ventanas de 3 s).
    * Aplicar la whitelist de variables.
    * Entrenar y evaluar varios modelos.
    * Buscar hiperparámetros con GroupKFold mediante GridSearchCV.
    * Guardar métricas, predicciones y modelos en data/results/modeling/experiments/.

Ejemplo de uso:
    python src/models/run_experiments.py \
        --dataset data/results/features_dataset.parquet \
        --target physical_fatigue_index \
        --group runner_id \
        --models gradient_boosting random_forest hist_gradient_boosting elasticnet xgboost catboost \
        --output-dir data/results/modeling/experiments
"""

# LIBRERÍAS 
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hashlib
import joblib # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor # type: ignore
from sklearn.linear_model import ElasticNet # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, max_error # type: ignore
from sklearn.model_selection import GroupKFold, GridSearchCV, GroupShuffleSplit # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.impute import SimpleImputer # type: ignore

try:
    from xgboost import XGBRegressor # type: ignore
except ImportError:  
    XGBRegressor = None

os.environ["LIGHTGBM_NO_DASK"] = "1"

try:
    from catboost import CatBoostRegressor # type: ignore
except ImportError:  
    CatBoostRegressor = None

# CONFIGURACIÓN DE LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# RUTAS Y WHITELIST DE CARACTERÍSTICAS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "results" / "features_dataset.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "modeling" / "experiments"
FEATURE_WHITELIST = [
    # Frecuencia cardíaca y SpO2
    "hr_mean", "spo2_mean",
    # Aceleración
    "acc_mean", "acc_std", "acc_mag_mad", "acc_mag_skew", "acc_mag_kurt",
    "acc_x_centered_mean", "acc_x_centered_std", "acc_x_centered_mad", "acc_x_centered_skew", "acc_x_centered_kurt",
    "acc_y_centered_mean", "acc_y_centered_std", "acc_y_centered_mad", "acc_y_centered_skew", "acc_y_centered_kurt",
    "acc_z_centered_mean", "acc_z_centered_std", "acc_z_centered_mad", "acc_z_centered_skew", "acc_z_centered_kurt",
    # Velocidad 
    "vtr_mean", "vtr_std", "vtr_mad", "vtr_skew", "vtr_kurt",
    # Jerk
    "jerk_mean", "jerk_std", "jerk_mad", "jerk_skew",
    # Giroscopio
    "roll_mean", "roll_std", "roll_mad", "roll_skew", "roll_kurt",
    "yaw_mean", "yaw_std", "yaw_mad", "yaw_skew", "yaw_kurt",
    "grav_x_mean", "grav_x_std", "grav_x_mad", "grav_x_skew", "grav_x_kurt",
    "grav_y_mean", "grav_y_std", "grav_y_mad", "grav_y_skew", "grav_y_kurt",
    "grav_z_mean", "grav_z_std", "grav_z_mad", "grav_z_skew", "grav_z_kurt",
]

# CONFIGURACIÓN DE METADATOS Y TARGET
META_COLS = {"file","source_file","runner_id","session_id","age","sex","start_s","duration","n_samples"}
TARGET_SIBLINGS = {"reported_rpe", "fatigue_level"}
TARGET_LEAKAGE_MAP = {"physical_fatigue_index": [],"fatigue_level": ["physical_fatigue_index"]}

# CONSTRUCCIÓN DE LA SERIE DE AGRUPACIÓN
def build_group_series(df: pd.DataFrame, spec: str) -> pd.Series:
    """Construye una serie de agrupación a partir de una especificación."""
    columns = spec.split("+")
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Grouping columns not found in the dataset: {missing}")
    series = df[columns[0]].astype(str)
    for col in columns[1:]:
        series = series + "__" + df[col].astype(str)
    return series

# CONFIGURACIÓN DE DATACLASS
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
    n_jobs: int
    fast_grid: bool

@dataclass
class FoldResult:
    model: str
    split: str
    fold: Optional[int]
    mae: float
    rmse: float
    r2: float
    med_ae: float
    max_err: float
    samples: int

# PARSING DE ARGUMENTOS
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecutor de experimentos para el modelado del cansancio físico.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Ruta al parquet de características.")
    parser.add_argument("--target", type=str, default="physical_fatigue_index", help="Nombre de la columna objetivo.")
    parser.add_argument("--group", type=str, default="runner_id", help="Columna de agrupación (o expresión con '+').")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción del split de test.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gradient_boosting", "random_forest", "hist_gradient_boosting", "elasticnet", "xgboost", "catboost"],
        help=(
            "Modelos a entrenar (disponibles: gradient_boosting, random_forest, hist_gradient_boosting, "
            "elasticnet, xgboost, catboost)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directorio de salida.")
    parser.add_argument("--no-save-predictions", action="store_true", help="No guardar predicciones de test.")
    parser.add_argument("--no-save-models", action="store_true", help="No guardar modelos entrenados.")
    parser.add_argument("--no-whitelist", action="store_true", help="Usar todas las columnas numéricas (ignorar whitelist).")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Paralelismo para GridSearchCV/entrenamiento.")
    parser.add_argument("--fast-grid", action="store_true", help="Usar grid reducido para pruebas rápidas.")
    return parser.parse_args()

# CARGA DEL DATASET
def load_dataset(path: Path) -> pd.DataFrame:
    """Carga el dataset desde un archivo parquet."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {path}")
    df = pd.read_parquet(path)
    logger.info("Dataset loaded from %s with %d rows and %d columns.", path, df.shape[0], df.shape[1])
    return df

def compute_file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Computa el hash MD5 del archivo dado."""
    md5 = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()

# PREPARACIÓN DE CARACTERÍSTICAS
def prepare_features(df: pd.DataFrame, target: str, use_whitelist: bool) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepara las características y el objetivo para el modelado."""
    if target not in df.columns:
        raise KeyError(f"La columna objetivo '{target}' no está presente en el dataset.")
    df = df.copy()
    y = pd.to_numeric(df[target], errors="coerce")
    df.drop(columns=[target], inplace=True, errors="ignore")
    df.drop(columns=[c for c in META_COLS if c in df.columns], inplace=True, errors="ignore")
    df.drop(columns=[c for c in TARGET_SIBLINGS if c in df.columns], inplace=True, errors="ignore")

    leakage_cols = TARGET_LEAKAGE_MAP.get(target, [])
    if leakage_cols:
        df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True, errors="ignore")

    if use_whitelist:
        available = [col for col in FEATURE_WHITELIST if col in df.columns]
        missing = [col for col in FEATURE_WHITELIST if col not in df.columns]
        if missing:
            logger.warning("Faltan columnas de la whitelist: %s", missing)
        if available:
            df = df[available]
        else:
            logger.warning("No hay columnas de la whitelist; se usan todas las numéricas.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No quedan columnas numéricas tras el filtrado.")
    X = df[numeric_cols]
    mask = np.isfinite(y)
    X, y = X.loc[mask], y.loc[mask]
    return X, y

# PARTICIONADO Y EVALUACIÓN
def split_data(X, y, groups, *, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Realiza un split estratificado por grupos."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    logger.info(
        "Split: %d muestras train (%d grupos), %d muestras test (%d grupos).",
        len(train_idx),
        groups.iloc[train_idx].nunique(),
        len(test_idx),
        groups.iloc[test_idx].nunique(),
    )
    return train_idx, test_idx

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, split: str, fold: Optional[int]) -> FoldResult:
    """Calcula métricas de evaluación para las predicciones dadas."""
    return FoldResult(
        model=model_name,
        split=split,
        fold=fold,
        mae=mean_absolute_error(y_true, y_pred),
        rmse=mean_squared_error(y_true, y_pred, squared=False),
        r2=r2_score(y_true, y_pred),
        med_ae=median_absolute_error(y_true, y_pred),
        max_err=max_error(y_true, y_pred),
        samples=len(y_true),
    )

# CONSTRUCCIÓN DEL PIPELINE NUMÉRICO
def make_numeric_pipeline(model) -> Pipeline:
    """Construye un pipeline para datos numéricos con imputación y escalado."""
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

# MODELOS Y GRIDS DE HIPERPARÁMETROS
def build_model_grid(
    name: str,
    seed: int,
    n_jobs: int,
    fast_grid: bool,
):
    """Construye el modelo y el grid de hiperparámetros según el nombre dado."""
    n_jobs = 1 if n_jobs == 0 else n_jobs
    if name == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=seed)
        param_grid = (
            {
                "model__n_estimators": [200],
                "model__max_depth": [3],
                "model__learning_rate": [0.1],
                "model__subsample": [1.0],
            }
            if fast_grid
            else {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 4],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
            }
        )
    elif name == "random_forest":
        model = RandomForestRegressor(random_state=seed, n_jobs=n_jobs)
        param_grid = (
            {"model__n_estimators": [200], "model__max_depth": [None], "model__min_samples_leaf": [2]}
            if fast_grid
            else {"model__n_estimators": [200, 400], "model__max_depth": [None, 10], "model__min_samples_leaf": [2, 4]}
        )
    elif name == "elasticnet":
        model = ElasticNet(random_state=seed, max_iter=10000)
        param_grid = (
            {"model__alpha": [0.1], "model__l1_ratio": [0.5]}
            if fast_grid
            else {"model__alpha": [0.01, 0.1, 1.0, 10.0], "model__l1_ratio": [0.1, 0.5, 0.9, 0.99]}
        )
    elif name == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(random_state=seed)
        param_grid = (
            {"model__learning_rate": [0.05], "model__max_depth": [None], "model__max_leaf_nodes": [31]}
            if fast_grid
            else {
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [None, 8],
                "model__max_leaf_nodes": [31, 63],
            }
        )
    elif name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost no está instalado.")
        model = XGBRegressor(
            random_state=seed,
            n_estimators=400,
            tree_method="hist",
            n_jobs=n_jobs,
        )
        param_grid = (
            {"model__max_depth": [3], "model__learning_rate": [0.1], "model__subsample": [1.0]}
            if fast_grid
            else {"model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1], "model__subsample": [0.8, 1.0]}
        )
    elif name == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("catboost no está instalado.")
        model = CatBoostRegressor(
            random_state=seed,
            verbose=False,
            allow_writing_files=False,
        )
        param_grid = (
            {"model__depth": [6], "model__learning_rate": [0.03], "model__iterations": [300]}
            if fast_grid
            else {"model__depth": [6, 8], "model__learning_rate": [0.03, 0.1], "model__iterations": [300, 600]}
        )
    else:
        raise ValueError(f"Modelo '{name}' no soportado.")
    pipeline = make_numeric_pipeline(model)
    return pipeline, param_grid

# EJECUCIÓN DEL EXPERIMENTO
def run_experiment(cfg: ExperimentConfig) -> None:
    """Ejecuta el experimento de modelado según la configuración dada."""
    df = load_dataset(cfg.dataset)
    dataset_hash = compute_file_hash(cfg.dataset)
    logger.info("Running experiment with grouping spec '%s'", cfg.group)
    groups = build_group_series(df, cfg.group).astype(str)
    X, y = prepare_features(df, cfg.target, cfg.whitelist)
    if len(X) != len(groups):
        groups = groups.loc[X.index]

    train_idx, test_idx = split_data(X, y, groups, test_size=cfg.test_size, seed=cfg.seed)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_slug = cfg.group.replace("+", "_plus_").replace("/", "_")
    exp_dir = cfg.output_dir / f"{group_slug}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    feature_list = X.columns.tolist()
    with open(exp_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_list, f, indent=2)
    (exp_dir / "dataset_hash.txt").write_text(dataset_hash)

    all_results: List[FoldResult] = []
    predictions_store: Dict[str, pd.DataFrame] = {}

    for model_name in cfg.models:
        if model_name == "xgboost" and XGBRegressor is None:
            logger.warning("xgboost no está instalado; se omite el modelo.")
            continue
        if model_name == "catboost" and CatBoostRegressor is None:
            logger.warning("catboost no está instalado; se omite el modelo.")
            continue
        logger.info("=== Model %s ===", model_name)
        pipeline, param_grid = build_model_grid(model_name, cfg.seed, cfg.n_jobs, cfg.fast_grid)
        gkf = GroupKFold(n_splits=min(5, groups_train.nunique()))
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=gkf,
            scoring="neg_mean_absolute_error",
            n_jobs=cfg.n_jobs,
            verbose=1,
        )
        search.fit(X_train, y_train, groups=groups_train)
        logger.info("%s -> mejores hiperparámetros: %s", model_name, search.best_params_)

        for fold, (train_idx_cv, val_idx_cv) in enumerate(gkf.split(X_train, y_train, groups_train), start=1):
            pipeline_best = search.best_estimator_
            pipeline_best.fit(X_train.iloc[train_idx_cv], y_train.iloc[train_idx_cv])
            val_pred = pipeline_best.predict(X_train.iloc[val_idx_cv])
            all_results.append(
                evaluate(y_train.iloc[val_idx_cv], val_pred, model_name=model_name, split="cv", fold=fold)
            )

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
                    "group_spec": cfg.group,
                }
            )
            predictions_store[model_name] = pred_df

        if cfg.save_models:
            model_path = exp_dir / f"{model_name}_best.joblib"
            joblib.dump(best_pipeline, model_path)
            logger.info("Modelo guardado en %s", model_path)

    results_df = pd.DataFrame([asdict(res) for res in all_results])
    results_path = exp_dir / "metrics.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Métricas guardadas en %s", results_path)

    summary = (
        results_df.groupby(["model", "split"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
             med_ae_mean=("med_ae", "mean"),
             med_ae_std=("med_ae", "std"),
             max_err_mean=("max_err", "mean"),
             max_err_std=("max_err", "std"),
            samples_total=("samples", "sum"),
        )
        .reset_index()
    )
    summary_path = exp_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Resumen guardado en %s", summary_path)

    cfg_payload = asdict(cfg)
    cfg_payload["models"] = cfg.models
    cfg_payload["timestamp"] = timestamp
    cfg_payload["dataset_hash"] = dataset_hash
    for key, value in list(cfg_payload.items()):
        if isinstance(value, Path):
            cfg_payload[key] = str(value)
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_payload, f, indent=2, ensure_ascii=False)

    if cfg.save_predictions and predictions_store:
        pred_concat = pd.concat(predictions_store.values(), ignore_index=True)
        pred_path = exp_dir / "predictions.parquet"
        pred_concat.to_parquet(pred_path, index=False)
        logger.info("Predicciones guardadas en %s", pred_path)

# FUNCIÓN PRINCIPAL
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
        save_predictions=not args.no_save_predictions,
        save_models=not args.no_save_models,
        whitelist=not args.no_whitelist,
        n_jobs=args.n_jobs,
        fast_grid=args.fast_grid,
    )
    run_experiment(cfg)

# PUNTO DE ENTRADA
if __name__ == "__main__":
    main()
