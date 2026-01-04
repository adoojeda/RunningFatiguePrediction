"""
Ejecuta experimentos excluyendo bloques de características (ablación).

Ejemplo:

    python src/models/run_ablation.py \
        --dataset data/results/features_dataset.parquet \
        --target physical_fatigue_index \
        --group runner_id \
        --models gradient_boosting hist_gradient_boosting \
        --exclude-blocks orientation physiology
"""

# LIBRERÍAS 
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Sequence
import fnmatch

import pandas as pd # type: ignore

# CONFIGURACIÓN DEL PROYECTO
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# IMPORTS DEL PROYECTO
from src.models.run_experiments import (
    ExperimentConfig,
    load_dataset,
    run_experiment,
)

# CONFIGURACIÓN DE LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET = PROJECT_ROOT / "data" / "results" / "features_dataset.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "results" / "modeling" / "ablation"

# BLOQUES DE CARACTERÍSTICAS
FEATURE_BLOCKS = {
    "physiology": ["hr_*", "spo2_*"],
    "accelerations": ["acc_*"],
    "velocity": ["vtr_*"],
    "jerk": ["jerk_*"],
    "orientation": ["roll_*", "yaw_*", "grav_*"],
}

# PARSING DE ARGUMENTOS
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta experimentos excluyendo bloques de características.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Ruta al dataset base.")
    parser.add_argument("--target", type=str, default="physical_fatigue_index", help="Columna objetivo.")
    parser.add_argument("--group", type=str, default="runner_id", help="Columna de agrupación.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gradient_boosting", "random_forest", "hist_gradient_boosting", "elasticnet", "xgboost", "catboost"],
        help="Modelos a entrenar.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción del split de test.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Paralelismo para GridSearchCV.")
    parser.add_argument("--fast-grid", action="store_true", help="Usar grids reducidos de hiperparámetros.")
    parser.add_argument("--no-save-predictions", action="store_true", help="No guardar predicciones.")
    parser.add_argument("--no-save-models", action="store_true", help="No guardar modelos entrenados.")
    parser.add_argument("--no-whitelist", action="store_true", help="Usar todas las columnas numéricas (sin whitelist).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directorio base de resultados.")
    parser.add_argument(
        "--exclude-blocks",
        nargs="*",
        default=[],
        help=f"Bloques a excluir ({', '.join(FEATURE_BLOCKS.keys())}).",
    )
    return parser.parse_args()

# FUNCIONES DE ABLACIÓN
def expand_patterns(columns: Sequence[str], patterns: Sequence[str]) -> List[str]:
    """Expande patrones con comodines a nombres de columnas."""
    selected: List[str] = []
    for pat in patterns:
        if "*" in pat or "?" in pat:
            selected.extend([col for col in columns if fnmatch.fnmatch(col, pat)])
        else:
            if pat in columns:
                selected.append(pat)
    return selected

def remove_blocks(df: pd.DataFrame, blocks: Sequence[str]) -> pd.DataFrame:
    """Elimina bloques de características del DataFrame."""
    if not blocks:
        return df
    invalid = [b for b in blocks if b not in FEATURE_BLOCKS]
    if invalid:
        raise ValueError(f"Bloques de características desconocidos: {invalid}")
    drop_cols: List[str] = []
    for block in blocks:
        patterns = FEATURE_BLOCKS[block]
        drop_cols.extend(expand_patterns(df.columns, patterns))
    drop_cols = sorted(set(drop_cols))
    logger.info("Se eliminaron %d columnas (%s).", len(drop_cols), ", ".join(drop_cols))
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# CONSTRUCCIÓN DEL DIRECTORIO DE SALIDA
def build_output_dir(base: Path, blocks: Sequence[str]) -> Path:
    """Construye el directorio de salida basado en los bloques excluidos."""
    if not blocks:
        return base / "baseline"
    tag = "no_" + "_".join(sorted(blocks))
    return base / tag

# FUNCIÓN PRINCIPAL
def main() -> None:
    args = parse_args()
    df = load_dataset(args.dataset)
    df_filtered = remove_blocks(df, args.exclude_blocks)

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dataset.parquet"
        df_filtered.to_parquet(tmp_path, index=False)

        output_dir = build_output_dir(args.output_dir, args.exclude_blocks)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = ExperimentConfig(
            dataset=tmp_path,
            target=args.target,
            group=args.group,
            test_size=args.test_size,
            seed=args.seed,
            output_dir=output_dir,
            models=args.models,
            save_predictions=not args.no_save_predictions,
            save_models=not args.no_save_models,
            whitelist=not args.no_whitelist,
            n_jobs=args.n_jobs,
            fast_grid=args.fast_grid,
        )
        run_experiment(cfg)

# EJECUCIÓN DEL SCRIPT
if __name__ == "__main__":
    main()
