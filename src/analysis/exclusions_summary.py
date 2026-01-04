"""
Genera un resumen de columnas excluidas comparando el dataset con la whitelist.

Ejemplo de uso:
    python src/analysis/exclusions_summary.py \
        --dataset data/results/features_dataset.parquet \
        --output data/results/exclusions_summary.csv
"""

# LIBRERÍAS 
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd # type: ignore

# RUTAS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.models.run_experiments import FEATURE_WHITELIST, META_COLS
except ImportError as exc:  # pragma: no cover - fallback defensivo
    raise ImportError("No se pudo importar FEATURE_WHITELIST/META_COLS desde run_experiments.py") from exc

try:
    from src.analysis.feature_audit import TARGET_COLS_BASE
except ImportError:
    TARGET_COLS_BASE = {"reported_rpe", "physical_fatigue_index", "fatigue_level"}

# FUNCIONES
def build_exclusions_summary(dataset_path: Path) -> pd.DataFrame:
    """Devuelve un DataFrame con columnas excluidas del entrenamiento."""
    df = pd.read_parquet(dataset_path)
    no_elegibles = set(META_COLS) | set(TARGET_COLS_BASE)
    excluded = sorted(set(df.columns) - set(FEATURE_WHITELIST) - no_elegibles)
    return pd.DataFrame({"columna_excluida": excluded})

# PARSEO DE ARGUMENTOS
def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Resumen de columnas excluidas (whitelist vs dataset).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/results/features_dataset.parquet"),
        help="Ruta al dataset de características (parquet).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/exclusions_summary.csv"),
        help="Ruta del CSV de salida.",
    )
    return parser.parse_args()

# PUNTO DE ENTRADA PRINCIPAL
def main() -> None:
    """Punto de entrada principal."""
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {args.dataset}")

    summary_df = build_exclusions_summary(args.dataset)
    summary_df.to_csv(args.output, index=False)
    print(f"Resumen guardado en: {args.output} ({len(summary_df)} columnas excluidas)")


if __name__ == "__main__":
    main()
