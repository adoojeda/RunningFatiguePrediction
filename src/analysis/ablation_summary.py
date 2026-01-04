"""
Resumen de resultados de ablaciones.

Ejemplo:
    python src/analysis/ablation_summary.py --ablation-dir data/results/modeling/ablation

Recorre subdirectorios bajo data/results/modeling/ablation/<tag>,
combina los archivos summary.csv más recientes en un solo CSV, y calcula deltas
respecto a la ejecución baseline.
"""

# LIBRERÍAS 
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd # type: ignore

# CONFIGURACIÓN DE LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# CONFIGURACIÓN DEL PROYECTO
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ABLATION_DIR = PROJECT_ROOT / "data" / "results" / "modeling" / "ablation"

# PARSEO DE ARGUMENTOS
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera un resumen combinado de las ablaciones.")
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR, help="Directorio base con ejecuciones de ablación.")
    parser.add_argument("--output", type=Path, default=None, help="Ruta del CSV combinado (opcional).")
    parser.add_argument("--deltas-output", type=Path, default=None, help="Ruta del CSV de deltas vs baseline (opcional).")
    return parser.parse_args()

# RECOGIDA DE SUMMARYS
def collect_summary(tag_dir: Path) -> Optional[pd.DataFrame]:
    """Recoge el summary.csv más reciente en un directorio de etiqueta."""
    if not tag_dir.exists():
        return None
    experiment_dirs = sorted([d for d in tag_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for exp_dir in experiment_dirs:
        summary_path = exp_dir / "summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df["experiment"] = exp_dir.name
            return df
    return None

# FUNCIÓN PRINCIPAL
def main() -> None:
    args = parse_args()
    if not args.ablation_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio de ablaciones: {args.ablation_dir}")

    combined: List[pd.DataFrame] = []
    for tag_dir in sorted([d for d in args.ablation_dir.iterdir() if d.is_dir()]):
        df = collect_summary(tag_dir)
        if df is None:
            logger.warning("summary.csv no encontrado en %s", tag_dir)
            continue
        df["block_tag"] = tag_dir.name
        combined.append(df)

    if not combined:
        raise RuntimeError("No se encontraron resultados de ablación.")

    combined_df = pd.concat(combined, ignore_index=True)
    combined_df_path = args.output or (args.ablation_dir / "ablation_summary.csv")
    combined_df.to_csv(combined_df_path, index=False)
    logger.info("Resumen combinado guardado en %s", combined_df_path)

    baseline_df = combined_df[combined_df["block_tag"] == "baseline"]
    if baseline_df.empty:
        logger.warning("Baseline no encontrado; se omiten los deltas.")
        return

    deltas: List[pd.DataFrame] = []
    baseline_key = ["model", "split"]
    for tag in combined_df["block_tag"].unique():
        if tag == "baseline":
            continue
        df_tag = combined_df[combined_df["block_tag"] == tag]
        merged = pd.merge(
            df_tag,
            baseline_df,
            on=baseline_key,
            suffixes=("", "_baseline"),
            how="inner",
        )
        for metric in ["mae_mean", "rmse_mean", "r2_mean"]:
            merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_baseline"]
        merged["block_tag"] = tag
        deltas.append(merged)

    if deltas:
        deltas_df = pd.concat(deltas, ignore_index=True)
        deltas_path = args.deltas_output or (args.ablation_dir / "ablation_deltas.csv")
        deltas_df.to_csv(deltas_path, index=False)
        logger.info("Deltas respecto a baseline guardados en %s", deltas_path)
    else:
        logger.warning("No se pudieron calcular deltas; revisa baseline y otras ejecuciones.")

# EJECUCIÓN DEL SCRIPT
if __name__ == "__main__":
    main()
