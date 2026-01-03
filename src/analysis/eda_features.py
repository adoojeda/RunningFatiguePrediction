"""
Análisis exploratorio (EDA) del dataset por ventanas.

Genera estadísticas descriptivas y gráficos que relacionan métricas biomecánicas/fisiológicas
con el esfuerzo percibido (RPE). Permite filtrar por sesión/métrica y exportar figuras en un ZIP.

Ejemplo:
    python src/analysis/eda_features.py --dataset data/results/features_dataset.parquet --output data/results/eda_figures
"""

# LIBRERÍAS 
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
from zipfile import ZipFile

import matplotlib # type: ignore

matplotlib.use("Agg")  

import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy import stats # type: ignore

# CONFIGURACIÓN DEL PROYECTO
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# IMPORTS DEL PROYECTO
from src.utils.data_loader import load_features_dataset
from src.utils.eda_plots import (
    plot_correlation_heatmap,
    plot_fatigue_levels,
    plot_rpe_relationships,
    plot_runner_facets,
    plot_specialised_relationships,
)

# CONFIGURACIÓN DE LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# DIRECTORIOS
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
DEFAULT_DATASET = os.path.join(RESULTS_DIR, "features_dataset.parquet")
DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "eda_figures")

# MÉTRICAS CANDIDATAS
RPE_METRIC_CANDIDATES = [
    "hr_mean",
    "physical_fatigue_index",
    "jerk_std",
    "acc_mean",
    "acc_std",
    "acc_mag_mad",
    "vtr_mean",
    "spo2_mean",
]

# PARES PARA CÁLCULO DE P-VALUES
P_VALUE_PAIRS: List[tuple[str, str]] = [
    ("hr_mean", "physical_fatigue_index"),
    ("hr_mean", "reported_rpe"),
    ("hr_mean", "vtr_mean"),
    ("hr_mean", "jerk_std"),
    ("physical_fatigue_index", "reported_rpe"),
    ("physical_fatigue_index", "vtr_mean"),
    ("physical_fatigue_index", "jerk_std"),
    ("reported_rpe", "vtr_mean"),
    ("reported_rpe", "jerk_std"),
    ("vtr_mean", "jerk_std"),
]

# CÁLCULO DE P-VALUES
def compute_pvalues(
    df: pd.DataFrame,
    output_dir: str,
    pairs: Optional[List[tuple[str, str]]] = None,
) -> Optional[str]:
    """Calcula los p-values de Pearson para pares de métricas y guarda en CSV."""
    pairs = pairs or P_VALUE_PAIRS
    records: List[Dict[str, float]] = []
    for x, y in pairs:
        if x not in df.columns or y not in df.columns:
            logger.warning("No se puede calcular el p-value para %s vs %s; faltan columnas.", x, y)
            continue
        sample = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sample) < 3:
            logger.warning("No hay suficientes muestras para %s vs %s; se omite.", x, y)
            continue
        try:
            r, p = stats.pearsonr(sample[x], sample[y])
            records.append(
                {
                    "x": x,
                    "y": y,
                    "n": len(sample),
                    "pearson_r": r,
                    "p_value": p,
                }
            )
        except Exception as exc:
            logger.warning("Error al calcular p-value para %s vs %s: %s", x, y, exc)

    if not records:
        logger.warning("No se generaron p-values; revisa el dataset y las columnas.")
        return None

    df_p = pd.DataFrame(records)
    path = os.path.join(output_dir, "pvalues.csv")
    df_p.to_csv(path, index=False)
    logger.info("Tabla de p-values guardada en %s", path)
    return path

def _available_numeric_metrics(df: pd.DataFrame) -> List[str]:
    """Devuelve una lista de métricas numéricas disponibles en el DataFrame."""
    present = [col for col in RPE_METRIC_CANDIDATES if col in df.columns]
    if present:
        return present
    return df.select_dtypes(include=[np.number]).columns.tolist()

# FUNCIÓN PRINCIPAL DEL EDA
def run_eda(
    dataset_path: str,
    output_dir: str,
    sessions: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    zip_name: Optional[str] = None,
    clean_after_zip: bool = False,
) -> Optional[str]:
    timestamped_dir = os.path.join(output_dir, datetime.now().strftime("eda_%Y%m%d_%H%M%S"))
    os.makedirs(timestamped_dir, exist_ok=True)
    df = load_features_dataset(path=dataset_path)
    if df is None or df.empty:
        raise ValueError("El dataset de características no se pudo cargar o está vacío.")

    if sessions:
        filters = set(str(s) for s in sessions)
        mask = pd.Series(False, index=df.index)
        for col in ("session_id", "file", "source_file", "runner_id"):
            if col in df.columns:
                mask |= df[col].astype(str).isin(filters)
        df = df[mask]
        if df.empty:
            raise ValueError("No quedan filas tras aplicar los filtros de sesión.")
        logger.info("Dataset filtrado a %d filas usando sesiones %s", len(df), sorted(filters))

    metric_list = [m.strip() for m in metrics] if metrics else None
    available_metrics = _available_numeric_metrics(df)
    generated_files: List[str] = []

    generated_files.extend(
        plot_rpe_relationships(
            df,
            timestamped_dir,
            metrics=metric_list,
            available_metrics=available_metrics,
        )
    )
    generated_files.extend(plot_specialised_relationships(df, timestamped_dir))
    heatmap = plot_correlation_heatmap(df, timestamped_dir)
    if heatmap:
        generated_files.append(heatmap)
    generated_files.extend(
        plot_runner_facets(
            df,
            timestamped_dir,
            metrics=available_metrics,
        )
    )
    level_box = plot_fatigue_levels(df, timestamped_dir)
    if level_box:
        generated_files.append(level_box)
    pvalue_file = compute_pvalues(df, timestamped_dir, pairs=P_VALUE_PAIRS)
    if pvalue_file:
        generated_files.append(pvalue_file)

    seen = set()
    unique_files = []
    for path in generated_files:
        if path and os.path.exists(path) and path not in seen:
            seen.add(path)
            unique_files.append(path)

    zip_path: Optional[str] = None
    if zip_name:
        base = zip_name if zip_name != "auto" else f"eda_report_{datetime.now():%Y%m%d_%H%M%S}"
        if not base.endswith(".zip"):
            base = f"{base}.zip"
        zip_path = os.path.join(os.path.dirname(timestamped_dir), base)
        with ZipFile(zip_path, "w") as archive:
            for file_path in unique_files:
                arcname = os.path.relpath(file_path, timestamped_dir) if file_path.startswith(timestamped_dir) else os.path.basename(file_path)
                archive.write(file_path, arcname=arcname)
        logger.info("Figuras comprimidas en %s", zip_path)
        if clean_after_zip:
            for file_path in unique_files:
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            try:
                os.rmdir(timestamped_dir)
            except OSError:
                logger.warning("No se pudo eliminar el directorio %s", timestamped_dir)
    else:
        zip_path = None

    logger.info("EDA completado. Figuras guardadas en %s", timestamped_dir)
    return zip_path

# PARSEAR ARGUMENTOS
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA del dataset de características por ventanas."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Ruta al dataset de características (por defecto: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directorio donde guardar las figuras (por defecto: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--session",
        nargs="+",
        help="Lista opcional de sesiones/archivos/corredores para filtrar el dataset.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Lista opcional de métricas a resaltar (aplica a gráficos y RPE).",
    )
    parser.add_argument(
        "--zip",
        nargs="?",
        const="auto",
        default=None,
        help="Crea un ZIP con las figuras generadas. Puede especificarse un nombre.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Elimina las figuras tras crear el ZIP (solo si se usa --zip).",
    )
    return parser.parse_args()

# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    args = parse_args()
    try:
        zip_path = run_eda(
            dataset_path=args.dataset,
            output_dir=args.output,
            sessions=args.session,
            metrics=args.metrics,
            zip_name=args.zip,
            clean_after_zip=args.clean,
        )
        if zip_path:
            logger.info("ZIP del EDA creado en %s", zip_path)
    except Exception as exc:
        logger.error("El proceso de EDA falló: %s", exc)
