"""
Análisis exploratorio de datos (EDA) para el dataset de ventanas deslizantes.

Genera estadísticas descriptivas y una serie de gráficas que vinculan métricas
biomecánicas/fisiológicas con el esfuerzo percibido (RPE). Permite filtrar por
sesión o métrica y exportar los resultados como un informe ZIP.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Optional
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")  

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils.data_loader import load_features_dataset

# CONFIGURACIÓN DEL LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# CONFIGURACIÓN DE RUTAS
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
DEFAULT_DATASET = os.path.join(RESULTS_DIR, "features_dataset.parquet")
DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "eda_figures")

AXIS_LABELS: Dict[str, str] = {
    "vtr_mean": "Velocidad de traslación media",
    "hr_mean": "Frecuencia cardíaca media",
    "fatigue_score": "Índice de fatiga",
    "jerk_std": "Desviación estándar del jerk",
    "reported_rpe": "RPE reportado",
    "session_id": "ID de sesión",
    "runner_id": "ID de corredor",
    "fatigue_level": "Nivel de fatiga",
    "start_s": "Tiempo (s)",
}

RPE_RELATIONSHIPS: Dict[str, str] = {
    "vtr_mean": "",
    "hr_mean": "",
    "jerk_std": "",
    "fatigue_score": "",
}

P_VALUE_PAIRS: List[tuple[str, str]] = [
    ("hr_mean", "fatigue_score"),
    ("hr_mean", "reported_rpe"),
    ("hr_mean", "vtr_mean"),
    ("hr_mean", "jerk_std"),
    ("fatigue_score", "reported_rpe"),
    ("fatigue_score", "vtr_mean"),
    ("fatigue_score", "jerk_std"),
    ("reported_rpe", "vtr_mean"),
    ("reported_rpe", "jerk_std"),
    ("vtr_mean", "jerk_std"),
]

FACET_FEATURES = ["hr_mean", "fatigue_score"]

def axis_label(name: str) -> str:
    """Devuelve la etiqueta legible para un eje dado."""
    return AXIS_LABELS.get(name, name)

# UTILIDADES PARA GRAFICAR
def safe_boxplot(df: pd.DataFrame, x: str, y: str, output_dir: str) -> Optional[str]:
    if y not in df.columns or x not in df.columns:
        logger.warning("Se omite el boxplot de %s vs %s; falta la columna.", y, x)
        return None

    data = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty or data[x].nunique() == 0:
        logger.warning("Se omite el boxplot de %s vs %s; no hay muestras válidas.", y, x)
        return None

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=x, y=y, data=data, palette="coolwarm")
    plt.title(f"{axis_label(y)} vs {axis_label(x)}")
    plt.xlabel(axis_label(x))
    plt.ylabel(axis_label(y))
    plt.grid(alpha=0.3)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(output_dir, f"box_{y}_vs_{x}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Boxplot guardado: %s", path)
    return path

def safe_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str],
    output_dir: str,
    palette: str = "viridis",
) -> Optional[str]:
    if x not in df.columns or y not in df.columns:
        logger.warning("Se omite el scatter %s vs %s; faltan columnas obligatorias.", x, y)
        return None

    data_cols = [x, y]
    if hue and hue in df.columns:
        data_cols.append(hue)
    data = df[data_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        logger.warning("Se omite el scatter %s vs %s; no hay muestras válidas.", x, y)
        return None

    plt.figure(figsize=(7, 5))
    hue_arg = hue if hue and hue in data.columns else None
    sns.scatterplot(x=x, y=y, hue=hue_arg, data=data, palette=palette, alpha=0.6)
    sns.regplot(x=x, y=y, data=data, scatter=False, color="black", ci=None)
    plt.title(f"{axis_label(y)} vs {axis_label(x)}")
    plt.xlabel(axis_label(x))
    plt.ylabel(axis_label(y))
    plt.grid(alpha=0.3)
    if hue_arg:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(output_dir, f"scatter_{x}_vs_{y}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Diagrama de dispersión guardado: %s", path)
    return path

# FUNCIONES DE GRAFICADO DEL EDA

def plot_rpe_relationships(df: pd.DataFrame, output_dir: str, metrics: Optional[Iterable[str]] = None) -> List[str]:
    comments = RPE_RELATIONSHIPS
    _ensure_columns(df, ["reported_rpe"])
    selected = list(metrics) if metrics else list(comments)
    paths: List[str] = []
    for variable in selected:
        if variable in df.columns and variable in comments:
            saved = safe_boxplot(df, x="reported_rpe", y=variable, output_dir=output_dir)
            if saved:
                paths.append(saved)
        elif variable not in df.columns:
            logger.warning("La métrica de relación con RPE %s no existe en el dataset.", variable)
    return paths

def plot_specialised_relationships(df: pd.DataFrame, output_dir: str) -> List[str]:
    _ensure_columns(df, ["reported_rpe"])
    paths: List[str] = []
    saved = safe_scatter(
        df,
        x="hr_mean",
        y="reported_rpe",
        hue="session_id" if "session_id" in df.columns else None,
        output_dir=output_dir,
    )
    if saved:
        paths.append(saved)
    saved = safe_boxplot(
        df,
        x="reported_rpe",
        y="jerk_std",
        output_dir=output_dir,
    )
    if saved:
        paths.append(saved)
    saved = safe_scatter(
        df,
        x="fatigue_score",
        y="reported_rpe",
        hue="session_id" if "session_id" in df.columns else None,
        output_dir=output_dir,
        palette="flare",
    )
    if saved:
        paths.append(saved)
    return paths

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> Optional[str]:
    """Genera un mapa de calor de correlaciones para las variables clave."""
    cols = ["hr_mean", "fatigue_score", "reported_rpe", "vtr_mean", "jerk_std"]
    subset = [c for c in cols if c in df.columns]
    if len(subset) < 2:
        logger.warning("No hay suficientes columnas para el heatmap de correlación.")
        return None
    data = df[subset].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        logger.warning("No hay datos válidos para el heatmap de correlación.")
        return None
    corr = data.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
    plt.title("Mapa de calor de correlaciones clave")
    path = os.path.join(output_dir, "heatmap_correlaciones.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Heatmap de correlaciones guardado en %s", path)
    return path

def plot_runner_facets(df: pd.DataFrame, output_dir: str, max_groups: int = 6) -> List[str]:
    """Boxplots por corredor para métricas clave."""
    if "runner_id" not in df.columns:
        logger.warning("No se pueden generar facetados por corredor; falta runner_id.")
        return []
    runner_counts = df["runner_id"].value_counts().head(max_groups).index
    df_subset = df[df["runner_id"].isin(runner_counts)]
    if df_subset.empty:
        return []
    paths: List[str] = []
    for metric in FACET_FEATURES:
        if metric not in df_subset.columns:
            continue
        plt.figure(figsize=(8, 4))
        sns.boxplot(
            data=df_subset,
            x="runner_id",
            y=metric,
            palette="Set2",
        )
        plt.title(f"{axis_label(metric)} por corredor (top {len(runner_counts)})")
        plt.xlabel("ID de corredor (subset)")
        plt.ylabel(axis_label(metric))
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = os.path.join(output_dir, f"runner_box_{metric}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        logger.info("Boxplots por corredor guardados para %s", metric)
        paths.append(path)
    return paths

def plot_fatigue_levels(df: pd.DataFrame, output_dir: str) -> Optional[str]:
    """Distribución del fatigue_score por nivel discreto."""
    if "fatigue_level" not in df.columns or "fatigue_score" not in df.columns:
        logger.warning("No se puede graficar por nivel de fatiga; faltan columnas.")
        return None
    data = df[["fatigue_level", "fatigue_score"]].dropna()
    if data.empty:
        return None
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=data, x="fatigue_level", y="fatigue_score", palette="Set3")
    plt.title("Distribución del fatigue_score por nivel")
    plt.xlabel(axis_label("fatigue_level"))
    plt.ylabel(axis_label("fatigue_score"))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "fatigue_level_boxplot.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Boxplot por nivel de fatiga guardado en %s", path)
    return path
def compute_pvalues(
    df: pd.DataFrame,
    output_dir: str,
    pairs: Optional[List[tuple[str, str]]] = None,
) -> Optional[str]:
    """Calcula p-values de correlación (Pearson) para los pares indicados."""
    pairs = pairs or P_VALUE_PAIRS
    records: List[Dict[str, float]] = []
    for x, y in pairs:
        if x not in df.columns or y not in df.columns:
            logger.warning("No se puede calcular p-value para %s vs %s; columnas ausentes.", x, y)
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
            logger.warning("Error calculando p-value para %s vs %s: %s", x, y, exc)

    if not records:
        logger.warning("No se generaron p-values; revisa datos y columnas.")
        return None

    df_p = pd.DataFrame(records)
    path = os.path.join(output_dir, "pvalues.csv")
    df_p.to_csv(path, index=False)
    logger.info("Tabla de p-values guardada en %s", path)
    return path

# ORQUESTACIÓN
def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Lanza ValueError si falta alguna columna requerida."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from dataset: {missing}")

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
        raise ValueError("El dataset de características no pudo cargarse o está vacío.")

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
    generated_files: List[str] = []

    generated_files.extend(plot_rpe_relationships(df, timestamped_dir, metrics=metric_list))
    generated_files.extend(plot_specialised_relationships(df, timestamped_dir))
    heatmap = plot_correlation_heatmap(df, timestamped_dir)
    if heatmap:
        generated_files.append(heatmap)
    generated_files.extend(plot_runner_facets(df, timestamped_dir))
    level_box = plot_fatigue_levels(df, timestamped_dir)
    if level_box:
        generated_files.append(level_box)
    pvalue_file = compute_pvalues(df, timestamped_dir, pairs=P_VALUE_PAIRS)
    if pvalue_file:
        generated_files.append(pvalue_file)

    # Elimina duplicados preservando el orden
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
        logger.info("ZIP con el informe generado en %s", zip_path)
        if clean_after_zip:
            for file_path in unique_files:
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            try:
                os.rmdir(timestamped_dir)
            except OSError:
                logger.warning("No se pudo eliminar el directorio %s tras la limpieza.", timestamped_dir)
    else:
        zip_path = None

    logger.info("EDA completado. Figuras almacenadas en %s", timestamped_dir)
    return zip_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análisis exploratorio de datos para el dataset de ventanas deslizantes."
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
        help=f"Directorio donde guardar las figuras generadas (por defecto: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--session",
        nargs="+",
        help="Identificadores opcionales de sesión (session_id/file/source_file) para incluir en el EDA.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Lista opcional de columnas de métricas a destacar (aplica a histogramas y gráficas de RPE).",
    )
    parser.add_argument(
        "--zip",
        nargs="?",
        const="auto",
        default=None,
        help="Crea un ZIP con las figuras generadas. Se puede indicar el nombre del archivo.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Elimina las figuras tras crear el ZIP (sólo si se usa --zip).",
    )
    return parser.parse_args()

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
            logger.info("Informe comprimido disponible en %s", zip_path)
    except Exception as exc:
        logger.error("El EDA falló: %s", exc, exc_info=True)
