"""
Utilidades de gráficos reutilizables para el dataset por ventanas (EDA).
"""

# LIBRERÍAS 
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import seaborn as sns # type: ignore

# CONSTANTES GLOBALES
AXIS_LABELS: Dict[str, str] = {
    "vtr_mean": "Velocidad translacional media",
    "hr_mean": "Frecuencia cardíaca media",
    "physical_fatigue_index": "Índice de cansancio físico",
    "jerk_std": "Desviación estándar del jerk",
    "reported_rpe": "RPE reportado",
    "session_id": "Identificador de sesión",
    "runner_id": "Identificador de corredor",
    "fatigue_level": "Nivel de cansancio",
    "start_s": "Tiempo (s)",
}

# FUNCIONES DE GRÁFICOS
def axis_label(name: str) -> str:
    """Obtiene una etiqueta legible para un nombre de variable dado."""
    return AXIS_LABELS.get(name, name)

def safe_boxplot(df: pd.DataFrame, x: str, y: str, output_dir: str) -> Optional[str]:
    """Genera un boxplot si ambas columnas existen."""
    if y not in df.columns or x not in df.columns:
        return None
    data = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty or data[x].nunique() == 0:
        return None
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=x, y=y, data=data, palette="coolwarm")
    plt.title(f"{axis_label(y)} frente a {axis_label(x)}")
    plt.xlabel(axis_label(x))
    plt.ylabel(axis_label(y))
    plt.grid(alpha=0.3)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(output_dir, f"box_{y}_vs_{x}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

def safe_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_dir: str,
    *,
    hue: Optional[str] = None,
    palette: str = "viridis",
) -> Optional[str]:
    """Genera un scatterplot con regresión si las columnas existen."""
    if x not in df.columns or y not in df.columns:
        return None
    data_cols = [x, y]
    if hue and hue in df.columns:
        data_cols.append(hue)
    data = df[data_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return None
    plt.figure(figsize=(7, 5))
    hue_arg = hue if hue and hue in data.columns else None
    sns.scatterplot(x=x, y=y, hue=hue_arg, data=data, palette=palette, alpha=0.6)
    sns.regplot(x=x, y=y, data=data, scatter=False, color="black", ci=None)
    plt.xlabel(axis_label(x))
    plt.ylabel(axis_label(y))
    plt.grid(alpha=0.3)
    if hue_arg:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(output_dir, f"scatter_{x}_vs_{y}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

def plot_rpe_relationships(
    df: pd.DataFrame,
    output_dir: str,
    metrics: Optional[Iterable[str]] = None,
    *,
    available_metrics: Iterable[str],
) -> List[str]:
    """Genera boxplots de métricas seleccionadas frente a RPE reportado."""
    if "reported_rpe" not in df.columns:
        return []
    selected = list(metrics) if metrics else list(available_metrics)
    paths: List[str] = []
    for variable in selected:
        if variable in df.columns:
            saved = safe_boxplot(df, x="reported_rpe", y=variable, output_dir=output_dir)
            if saved:
                paths.append(saved)
    return paths

def plot_specialised_relationships(df: pd.DataFrame, output_dir: str) -> List[str]:
    """Gráficos especializados para relaciones clave entre métricas."""
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
    saved = safe_boxplot(df, x="reported_rpe", y="jerk_std", output_dir=output_dir)
    if saved:
        paths.append(saved)
    saved = safe_scatter(
        df,
        x="physical_fatigue_index",
        y="reported_rpe",
        hue="session_id" if "session_id" in df.columns else None,
        output_dir=output_dir,
        palette="flare",
    )
    if saved:
        paths.append(saved)
    return paths

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> Optional[str]:
    """Genera un mapa de calor de correlaciones entre métricas clave."""
    cols = ["hr_mean", "spo2_mean", "acc_std", "physical_fatigue_index", "reported_rpe", "vtr_mean", "jerk_std"]
    subset = [c for c in cols if c in df.columns]
    if len(subset) < 2:
        return None
    data = df[subset].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return None
    corr = data.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
    plt.title("Mapa de calor de correlaciones")
    path = os.path.join(output_dir, "heatmap_correlations.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

def plot_runner_facets(
    df: pd.DataFrame,
    output_dir: str,
    metrics: Iterable[str],
    *,
    max_groups: int = 6,
) -> List[str]:
    """Genera boxplots de métricas por corredor (top N corredores)."""
    if "runner_id" not in df.columns:
        return []
    runner_counts = df["runner_id"].value_counts().head(max_groups).index
    df_subset = df[df["runner_id"].isin(runner_counts)]
    if df_subset.empty:
        return []
    paths: List[str] = []
    for metric in metrics:
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
        plt.xlabel("Corredor (subset)")
        plt.ylabel(axis_label(metric))
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = os.path.join(output_dir, f"runner_box_{metric}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths

def plot_fatigue_levels(df: pd.DataFrame, output_dir: str) -> Optional[str]:
    """Genera un boxplot del índice de cansancio físico por nivel de cansancio."""
    if "fatigue_level" not in df.columns or "physical_fatigue_index" not in df.columns:
        return None
    data = df[["fatigue_level", "physical_fatigue_index"]].dropna()
    if data.empty:
        return None
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=data, x="fatigue_level", y="physical_fatigue_index", palette="Set3")
    plt.title("Índice de cansancio físico por nivel")
    plt.xlabel(axis_label("fatigue_level"))
    plt.ylabel(axis_label("physical_fatigue_index"))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "fatigue_level_boxplot.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path
