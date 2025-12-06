"""
Panel interactivo (Dash) para explorar sesiones enriquecidas de running.
Muestra aceleraciones, gravedad, rotaciones, orientación, FC/SpO₂ y velocidad de traslación.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import get_config
from src.features.features_extraction import extract_features_from_file
from src.utils.data_loader import load_data
from src.utils.kinematics import DEFAULT_VTR_SMOOTHING

# CONFIGURACIÓN DEL LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# RUTAS Y CONSTANTES
DATA_DIR = BASE_DIR / "data"
ENRICHED_DIR = DATA_DIR / "enriched"
EXPERIMENTS_DIR = DATA_DIR / "results" / "modeling" / "experiments"
EXCLUDED_PREFIXES = ("all_sessions_metrics", "features_dataset")
DEFAULT_MODEL_NAME = "gradient_boosting"
CFG = get_config()

# Mapeo de columnas legacy/CamelCase a snake_case
COL_RENAME = {
    "Relative_Time": "relative_time",
    "Tiempo_rel": "relative_time",
    "AccX": "acc_x",
    "AccY": "acc_y",
    "AccZ": "acc_z",
    "GravX": "grav_x",
    "GravY": "grav_y",
    "GravZ": "grav_z",
    "RotX": "rot_x",
    "RotY": "rot_y",
    "RotZ": "rot_z",
    "Roll": "roll",
    "Pitch": "pitch",
    "Yaw": "yaw",
    "FC": "hr",
    "HR": "hr",
    "SpO2": "spo2",
    "Vtr": "vtr",
}

# FUNCIONES AUXILIARES DE DATOS
def available_files() -> List[Dict[str, str]]:
    """ Devuelve la lista de parquets enriched disponibles en data/enriched/. """
    if not ENRICHED_DIR.exists():
        return []

    options = []
    for path in sorted(ENRICHED_DIR.glob("enriched_*.parquet")):
        if any(path.stem.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        options.append({"label": path.name, "value": str(path)})

    return options

def experiment_options(model_name: str = DEFAULT_MODEL_NAME) -> List[Dict[str, str]]:
    """ Devuelve los experimentos que contienen el modelo indicado."""
    if not EXPERIMENTS_DIR.exists():
        return []
    dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime)
    options: List[Dict[str, str]] = []
    for d in dirs:
        if (d / f"{model_name}_best.joblib").exists():
            options.append({"label": d.name, "value": str(d)})
    return options

@lru_cache(maxsize=32)
def load_dataset(path_str: str) -> pd.DataFrame:
    """ Carga un dataset con caché básica para evitar lecturas repetidas."""
    try:
        df = load_data(path_str)
        if df is None:
            return pd.DataFrame()
        df = df.rename(columns=COL_RENAME)
        return df
    except Exception as exc:
        logger.error("No se pudo cargar el dataset %s: %s", path_str, exc, exc_info=True)
        return pd.DataFrame()

@lru_cache(maxsize=8)
def load_pipeline(experiment_path: str, model_name: str):
    """ Carga el pipeline entrenado y la lista de features desde el experimento. """
    exp_dir = Path(experiment_path)
    model_path = exp_dir / f"{model_name}_best.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    pipeline = joblib.load(model_path)
    feature_cols_path = exp_dir / "feature_columns.json"
    if feature_cols_path.exists():
        feature_columns = json.loads(feature_cols_path.read_text())
    else:
        feature_columns = None
        logger.warning("Falta feature_columns.json en %s; se infieren desde el dataframe.", exp_dir.name)
    return pipeline, feature_columns

def prepare_feature_matrix(df: pd.DataFrame, feature_columns: Optional[List[str]]) -> pd.DataFrame:
    """ Selecciona las columnas en el orden esperado por el pipeline. """
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in {"file", "source_file", "start_s", "duration", "n_samples"}]
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        logger.warning("Faltan columnas esperadas en los datos de entrada: %s", missing)
    return df.reindex(columns=feature_columns)

def compute_window_predictions(
    session_path: str,
    experiment_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    window: float = CFG.windows.size_seconds,
    overlap: float = CFG.windows.overlap_ratio,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """ Extrae ventanas de la sesión seleccionada y aplica el modelo entrenado."""
    feats = extract_features_from_file(
        session_path,
        window=window,
        overlap=overlap,
        file_id=Path(session_path).name.replace("enriched_", "clean_", 1),
    )
    if not feats:
        raise RuntimeError("No se pudieron generar ventanas para esta sesión.")
    df_windows = pd.DataFrame(feats).sort_values("start_s").reset_index(drop=True)
    pipeline, feature_columns = load_pipeline(experiment_path, model_name)
    X = prepare_feature_matrix(df_windows, feature_columns)
    df_windows["fatigue_pred"] = pipeline.predict(X)

    metrics: Dict[str, float] = {}
    if "fatigue_score" in df_windows.columns and df_windows["fatigue_score"].notna().any():
        y_true = df_windows["fatigue_score"].to_numpy()
        y_pred = df_windows["fatigue_pred"].to_numpy()
        metrics = {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
            "R2": float(r2_score(y_true, y_pred)),
        }
    return df_windows, metrics

def relative_time_bounds(df: pd.DataFrame) -> Tuple[float, float]:
    """ Devuelve el intervalo mínimo/máximo de `relative_time`."""
    if "relative_time" not in df.columns:
        return 0.0, 0.0
    return float(df["relative_time"].min()), float(df["relative_time"].max())

# EXTRACCIÓN DE METADATOS
def session_metadata(df: pd.DataFrame, source_path: str) -> Dict[str, str]:
    duration = df["relative_time"].max() - df["relative_time"].min() if "relative_time" in df.columns else 0
    info = {
        "archivo": os.path.basename(source_path),
        "filas": f"{len(df):,}",
        "duración": f"{duration:.1f} s" if duration else "N/A",
    }
    for col in ("runner_id", "session_id", "reported_rpe", "age", "sex"):
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            if unique_vals.size == 1:
                info[col] = str(unique_vals[0])
            elif unique_vals.size > 1:
                info[col] = f"Mixed ({unique_vals.size})"
    for col in ("fatigue_score", "fatigue_level"):
        if col in df.columns and df[col].notna().any():
            val = df[col].dropna().iloc[0]
            info[col] = f"{val:.3f}" if pd.api.types.is_numeric_dtype(df[col]) else str(val)
    return info

# INICIALIZACIÓN DEL DASHBOARD
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
app.title = "Panel de Señales de Running"

file_options = available_files()
default_file = file_options[0]["value"] if file_options else None

empty_notice = html.Div(
    dbc.Alert(
        "No se encontraron parquets enriched en data/enriched/. Ejecuta primero los pipelines de preprocesado, cinemática y métricas.",
        color="warning",
    ),
    className="mb-4",
) if not file_options else None

app.layout = dbc.Container(
    [
html.H1(
            "Panel de Señales de Running",
            className="text-center mb-4",
            style={"color": "#1f2c56"},
        ),
        empty_notice if empty_notice else html.Div(),
        dbc.Row(
            [
        dbc.Col(
                    [
                        html.Label("Sesión:"),
                        dcc.Dropdown(
                            id="file-dropdown",
                            options=file_options,
                            value=default_file,
                            clearable=False,
                            persistence=True,
                        ),
                    ],
                    width=5,
                ),
                dbc.Col(
                    [
                        html.Label("Rango temporal (s):"),
                        dcc.RangeSlider(
                            id="time-slider",
                            min=0,
                            max=60,
                            step=1,
                            value=[0, 60],
                            marks={i: str(i) for i in range(0, 61, 10)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="mouseup",
                        ),
                    ],
                    width=7,
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Metadatos de la sesión", className="card-title"),
                            html.Div(id="metadata-panel", children="Selecciona una sesión para ver los detalles."),
                        ]),
                        className="mb-3",
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Opciones de visualización", className="card-title"),
                            dbc.Checklist(
                                options=[
                                    {"label": "Usar aceleración centrada (si existe)", "value": "centered"},
                                    {"label": "Suavizar velocidad de traslación", "value": "smooth"},
                                ],
                                value=["smooth"],
                                id="display-options",
                                switch=True,
                            ),
                        ]),
                        className="mb-3",
                    ),
                    width=6,
                ),
            ]
        ),
        dcc.Tabs(
            id="tabs",
            value="acc_tab",
            children=[
                dcc.Tab(label="📈 Aceleración", value="acc_tab"),
                dcc.Tab(label="🌍 Gravedad", value="grav_tab"),
                dcc.Tab(label="🔄 Rotación", value="rot_tab"),
                dcc.Tab(label="🧭 Orientación", value="orient_tab"),
                dcc.Tab(label="❤️ FC / SpO₂", value="hr_tab"),
                dcc.Tab(label="🚀 Velocidad de traslación", value="vtr_tab"),
                dcc.Tab(label="🧠 Inferencia de fatiga", value="infer_tab"),
            ],
        ),
        html.Div(id="tab-content", className="mt-3"),
    ],
    fluid=True,
)

# FUNCIONES DE RENDERIZADO DE GRÁFICOS
def render_fatigue_plot(window: pd.DataFrame):
    if "fatigue_score" not in window.columns:
        return dbc.Alert("Este archivo no contiene fatigue_score.", color="warning")
    fig = px.line(
        window,
        x="relative_time",
        y="fatigue_score",
        title="Fatigue score a lo largo del tiempo",
        labels={"relative_time": "Tiempo (s)", "fatigue_score": "Puntuación"},
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=12, color="#2a3f5f"),
        title_font=dict(size=16, color="#1f2c56"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return dcc.Graph(figure=fig)

def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{"text": message, "xref": "paper", "yref": "paper", "showarrow": False}],
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

def style_inference_line(df: pd.DataFrame) -> go.Figure:
    if "fatigue_score" in df.columns and df["fatigue_score"].notna().any():
        long_df = df.melt(
            id_vars=["start_s"],
            value_vars=["fatigue_pred", "fatigue_score"],
            var_name="serie",
            value_name="valor",
        )
    else:
        long_df = df.assign(serie="fatigue_pred", valor=df["fatigue_pred"])
    fig = px.line(
        long_df,
        x="start_s",
        y="valor",
        color="serie",
        title="Fatigue score: predicción vs. valor calculado",
        labels={"start_s": "Tiempo (s)", "valor": "Fatigue score", "serie": "Serie"},
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=12, color="#2a3f5f"),
        title_font=dict(size=16, color="#1f2c56"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig

def style_inference_scatter(df: pd.DataFrame) -> go.Figure:
    if "fatigue_score" not in df.columns or df["fatigue_score"].isna().all():
        return empty_figure("La sesión no contiene fatigue_score para comparar.")
    fig = px.scatter(
        df,
        x="fatigue_score",
        y="fatigue_pred",
        title="Fatigue score real vs. predicho",
        labels={"fatigue_score": "Valor real", "fatigue_pred": "Valor predicho"},
    )
    min_val = float(df[["fatigue_score", "fatigue_pred"]].min().min())
    max_val = float(df[["fatigue_score", "fatigue_pred"]].max().max())
    fig.add_shape(
        type="line",
        x0=min_val,
        x1=max_val,
        y0=min_val,
        y1=max_val,
        line=dict(color="gray", dash="dash"),
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=12, color="#2a3f5f"),
        title_font=dict(size=16, color="#1f2c56"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig

# CALLBACKS
@app.callback(
    Output("time-slider", "min"),
    Output("time-slider", "max"),
    Output("time-slider", "value"),
    Output("time-slider", "marks"),
    Input("file-dropdown", "value"),
)
def update_time_slider(selected_path: Optional[str]):
    """ Ajusta los límites del slider según el archivo seleccionado. """
    if not selected_path:
        default_marks = {i: str(i) for i in range(0, 61, 10)}
        return 0, 60, [0, 60], default_marks

    df = load_dataset(selected_path)
    if df.empty:
        default_marks = {i: str(i) for i in range(0, 61, 10)}
        return 0, 60, [0, 60], default_marks

    t_min, t_max = relative_time_bounds(df)
    if t_max <= t_min:
        default_marks = {int(t_min): str(int(t_min))}
        return t_min, t_max, [t_min, t_max], default_marks

    span = t_max - t_min
    step = max(int(span // 10), 1)
    marks = {int(t): str(int(t)) for t in range(int(t_min), int(t_max) + 1, step)}
    return t_min, t_max, [t_min, t_max], marks

@app.callback(
    Output("metadata-panel", "children"),
    Input("file-dropdown", "value"),
)
def update_metadata_panel(selected_path: Optional[str]):
    if not selected_path:
        return "Selecciona una sesión para ver sus detalles."
    df = load_dataset(selected_path)
    if df.empty:
        return "No se pudo cargar la sesión seleccionada."
    metadata = session_metadata(df, selected_path)
    items = [html.Div([html.Strong(f"{k.replace('_', ' ').title()}:"), f" {v}"]) for k, v in metadata.items()]
    return html.Div(items)

@app.callback(
    Output("tab-content", "children"),
    Input("file-dropdown", "value"),
    Input("time-slider", "value"),
    Input("tabs", "value"),
    Input("display-options", "value"),
)
def render_tab(selected_path: Optional[str], selected_time: List[float], selected_tab: str, options: List[str]):
    """ Renderiza la pestaña seleccionada. """
    if not selected_path:
        return dbc.Alert("Por favor selecciona un archivo de sesión.", color="secondary")

    df = load_dataset(selected_path)
    if df.empty:
        return dbc.Alert("No se pudo cargar el archivo seleccionado o no contiene datos.", color="danger")
    if "relative_time" not in df.columns:
        return dbc.Alert("El dataset no incluye una columna de tiempo válida.", color="danger")

    start, end = selected_time
    window = df[(df["relative_time"] >= start) & (df["relative_time"] <= end)].copy()
    if window.empty:
        return dbc.Alert("No hay muestras para el intervalo seleccionado.", color="warning")

    use_centered = "centered" in (options or [])
    smooth_vtr = "smooth" in (options or [])

    def style_fig(fig):
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Arial", size=12, color="#2a3f5f"),
            title_font=dict(size=16, color="#1f2c56"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    axis_tabs = {
        "acc_tab": (["acc_x", "acc_y", "acc_z"], "Aceleración (g)", "Aceleración (g)"),
        "grav_tab": (["grav_x", "grav_y", "grav_z"], "Componentes de gravedad (g)", "Gravedad (g)"),
        "rot_tab": (["rot_x", "rot_y", "rot_z"], "Velocidad angular (rad/s)", "Velocidad angular (rad/s)"),
        "orient_tab": (["roll", "pitch", "yaw"], "Orientación (rad)", "Ángulo (rad)"),
    }

    if selected_tab in axis_tabs:
        cols, title, y_label = axis_tabs[selected_tab]
        if use_centered and selected_tab == "acc_tab":
            centred_cols = [c + "_centered" for c in cols]
            available = [c for c in centred_cols if c in window.columns]
            display_cols = available
            y_label += " (centrada)"
        else:
            available = [c for c in cols if c in window.columns]
            display_cols = available
        if not available:
            return dbc.Alert(f"No se encuentran las columnas esperadas: {cols}", color="warning")
        fig = px.line(
            window,
            x="relative_time",
            y=display_cols,
            title=title,
            labels={"value": y_label, "variable": "Eje", "relative_time": "Tiempo (s)"},
        )
        return dcc.Graph(figure=style_fig(fig))

    if selected_tab == "hr_tab":
        graphs = []
        if "hr" in window.columns:
            fig_fc = px.line(
                window,
                x="relative_time",
                y="hr",
                title="Frecuencia cardiaca (bpm)",
                labels={"hr": "Pulsaciones/min", "relative_time": "Tiempo (s)"},
            )
            graphs.append(dcc.Graph(figure=style_fig(fig_fc)))
        if "spo2" in window.columns:
            fig_spo2 = px.line(
                window,
                x="relative_time",
                y="spo2",
                title="Oximetría (SpO₂ %)",
                labels={"spo2": "Porcentaje (%)", "relative_time": "Tiempo (s)"},
            )
            graphs.append(dcc.Graph(figure=style_fig(fig_spo2)))
        return html.Div(graphs) if graphs else dbc.Alert(
            "Esta sesión no contiene datos de FC ni SpO₂.", color="warning"
        )

    if selected_tab == "vtr_tab":
        if "vtr" not in window.columns:
            return dbc.Alert("Este archivo no incluye velocidad de traslación.", color="warning")
        if smooth_vtr:
            smoothed = window["vtr"].rolling(window=DEFAULT_VTR_SMOOTHING, center=True, min_periods=1).mean()
            fig = px.line(
                window,
                x="relative_time",
                y=smoothed,
                title="Velocidad de traslación (suavizada)",
                labels={"relative_time": "Tiempo (s)", "value": "Velocidad (m/s)"},
            )
            fig.add_scatter(
                x=window["relative_time"],
                y=window["vtr"],
                mode="lines",
                name="Raw Vtr",
                line=dict(color="rgba(150,150,150,0.3)", width=1, dash="dot"),
            )
        else:
            fig = px.line(
                window,
                x="relative_time",
                y="vtr",
                title="Velocidad de traslación (sin suavizar)",
                labels={"relative_time": "Tiempo (s)", "vtr": "Velocidad (m/s)"},
            )
        return dcc.Graph(figure=style_fig(fig))

    if selected_tab == "infer_tab":
        exp_opts = experiment_options()
        if not exp_opts:
            return dbc.Alert(
                "No se encontraron experimentos con gradient_boosting en data/results/modeling/experiments/. "
                "Ejecuta run_experiments.py primero.",
                color="warning",
            )
        default_exp = exp_opts[-1]["value"]
        return html.Div(
            [
                html.P(
                    "Ejecuta el modelo entrenado sobre la sesión seleccionada para comparar la predicción con el fatigue_score real ventana a ventana.",
                    className="mb-3",
                ),
                dbc.Row(
                    [
                                dbc.Col(
                                    [
                                        html.Label("Selecciona el experimento:"),
                                        dcc.Dropdown(
                                            id="experiment-dropdown",
                                    options=exp_opts,
                                    value=default_exp,
                                    persistence=True,
                                    clearable=False,
                                ),
                            ],
                            md=6,
                        ),
                                dbc.Col(
                                    [
                                        html.Label("Modelo:"),
                                dcc.Input(
                                    id="model-name-input",
                                    type="text",
                                    value=DEFAULT_MODEL_NAME,
                                    readOnly=True,
                                    className="form-control",
                                ),
                            ],
                            md=3,
                        ),
                                dbc.Col(
                                    [
                                        html.Label("Acciones:"),
                                        dbc.Button(
                                            "Calcular predicciones",
                                            id="run-inference-btn",
                                            color="primary",
                                    className="w-100",
                                ),
                            ],
                            md=3,
                        ),
                    ],
                    className="mb-3",
                ),
                html.Div(id="inference-status", className="mb-3"),
                dcc.Loading(
                    dcc.Graph(id="inference-graph"),
                    type="circle",
                ),
                dcc.Loading(
                    dcc.Graph(id="inference-scatter"),
                    type="circle",
                ),
            ],
            className="p-2",
        )

    return dbc.Alert("Pestaña no reconocida.", color="danger")

@app.callback(
    Output("inference-status", "children"),
    Output("inference-graph", "figure"),
    Output("inference-scatter", "figure"),
    Input("run-inference-btn", "n_clicks"),
    State("file-dropdown", "value"),
    State("experiment-dropdown", "value"),
    prevent_initial_call=True,
)
def run_inference_view(n_clicks: Optional[int], session_path: Optional[str], experiment_path: Optional[str]):
    if not n_clicks:
        raise PreventUpdate
    if not session_path:
        return dbc.Alert("Selecciona una sesión primero.", color="warning"), empty_figure("Sin datos"), empty_figure("Sin datos")
    if not experiment_path:
        return dbc.Alert("Selecciona un directorio de experimento.", color="warning"), empty_figure("Sin datos"), empty_figure("Sin datos")

    try:
        df_pred, metrics = compute_window_predictions(session_path, experiment_path, model_name=DEFAULT_MODEL_NAME)
    except Exception as exc:
        logger.exception("La inferencia falló.")
        msg = dbc.Alert(f"Error durante la inferencia: {exc}", color="danger")
        return msg, empty_figure("Error"), empty_figure("Error")

    status_parts = [
        html.Div(f"Experimento: {Path(experiment_path).name}"),
        html.Div(f"Ventanas: {len(df_pred)}"),
    ]
    if metrics:
        status_parts.append(
            html.Div(f"MAE={metrics['MAE']:.3f} | RMSE={metrics['RMSE']:.3f} | R²={metrics['R2']:.3f}")
        )
    else:
        status_parts.append(html.Div("La sesión no incluye fatigue_score; se muestran sólo las predicciones."))
    status = dbc.Alert(status_parts, color="info")
    return status, style_inference_line(df_pred), style_inference_scatter(df_pred)

# FUNCIONES DE LANZAMIENTO
def launch_dashboard(debug: bool = False):
    """Lanza la aplicación de Dash."""
    logger.info("Panel disponible en http://127.0.0.1:8050/")
    app.run(debug=debug)

if __name__ == "__main__":
    launch_dashboard(debug=False)
