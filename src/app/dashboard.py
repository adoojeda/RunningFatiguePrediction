"""Panel interactivo en Dash para explorar sesiones enriched de carrera.

Muestra aceleraciones, jerk, gravedad, rotaciones, orientación, FC/SpO₂ y velocidad de traslación.
"""

# LIBRERÍAS
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import dash_bootstrap_components as dbc # type: ignore
import pandas as pd # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
from dash import Dash, Input, Output, State, dcc, html # type: ignore
from dash.exceptions import PreventUpdate # type: ignore

# CONFIGURACIÓN DEL PATH
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# IMPORTACIONES LOCALES
from src.utils.dashboard_data import (
    available_files,
    compute_window_predictions,
    experiment_options,
    model_options,
    load_dataset,
    _file_mtime,
    relative_time_bounds,
    session_metadata,
    DEFAULT_MODEL_NAME,
)
from src.utils.kinematics_utils import DEFAULT_VTR_SMOOTHING

# CONFIGURACIÓN DEL LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# CONFIGURACIÓN DE LA APLICACIÓN DASH
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True,
    assets_folder=str(BASE_DIR / "assets"),
    assets_url_path="/assets",
)
app.title = "Análisis del cansancio en corredores"

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
        html.Div(style={"height": "16px"}),
        empty_notice if empty_notice else html.Div(),
        html.Div(
            id="controls-container",
            children=[
        dbc.Row(
            [
                dbc.Col(
                    html.Label(
                        "SELECCIONA UNA SESIÓN:",
                        style={"fontSize": "20px", "fontWeight": "700"},
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="file-dropdown",
                        options=file_options,
                        value=default_file,
                        clearable=False,
                        persistence=True,
                        style={"fontSize": "18px", "minHeight": "40px", "height": "40px"},
                    ),
                    width=True,
                ),
            ],
            className="mb-3 align-items-center g-2",
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Label(
                        "ESTABLECE EL RANGO TEMPORAL (S):",
                        style={"fontSize": "20px", "fontWeight": "700"},
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dcc.RangeSlider(
                        id="time-slider",
                        min=0,
                        max=60,
                        step=1,
                        value=[0, 60],
                        marks={i: str(i) for i in range(0, 61, 10)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode="mouseup",
                        className="mt-2",
                        persistence=True,
                        persistence_type="local",
                    ),
                    width=True,
                ),
            ],
            className="mb-4 align-items-center g-2",
        ),
            ],
        ),
        dcc.Tabs(
            id="tabs",
            value="acc_tab",
            persistence=True,
            persistence_type="local",
            children=[
                dcc.Tab(label="ACELERACIÓN", value="acc_tab"),
                dcc.Tab(label="JERK", value="jerk_tab"),
                dcc.Tab(label="VELOCIDAD DE TRASLACIÓN", value="vtr_tab"),
                dcc.Tab(label="GRAVEDAD", value="grav_tab"),
                dcc.Tab(label="ROTACIÓN", value="rot_tab"),
                dcc.Tab(label="ORIENTACIÓN", value="orient_tab"),
                dcc.Tab(label="FRECUENCIA CARDÍACA", value="hr_tab"),
                dcc.Tab(label="SATURACIÓN DE OXÍGENO", value="spo2_tab"),
                dcc.Tab(label="INFERENCIA", value="infer_tab"),
            ],
        ),
        html.Div(
            id="display-options-container",
            children=dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Checklist(
                            options=[],
                            value=[],
                            id="display-options",
                            switch=True,
                        ),
                    ]
                ),
                className="mb-3",
            ),
            className="mt-3",
        ),
        html.Div(id="tab-content", className="mt-3"),
    ],
    fluid=True,
    className="px-4",
)

# FUNCIONES DE ESTILO DE GRÁFICOS
def render_fatigue_plot(window: pd.DataFrame):
    """Genera el gráfico del índice de cansancio físico en la ventana dada."""
    if "physical_fatigue_index" not in window.columns:
        return dbc.Alert("Este archivo no contiene el índice de cansancio físico.", color="warning")
    fig = px.line(
        window,
        x="relative_time",
        y="physical_fatigue_index",
        title="Índice de Cansancio Físico a lo Largo del Tiempo",
        labels={"relative_time": "Tiempo (s)", "physical_fatigue_index": "Índice de Cansancio Físico"},
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=12, color="#2a3f5f"),
        title_font=dict(size=16, color="#1f2c56"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return dcc.Graph(figure=fig)

def empty_figure(message: str) -> go.Figure:
    """Genera una figura vacía con un mensaje centralizado."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{"text": message, "xref": "paper", "yref": "paper", "showarrow": False}],
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

def _target_label(target_col: str) -> str:
    """Obtiene la etiqueta legible para el objetivo dado."""
    labels = {
        "physical_fatigue_index": "Índice de Cansancio Físico",
        "reported_rpe": "RPE Reportado",
    }
    return labels.get(target_col, target_col)

def style_inference_line(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Genera el gráfico de línea de predicciones vs. valores reales."""
    if target_col in df.columns and df[target_col].notna().any():
        long_df = df.melt(
            id_vars=["start_s"],
            value_vars=["fatigue_pred", target_col],
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
        title=f"Predicción vs. {_target_label(target_col)} por Ventana",
        labels={"start_s": "Tiempo (s)", "valor": _target_label(target_col), "serie": "Serie"},
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=12, color="#2a3f5f"),
        title_font=dict(size=16, color="#1f2c56"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig

# FUNCIONES DE CALLBACK
@app.callback(
    Output("time-slider", "min"),
    Output("time-slider", "max"),
    Output("time-slider", "value"),
    Output("time-slider", "marks"),
    Input("file-dropdown", "value"),
)
def update_time_slider(selected_path: Optional[str]):
    """Ajusta los límites del rango temporal según el archivo seleccionado."""
    if not selected_path:
        default_marks = {i: str(i) for i in range(0, 61, 10)}
        return 0, 60, [0, 60], default_marks

    df = load_dataset(selected_path, _file_mtime(selected_path))
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
    Output("tab-content", "children"),
    Input("file-dropdown", "value"),
    Input("time-slider", "value"),
    Input("tabs", "value"),
    Input("display-options", "value"),
)
def render_tab(selected_path: Optional[str], selected_time: List[float], selected_tab: str, options: List[str]):
    """Renderiza el contenido de la pestaña seleccionada."""
    if not selected_path:
        return dbc.Alert("Por favor selecciona un archivo de sesión.", color="secondary")

    df = load_dataset(selected_path, _file_mtime(selected_path))
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
        "jerk_tab": (["jerk_x", "jerk_y", "jerk_z"], "Jerk (g/s)", "Jerk (g/s)"),
        "grav_tab": (["grav_x", "grav_y", "grav_z"], "Componentes de Gravedad (g)", "Gravedad (g)"),
        "rot_tab": (["rot_x", "rot_y", "rot_z"], "Velocidad Angular (rad/s)", "Velocidad Angular (rad/s)"),
        "orient_tab": (["roll", "pitch", "yaw"], "Orientación (rad)", "Ángulo (rad)"),
    }

    if selected_tab in axis_tabs:
        cols, title, y_label = axis_tabs[selected_tab]
        options = options or []
        if selected_tab == "acc_tab":
            mag_cols = []
            if "acc_mag_pair" in options:
                if "acc_mag" in window.columns:
                    mag_cols.append("acc_mag")
                if "acc_dyn_mag" in window.columns:
                    mag_cols.append("acc_dyn_mag")
            if mag_cols:
                display_cols = mag_cols
                title = "Aceleración Total/Dinámica (g)"
                y_label = "Aceleración (g)"
                available = display_cols
            elif use_centered:
                centred_cols = [c + "_centered" for c in cols]
                available = [c for c in centred_cols if c in window.columns]
                display_cols = available
                y_label = "Aceleración Centrada (g)"
            else:
                available = [c for c in cols if c in window.columns]
                display_cols = available
        elif selected_tab == "jerk_tab":
            if "jerk_mag" in options and "jerk_mag" in window.columns:
                display_cols = ["jerk_mag"]
                title = "Jerk Magnitud (g/s)"
                y_label = "Jerk (g/s)"
                available = display_cols
            else:
                available = [c for c in cols if c in window.columns]
                display_cols = available
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
            labels={"value": y_label, "variable": "Señal", "relative_time": "Tiempo (s)"},
        )
        return dcc.Graph(figure=style_fig(fig), style={"height": "70vh"})

    if selected_tab == "hr_tab":
        if "hr" not in window.columns:
            return dbc.Alert("Esta sesión no contiene datos de frecuencia cardíaca.", color="warning")
        fig_fc = px.line(
            window,
            x="relative_time",
            y="hr",
            title="Frecuencia Cardíaca (Pulsaciones por Minuto)",
            labels={"hr": "Pulsaciones por Minuto", "relative_time": "Tiempo (s)"},
        )
        return dcc.Graph(figure=style_fig(fig_fc), style={"height": "70vh"})

    if selected_tab == "spo2_tab":
        if "spo2" not in window.columns:
            return dbc.Alert("Esta sesión no contiene datos de saturación de oxígeno.", color="warning")
        fig_spo2 = px.line(
            window,
            x="relative_time",
            y="spo2",
            title="Saturación de Oxígeno (%)",
            labels={"spo2": "Saturación de Oxígeno (%)", "relative_time": "Tiempo (s)"},
        )
        return dcc.Graph(figure=style_fig(fig_spo2), style={"height": "70vh"})

    if selected_tab == "vtr_tab":
        if "vtr" not in window.columns:
            return dbc.Alert("Este archivo no incluye velocidad de traslación.", color="warning")
        if smooth_vtr:
            smoothed = window["vtr"].rolling(window=DEFAULT_VTR_SMOOTHING, center=True, min_periods=1).mean()
            window_plot = window.copy()
            window_plot["vtr_suavizada"] = smoothed
            fig = px.line(
                window_plot,
                x="relative_time",
                y="vtr_suavizada",
                title="Velocidad de Traslación (Suavizada)",
                labels={"relative_time": "Tiempo (s)", "vtr_suavizada": "Velocidad de Traslación (m/s)"},
            )
            fig.add_scatter(
                x=window["relative_time"],
                y=window["vtr"],
                mode="lines",
                name="Vtr sin suavizar",
                line=dict(color="rgba(150,150,150,0.3)", width=1, dash="dot"),
            )
        else:
            fig = px.line(
                window,
                x="relative_time",
                y="vtr",
                title="Velocidad de Traslación (Sin Suavizar)",
                labels={"relative_time": "Tiempo (s)", "vtr": "Velocidad de Traslación (m/s)"},
            )
        return dcc.Graph(figure=style_fig(fig), style={"height": "70vh"})

    if selected_tab == "infer_tab":
        exp_opts = experiment_options()
        if not exp_opts:
            return dbc.Alert(
                "No se encontraron experimentos con gradient_boosting en data/results/modeling/experiments/. "
                "Ejecuta run_experiments.py primero.",
                color="warning",
            )
        default_exp = exp_opts[-1]["value"]
        model_opts = model_options(default_exp)
        default_model = model_opts[0]["value"] if model_opts else DEFAULT_MODEL_NAME
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Label("Experimento:"),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="experiment-dropdown",
                                            options=exp_opts,
                                            value=default_exp,
                                            persistence=True,
                                            clearable=False,
                                        ),
                                        width=True,
                                    ),
                                ],
                                className="g-2 align-items-center",
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Label("Modelo:"),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="model-name-input",
                                            options=model_opts,
                                            value=default_model,
                                            clearable=False,
                                        ),
                                        width=True,
                                    ),
                                ],
                                className="g-2 align-items-center",
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Calcular predicciones",
                                    id="run-inference-btn",
                                    color="primary",
                                    className="w-100",
                                    style={
                                        "height": "32px",
                                        "fontSize": "12px",
                                        "fontWeight": "700",
                                        "letterSpacing": "0.6px",
                                        "borderRadius": "10px",
                                        "padding": "0 12px",
                                    },
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="mb-3",
                ),
                html.Div(
                    id="inference-status",
                    className="mb-3",
                    style={"maxWidth": "520px", "margin": "0 auto"},
                ),
                html.Div(id="inference-stats", className="mb-3"),
                dcc.Loading(
                    dcc.Graph(id="inference-graph", style={"height": "60vh"}),
                    type="circle",
                ),
            ],
            className="p-2",
        )

    return dbc.Alert("Pestaña no reconocida.", color="danger")

@app.callback(
    Output("display-options-container", "style"),
    Output("display-options", "options"),
    Output("display-options", "value"),
    Input("tabs", "value"),
)
def toggle_display_options(selected_tab: str):
    """Muestra opciones de visualización solo en las pestañas relevantes."""
    if selected_tab == "acc_tab":
        return {}, [
            {"label": "Usar Aceleración Centrada", "value": "centered"},
            {"label": "Usar Aceleración Total y Dinámica", "value": "acc_mag_pair"},
        ], []
    if selected_tab == "jerk_tab":
        return {}, [{"label": "Usar Magnitud del Jerk", "value": "jerk_mag"}], []
    if selected_tab == "vtr_tab":
        return {}, [{"label": "Suavizar Velocidad de Traslación", "value": "smooth"}], ["smooth"]
    return {"display": "none"}, [], []

@app.callback(
    Output("model-name-input", "options"),
    Output("model-name-input", "value"),
    Input("experiment-dropdown", "value"),
)
def update_model_dropdown(experiment_path: Optional[str]):
    """Actualiza las opciones de modelo según el experimento seleccionado."""
    if not experiment_path:
        return [], None
    options = model_options(experiment_path)
    default_value = options[0]["value"] if options else None
    return options, default_value

@app.callback(
    Output("inference-status", "children"),
    Output("inference-graph", "figure"),
    Output("inference-stats", "children"),
    Input("run-inference-btn", "n_clicks"),
    State("file-dropdown", "value"),
    State("experiment-dropdown", "value"),
    State("model-name-input", "value"),
    State("time-slider", "value"),
    prevent_initial_call=True,
)
def run_inference_view(
    n_clicks: Optional[int],
    session_path: Optional[str],
    experiment_path: Optional[str],
    model_name: Optional[str],
    selected_time: Optional[List[float]],
):
    """Ejecuta la inferencia y actualiza el gráfico y las estadísticas."""
    if not n_clicks:
        raise PreventUpdate
    if not session_path:
        return dbc.Alert("Selecciona una sesión primero.", color="warning"), empty_figure("Sin datos"), ""
    if not experiment_path:
        return dbc.Alert("Selecciona un directorio de experimento.", color="warning"), empty_figure("Sin datos"), ""
    options = model_options(experiment_path)
    if not options:
        return dbc.Alert("El experimento seleccionado no contiene modelos disponibles.", color="warning"), empty_figure("Sin datos"), ""
    if model_name not in {opt["value"] for opt in options}:
        model_name = options[0]["value"]
    if not model_name:
        return dbc.Alert("Selecciona un modelo válido.", color="warning"), empty_figure("Sin datos"), ""
    target_col = "physical_fatigue_index"

    try:
        df_pred, metrics = compute_window_predictions(
            session_path,
            experiment_path,
            model_name=model_name,
            target_col=target_col,
        )
    except Exception as exc:
        logger.exception("Falló la inferencia.")
        msg = dbc.Alert(f"Error durante la inferencia: {exc}", color="danger")
        return msg, empty_figure("Error"), ""

    stats_df = df_pred
    if selected_time:
        start, end = selected_time
        stats_df = df_pred[(df_pred["start_s"] >= start) & (df_pred["start_s"] <= end)]

    stats_panel = ""
    if target_col in stats_df.columns and stats_df[target_col].notna().any():
        series = stats_df[target_col].dropna()
    else:
        series = df_pred["fatigue_pred"].dropna()

    status_parts = []
    if metrics:
        status_parts.append(
            html.Div(f"MAE={metrics['MAE']:.3f} | RMSE={metrics['RMSE']:.3f} | R²={metrics['R2']:.3f}")
        )
    if not series.empty:
        status_parts.append(
            html.Div(
                f"Min={series.min():.3f} | Media={series.mean():.3f} | Max={series.max():.3f}"
            )
        )
    status = dbc.Alert(
        status_parts,
        color="info",
        style={
            "textAlign": "center",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center",
            "alignItems": "center",
        },
    )
    return status, style_inference_line(df_pred, target_col), stats_panel

# FUNCIÓN PRINCIPAL
def launch_dashboard(debug: bool = False):
    """Lanza la aplicación Dash para el dashboard interactivo."""
    logger.info("Dashboard disponible en http://127.0.0.1:8050/")
    app.run(debug=debug)

if __name__ == "__main__":
    launch_dashboard(debug=False)
