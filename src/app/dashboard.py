"""
Interactive Dash dashboard to explore enriched running sessions.
Displays acceleration, gravity, rotation, orientation, HR/SpO₂, and translational velocity.
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

# ===========================
# LOGGING SETUP
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ===========================
# PATHS AND CONSTANTS
# ===========================
DATA_DIR = BASE_DIR / "data"
ENRICHED_DIR = DATA_DIR / "enriched"
EXPERIMENTS_DIR = DATA_DIR / "results" / "modeling" / "experiments"
EXCLUDED_PREFIXES = ("all_sessions_metrics", "features_dataset")
DEFAULT_MODEL_NAME = "gradient_boosting"
CFG = get_config()

# Mapping from legacy/camel columns to snake_case
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

# ===========================
# DATA HELPERS
# ===========================
def available_files() -> List[Dict[str, str]]:
    """
    Return a list of enriched parquet files available in data/enriched/.
    """
    if not ENRICHED_DIR.exists():
        return []

    options = []
    for path in sorted(ENRICHED_DIR.glob("enriched_*.parquet")):
        if any(path.stem.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        options.append({"label": path.name, "value": str(path)})

    return options

def experiment_options(model_name: str = DEFAULT_MODEL_NAME) -> List[Dict[str, str]]:
    """
    List experiment directories that contain the requested model.
    """
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
    """
    Wrap `load_data` with basic caching to avoid repeated disk reads.
    """
    try:
        df = load_data(path_str)
        if df is None:
            return pd.DataFrame()
        df = df.rename(columns=COL_RENAME)
        return df
    except Exception as exc:
        logger.error("Failed to load dataset %s: %s", path_str, exc, exc_info=True)
        return pd.DataFrame()

@lru_cache(maxsize=8)
def load_pipeline(experiment_path: str, model_name: str):
    """
    Load the persisted sklearn pipeline and feature list from an experiment dir.
    """
    exp_dir = Path(experiment_path)
    model_path = exp_dir / f"{model_name}_best.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    pipeline = joblib.load(model_path)
    feature_cols_path = exp_dir / "feature_columns.json"
    if feature_cols_path.exists():
        feature_columns = json.loads(feature_cols_path.read_text())
    else:
        feature_columns = None
        logger.warning("feature_columns.json missing in %s; columns inferred from dataframe.", exp_dir.name)
    return pipeline, feature_columns

def prepare_feature_matrix(df: pd.DataFrame, feature_columns: Optional[List[str]]) -> pd.DataFrame:
    """
    Select features in the order expected by the pipeline.
    """
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in {"file", "source_file", "start_s", "duration", "n_samples"}]
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        logger.warning("Input features missing expected columns: %s", missing)
    return df.reindex(columns=feature_columns)

def compute_window_predictions(
    session_path: str,
    experiment_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    window: float = CFG.windows.size_seconds,
    overlap: float = CFG.windows.overlap_ratio,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Extract sliding-window features from the selected session and run the trained pipeline.
    """
    feats = extract_features_from_file(
        session_path,
        window=window,
        overlap=overlap,
        file_id=Path(session_path).name.replace("enriched_", "clean_", 1),
    )
    if not feats:
        raise RuntimeError("No windows could be generated for this session.")
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
    """
    Return min/max bounds for the `Relative_Time` column. Falls back to zero-length interval.
    """
    if "relative_time" not in df.columns:
        return 0.0, 0.0
    return float(df["relative_time"].min()), float(df["relative_time"].max())

# ===========================
# SESSION METADATA EXTRACTION
# ===========================
def session_metadata(df: pd.DataFrame, source_path: str) -> Dict[str, str]:
    duration = df["relative_time"].max() - df["relative_time"].min() if "relative_time" in df.columns else 0
    info = {
        "file": os.path.basename(source_path),
        "rows": f"{len(df):,}",
        "duration": f"{duration:.1f} s" if duration else "N/A",
    }
    for col in ("runner_id", "session_id", "reported_rpe", "age", "sex"):
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            if unique_vals.size == 1:
                info[col] = str(unique_vals[0])
            elif unique_vals.size > 1:
                info[col] = f"Mixed ({unique_vals.size})"
    # Optional targets
    for col in ("fatigue_score", "fatigue_level"):
        if col in df.columns and df[col].notna().any():
            val = df[col].dropna().iloc[0]
            info[col] = f"{val:.3f}" if pd.api.types.is_numeric_dtype(df[col]) else str(val)
    return info

# ===========================
# DASH APP INITIALISATION
# ===========================
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
app.title = "Running Signals Dashboard"

# ===========================
# LAYOUT DEFINITION
# ===========================
file_options = available_files()
default_file = file_options[0]["value"] if file_options else None

empty_notice = html.Div(
    dbc.Alert(
        "No enriched parquet files found in data/enriched/. Run the preprocessing, kinematics, and metrics pipelines first.",
        color="warning",
    ),
    className="mb-4",
) if not file_options else None

app.layout = dbc.Container(
    [
        html.H1(
            "Running Signals Dashboard",
            className="text-center mb-4",
            style={"color": "#1f2c56"},
        ),
        empty_notice if empty_notice else html.Div(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Select enriched session:"),
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
                        html.Label("Select time range (s):"),
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
                            html.H5("Session metadata", className="card-title"),
                            html.Div(id="metadata-panel", children="Select a session to inspect its details."),
                        ]),
                        className="mb-3",
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Display options", className="card-title"),
                            dbc.Checklist(
                                options=[
                                    {"label": "Use centred acceleration (if available)", "value": "centered"},
                                    {"label": "Smooth translational velocity", "value": "smooth"},
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
                dcc.Tab(label="📈 Acceleration", value="acc_tab"),
                dcc.Tab(label="🌍 Gravity", value="grav_tab"),
                dcc.Tab(label="🔄 Rotation", value="rot_tab"),
                dcc.Tab(label="🧭 Orientation", value="orient_tab"),
                dcc.Tab(label="❤️ HR / SpO₂", value="hr_tab"),
                dcc.Tab(label="🚀 Translational Velocity", value="vtr_tab"),
                dcc.Tab(label="🧠 Fatigue Inference", value="infer_tab"),
            ],
        ),
        html.Div(id="tab-content", className="mt-3"),
    ],
    fluid=True,
)

# ===========================
# HELPER FUNCTIONS
# ===========================
def render_fatigue_plot(window: pd.DataFrame):
    if "fatigue_score" not in window.columns:
        return dbc.Alert("Fatigue score not available in this file.", color="warning")
    fig = px.line(
        window,
        x="relative_time",
        y="fatigue_score",
        title="Fatigue score over time",
        labels={"relative_time": "Time (s)", "fatigue_score": "Score"},
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
        title="Fatigue score: prediction vs. calculated value",
        labels={"start_s": "Time (s)", "valor": "Fatigue score", "serie": "Series"},
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
        return empty_figure("The session does not contain fatigue_score for comparison.")
    fig = px.scatter(
        df,
        x="fatigue_score",
        y="fatigue_pred",
        title="True vs. predicted fatigue score",
        labels={"fatigue_score": "True score", "fatigue_pred": "Predicted score"},
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

# ===========================
# CALLBACKS
# ===========================
@app.callback(
    Output("time-slider", "min"),
    Output("time-slider", "max"),
    Output("time-slider", "value"),
    Output("time-slider", "marks"),
    Input("file-dropdown", "value"),
)
def update_time_slider(selected_path: Optional[str]):
    """
    Adjust slider bounds based on the chosen file.
    """
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
        return "Select a session to inspect its details."
    df = load_dataset(selected_path)
    if df.empty:
        return "Unable to load the selected session."
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
    """
    Render the currently selected tab.
    """
    if not selected_path:
        return dbc.Alert("Please select a session file.", color="secondary")

    df = load_dataset(selected_path)
    if df.empty:
        return dbc.Alert("Unable to load the selected file or it contains no data.", color="danger")
    if "relative_time" not in df.columns:
        return dbc.Alert("The dataset does not include a valid time column.", color="danger")

    start, end = selected_time
    window = df[(df["relative_time"] >= start) & (df["relative_time"] <= end)].copy()
    if window.empty:
        return dbc.Alert("No samples available for the selected time interval.", color="warning")

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
        "acc_tab": (["acc_x", "acc_y", "acc_z"], "Acceleration (m/s²)", "Acceleration (m/s²)"),
        "grav_tab": (["grav_x", "grav_y", "grav_z"], "Gravity components (g)", "Gravity (g)"),
        "rot_tab": (["rot_x", "rot_y", "rot_z"], "Angular velocity (rad/s)", "Angular velocity (rad/s)"),
        "orient_tab": (["roll", "pitch", "yaw"], "Orientation (rad)", "Angle (rad)"),
    }

    if selected_tab in axis_tabs:
        cols, title, y_label = axis_tabs[selected_tab]
        if use_centered and selected_tab == "acc_tab":
            centred_cols = [c + "_centered" for c in cols]
            available = [c for c in centred_cols if c in window.columns]
            display_cols = available
            y_label += " (centered)"
        else:
            available = [c for c in cols if c in window.columns]
            display_cols = available
        if not available:
            return dbc.Alert(f"Expected columns not found: {cols}", color="warning")
        fig = px.line(
            window,
            x="relative_time",
            y=display_cols,
            title=title,
            labels={"value": y_label, "variable": "Axis", "relative_time": "Time (s)"},
        )
        return dcc.Graph(figure=style_fig(fig))

    if selected_tab == "hr_tab":
        graphs = []
        if "hr" in window.columns:
            fig_fc = px.line(
                window,
                x="relative_time",
                y="hr",
                title="Heart rate (bpm)",
                labels={"hr": "Beats per minute", "relative_time": "Time (s)"},
            )
            graphs.append(dcc.Graph(figure=style_fig(fig_fc)))
        if "spo2" in window.columns:
            fig_spo2 = px.line(
                window,
                x="relative_time",
                y="spo2",
                title="Pulse oximetry (SpO₂ %)",
                labels={"spo2": "Percentage (%)", "relative_time": "Time (s)"},
            )
            graphs.append(dcc.Graph(figure=style_fig(fig_spo2)))
        return html.Div(graphs) if graphs else dbc.Alert(
            "No HR or SpO₂ data available for this session.", color="warning"
        )

    if selected_tab == "vtr_tab":
        if "vtr" not in window.columns:
            return dbc.Alert("Translational velocity is not available in this file.", color="warning")
        if smooth_vtr:
            smoothed = window["vtr"].rolling(window=DEFAULT_VTR_SMOOTHING, center=True, min_periods=1).mean()
            fig = px.line(
                window,
                x="relative_time",
                y=smoothed,
                title="Translational velocity (smoothed)",
                labels={"relative_time": "Time (s)", "value": "Velocity (m/s)"},
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
                title="Translational velocity (raw)",
                labels={"relative_time": "Time (s)", "vtr": "Velocity (m/s)"},
            )
        return dcc.Graph(figure=style_fig(fig))

    if selected_tab == "infer_tab":
        exp_opts = experiment_options()
        if not exp_opts:
            return dbc.Alert(
                "No experiments with gradient_boosting models were found in data/results/modeling/experiments/. "
                "Run run_experiments.py first.",
                color="warning",
            )
        default_exp = exp_opts[-1]["value"]
        return html.Div(
            [
                html.P(
                    "Run the trained model over the selected session to compare predicted and actual fatigue scores window by window.",
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Select the experiment:"),
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
                                html.Label("Model:"),
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
                                html.Label("Actions:"),
                                dbc.Button(
                                    "Calculate Predictions",
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

    return dbc.Alert("Tab not recognised.", color="danger")

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
        return dbc.Alert("Select a session first.", color="warning"), empty_figure("Sin datos"), empty_figure("Sin datos")
    if not experiment_path:
        return dbc.Alert("Select an experiment directory.", color="warning"), empty_figure("Sin datos"), empty_figure("Sin datos")

    try:
        df_pred, metrics = compute_window_predictions(session_path, experiment_path, model_name=DEFAULT_MODEL_NAME)
    except Exception as exc:
        logger.exception("Inference failed.")
        msg = dbc.Alert(f"Error during inference: {exc}", color="danger")
        return msg, empty_figure("Error"), empty_figure("Error")

    status_parts = [
        html.Div(f"Experiment: {Path(experiment_path).name}"),
        html.Div(f"Windows: {len(df_pred)}"),
    ]
    if metrics:
        status_parts.append(
            html.Div(f"MAE={metrics['MAE']:.3f} | RMSE={metrics['RMSE']:.3f} | R²={metrics['R2']:.3f}")
        )
    else:
        status_parts.append(html.Div("The session does not include a fatigue_score; showing predictions only."))
    status = dbc.Alert(status_parts, color="info")
    return status, style_inference_line(df_pred), style_inference_scatter(df_pred)

# ===========================
# LAUNCHER
# ===========================
def launch_dashboard(debug: bool = False):
    """Start the Dash application."""
    logger.info("Dashboard available at http://127.0.0.1:8050/")
    app.run(debug=debug)

if __name__ == "__main__":
    launch_dashboard(debug=False)
