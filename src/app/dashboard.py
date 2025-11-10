"""
Interactive Dash dashboard to explore enriched running sessions.
Displays acceleration, gravity, rotation, orientation, HR/SpO₂, and translational velocity.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.data_loader import load_data
from src.utils.kinematics import DEFAULT_VTR_SMOOTHING

# ======================================================================
# LOGGING CONFIGURATION
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================================================
# PATH CONFIGURATION
# ======================================================================
DATA_DIR = BASE_DIR / "data"
ENRICHED_DIR = DATA_DIR / "enriched"
EXCLUDED_PREFIXES = ("all_sessions_metrics", "features_dataset")

# ======================================================================
# DATA HELPERS
# ======================================================================
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

@lru_cache(maxsize=32)
def load_dataset(path_str: str) -> pd.DataFrame:
    """
    Wrap `load_data` with basic caching to avoid repeated disk reads.
    """
    try:
        df = load_data(path_str)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception as exc:
        logger.error("Failed to load dataset %s: %s", path_str, exc, exc_info=True)
        return pd.DataFrame()

def relative_time_bounds(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Return min/max bounds for the `Relative_Time` column. Falls back to zero-length interval.
    """
    if "Relative_Time" not in df.columns:
        if "Tiempo_rel" in df.columns:
            df = df.rename(columns={"Tiempo_rel": "Relative_Time"})
        else:
            return 0.0, 0.0
    return float(df["Relative_Time"].min()), float(df["Relative_Time"].max())

# ======================================================================
def session_metadata(df: pd.DataFrame, source_path: str) -> Dict[str, str]:
    duration = df["Relative_Time"].max() - df["Relative_Time"].min() if "Relative_Time" in df.columns else 0
    info = {
        "file": os.path.basename(source_path),
        "rows": f"{len(df):,}",
        "duration": f"{duration:.1f} s" if duration else "N/A",
    }
    for col in ("runner_id", "session_id", "reported_rpe"):
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            if unique_vals.size == 1:
                info[col] = str(unique_vals[0])
            elif unique_vals.size > 1:
                info[col] = f"Mixed ({unique_vals.size})"
    return info

# ======================================================================
# DASH APP INITIALISATION
# ======================================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
app.title = "Running Signals Dashboard"

# ======================================================================
# LAYOUT
# ======================================================================
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
                dcc.Tab(label="❤️ HR / SpO₂", value="fc_tab"),
                dcc.Tab(label="🚀 Translational Velocity", value="vtr_tab"),
            ],
        ),
        html.Div(id="tab-content", className="mt-3"),
    ],
    fluid=True,
)

# ======================================================================
# CALLBACKS
# ======================================================================
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

    if "Relative_Time" not in df.columns:
        if "Tiempo_rel" in df.columns:
            df = df.rename(columns={"Tiempo_rel": "Relative_Time"})
        else:
            return dbc.Alert("The dataset does not include a valid time column.", color="danger")

    start, end = selected_time
    window = df[(df["Relative_Time"] >= start) & (df["Relative_Time"] <= end)].copy()
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
        "acc_tab": (["AccX", "AccY", "AccZ"], "Acceleration (m/s²)", "Acceleration (m/s²)"),
        "grav_tab": (["GravX", "GravY", "GravZ"], "Gravity components (g)", "Gravity (g)"),
        "rot_tab": (["RotX", "RotY", "RotZ"], "Angular velocity (rad/s)", "Angular velocity (rad/s)"),
        "orient_tab": (["Roll", "Pitch", "Yaw"], "Orientation (rad)", "Angle (rad)"),
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
            x="Relative_Time",
            y=display_cols,
            title=title,
            labels={"value": y_label, "variable": "Axis", "Relative_Time": "Time (s)"},
        )
        return dcc.Graph(figure=style_fig(fig))

    if selected_tab == "fc_tab":
        graphs = []
        if "FC" in window.columns:
            fig_fc = px.line(
                window,
                x="Relative_Time",
                y="FC",
                title="Heart rate (bpm)",
                labels={"FC": "Beats per minute", "Relative_Time": "Time (s)"},
            )
            graphs.append(dcc.Graph(figure=style_fig(fig_fc)))
        if "SpO2" in window.columns:
            fig_spo2 = px.line(
                window,
                x="Relative_Time",
                y="SpO2",
                title="Pulse oximetry (SpO₂ %)",
                labels={"SpO2": "Percentage (%)", "Relative_Time": "Time (s)"},
            )
            graphs.append(dcc.Graph(figure=style_fig(fig_spo2)))
        return html.Div(graphs) if graphs else dbc.Alert(
            "No HR or SpO₂ data available for this session.", color="warning"
        )

    if selected_tab == "vtr_tab":
        if "Vtr" not in window.columns:
            return dbc.Alert("Translational velocity is not available in this file.", color="warning")
        if smooth_vtr:
            smoothed = window["Vtr"].rolling(window=DEFAULT_VTR_SMOOTHING, center=True, min_periods=1).mean()
            fig = px.line(
                window,
                x="Relative_Time",
                y=smoothed,
                title="Translational velocity (smoothed)",
                labels={"Relative_Time": "Time (s)", "value": "Velocity (m/s)"},
            )
            fig.add_scatter(
                x=window["Relative_Time"],
                y=window["Vtr"],
                mode="lines",
                name="Raw Vtr",
                line=dict(color="rgba(150,150,150,0.3)", width=1, dash="dot"),
            )
        else:
            fig = px.line(
                window,
                x="Relative_Time",
                y="Vtr",
                title="Translational velocity (raw)",
                labels={"Relative_Time": "Time (s)", "Vtr": "Velocity (m/s)"},
            )
        return dcc.Graph(figure=style_fig(fig))

    return dbc.Alert("Tab not recognised.", color="danger")

# ======================================================================
# LAUNCHER
# ======================================================================
def launch_dashboard(debug: bool = False):
    """Start the Dash application."""
    logger.info("Dashboard available at http://127.0.0.1:8050/")
    app.run(debug=debug)

if __name__ == "__main__":
    launch_dashboard(debug=False)
