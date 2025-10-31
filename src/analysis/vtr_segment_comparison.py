"""
Compare translational velocity (Vtr) segments within a single session using side-by-side plots.

Examples
--------
    python src/analysis/vtr_segment_comparison.py --file data/enriched/enriched_session.parquet
    python src/analysis/vtr_segment_comparison.py --file ... --segments 10 12 40 43
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "visualisations")

# ======================================================================
# DATA STRUCTURES
# ======================================================================
@dataclass
class Segment:
    start: float
    end: float

    def label(self) -> str:
        return f"{self.start:.1f}-{self.end:.1f}s"

# ======================================================================
# HELPERS
# ======================================================================
def load_session(path: str) -> pd.DataFrame:
    """
    Load an enriched/processed parquet file and ensure Vtr & Relative_Time columns exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_parquet(path)
    if "Vtr" not in df.columns:
        raise ValueError("Column 'Vtr' not found in the input file.")

    if "Relative_Time" not in df.columns:
        if "Tiempo_rel" in df.columns:
            df = df.rename(columns={"Tiempo_rel": "Relative_Time"})
        else:
            raise ValueError("Column 'Relative_Time' not found in the input file.")

    return df.sort_values("Relative_Time").reset_index(drop=True)

def select_random_segments(
    df: pd.DataFrame,
    duration: float,
    n_segments: int = 2,
    seed: Optional[int] = None,
) -> List[Segment]:
    """
    Randomly select n non-overlapping segments of fixed duration (seconds).
    """
    rng = np.random.default_rng(seed)
    t_min, t_max = df["Relative_Time"].min(), df["Relative_Time"].max()
    total_duration = t_max - t_min
    if total_duration <= duration:
        raise ValueError("Signal is too short for the requested segments.")

    segments: List[Segment] = []
    for _ in range(n_segments):
        start = float(rng.uniform(t_min, t_max - duration))
        end = start + duration
        segments.append(Segment(start=start, end=end))

    logger.info("Random segments selected: %s", [s.label() for s in segments])
    return segments


def parse_manual_segments(values: List[float]) -> List[Segment]:
    """Convert a flat list of [start1, end1, start2, end2, ...] into Segment objects."""
    if len(values) < 2 or len(values) % 2 != 0:
        raise ValueError("Manual segments require an even number of values (start end ...).")

    segments: List[Segment] = []
    for start, end in zip(values[::2], values[1::2]):
        if end <= start:
            raise ValueError(f"Segment end {end} must be greater than start {start}.")
        segments.append(Segment(start=float(start), end=float(end)))
    return segments

def plot_two_segments_side_by_side(
    df: pd.DataFrame,
    segments: List[Segment],
    output_dir: str,
    filename: str = "vtr_segment_comparison",
    show_figure: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Plot two Vtr segments side by side and persist the figure.

    Parameters
    ----------
    df:
        DataFrame containing 'Vtr' and 'Relative_Time'.
    segments:
        Segments to visualise (only the first two are used).
    output_dir:
        Directory where the figure will be saved.
    filename:
        Base filename (without extension).
    show_figure:
        If True, call `figure.show()` for interactive view.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if len(segments) < 2:
        raise ValueError("At least two segments are required for comparison.")

    os.makedirs(output_dir, exist_ok=True)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"Segment 1: {segments[0].label()}", f"Segment 2: {segments[1].label()}"],
        shared_yaxes=True,
    )

    colors = ["royalblue", "tomato"]
    global_min = float(df["Vtr"].min())
    global_max = float(df["Vtr"].max())

    for idx, segment in enumerate(segments[:2]):
        mask = (df["Relative_Time"] >= segment.start) & (df["Relative_Time"] <= segment.end)
        tramo = df.loc[mask]
        if tramo.empty:
            logger.warning("Segment %s contains no samples; skipping.", segment.label())
            continue

        tramo_time = tramo["Relative_Time"] - tramo["Relative_Time"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=tramo_time,
                y=tramo["Vtr"],
                mode="lines",
                name=f"Segment {idx + 1}",
                line=dict(color=colors[idx], width=2),
            ),
            row=1,
            col=idx + 1,
        )

        fig.update_xaxes(title_text="Time (s)", row=1, col=idx + 1)
        fig.update_yaxes(title_text="Velocity (m/s)", range=[global_min, global_max], row=1, col=idx + 1)

    fig.update_layout(
        template="plotly_white",
        title="Translational Velocity (Vtr) – Two-Segment Comparison",
        autosize=True,
        height=600,
        margin=dict(l=60, r=60, t=80, b=60),
        showlegend=False,
    )

    html_path = os.path.join(output_dir, f"{filename}.html")
    png_path = os.path.join(output_dir, f"{filename}.png")

    fig.write_html(html_path)
    try:
        fig.write_image(png_path)
    except Exception as exc:
        logger.warning("PNG export failed (install 'kaleido' for PNG support): %s", exc)
        png_path = None

    logger.info("Figure saved to %s", html_path)

    if show_figure:
        fig.show()

    return html_path, png_path

# ======================================================================
# PIPELINE
# ======================================================================
def main(
    file_path: str,
    duration: float = 2.0,
    seed: Optional[int] = None,
    show: bool = False,
    manual_segments: Optional[List[Segment]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Generate the comparison visualisation for two random segments.
    """
    df = load_session(file_path)
    if manual_segments:
        segments = manual_segments
    else:
        segments = select_random_segments(df, n_segments=2, duration=duration, seed=seed)

    t_min, t_max = df["Relative_Time"].min(), df["Relative_Time"].max()
    for seg in segments:
        if seg.start < t_min or seg.end > t_max:
            raise ValueError(
                f"Segment {seg.label()} is outside the available time range ({t_min:.2f}-{t_max:.2f}s)."
            )

    session_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, session_name)

    return plot_two_segments_side_by_side(
        df=df,
        segments=segments,
        output_dir=output_dir,
        filename=f"vtr_segment_comparison_{duration:.1f}s",
        show_figure=show,
    )

# ======================================================================
# CLI
# ======================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two random 2-second Vtr segments from the same session."
    )
    parser.add_argument("--file", required=True, help="Path to the enriched parquet file (enriched_*.parquet).")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration of each segment in seconds (default 2.0).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed to reproduce segment selection.")
    parser.add_argument(
        "--segments",
        type=float,
        nargs="+",
        help="Manual segment boundaries [start1 end1 start2 end2 ...] in seconds.",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        manual_segments = parse_manual_segments(args.segments) if args.segments else None
        html_path, png_path = main(
            file_path=args.file,
            duration=args.duration,
            seed=None if manual_segments else args.seed,
            show=args.show,
            manual_segments=manual_segments,
        )
        logger.info("✅ Visualisation ready: %s%s", html_path, f' and {png_path}' if png_path else "")
    except Exception as exc:
        logger.error("❌ Failed to generate segment comparison: %s", exc, exc_info=True)
