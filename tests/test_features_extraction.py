from pathlib import Path
import sys

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.features_extraction import compute_window_features

def test_compute_window_features_includes_fatigue_score():
    df_win = pd.DataFrame(
        {
            "relative_time": [0.0, 0.5, 1.0],
            "acc_x_centered": [0.1, 0.2, 0.3],
            "acc_y_centered": [0.1, 0.2, 0.3],
            "acc_z_centered": [0.1, 0.2, 0.3],
            "acc_mag": [0.2, 0.3, 0.4],
            "vtr": [0.1, 0.1, 0.1],
            "jerk_mag": [0.05, 0.06, 0.07],
            "hr": [120, 125, 130],
            "spo2": [97, 96, 95],
        }
    )
    refs = {"hr_max": 150, "spo2_min": 90, "acc_std_ref": 1.0, "jerk_std_ref": 1.0}
    features = compute_window_features(df_win, "file.parquet", "file.parquet", fatigue_refs=refs)
    assert "fatigue_score" in features
    assert 0.0 <= features["fatigue_score"] <= 1.0
