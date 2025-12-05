from pathlib import Path
import sys

import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.preprocess_utils import (
    load_raw_file,
    ColumnCountError,
    apply_physio_filters,
)

def test_load_raw_file_assigns_expected_columns(tmp_path):
    row = list(range(15))
    data = [row]
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(data).to_csv(csv_path, header=False, index=False)

    df = load_raw_file(str(csv_path))
    assert list(df.columns) == [
        "time",
        "acc_x",
        "acc_y",
        "acc_z",
        "grav_x",
        "grav_y",
        "grav_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "roll",
        "pitch",
        "yaw",
        "hr",
        "spo2",
    ]

def test_load_raw_file_invalid_column_count(tmp_path):
    data = [[0.0] * 10]
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame(data).to_csv(csv_path, header=False, index=False)

    with pytest.raises(ColumnCountError):
        load_raw_file(str(csv_path))

def test_apply_physio_filters_replaces_sentinels_and_outliers():
    df = pd.DataFrame(
        {
            "hr": [60, 999, 20, 220],
            "spo2": [95, 999, 60, 110],
        }
    )
    apply_physio_filters(df, hr_range=(40, 200), spo2_range=(80, 100))

    assert np.isnan(df.loc[1, "hr"])
    assert np.isnan(df.loc[1, "spo2"])
    assert np.isnan(df.loc[2, "hr"])  
    assert np.isnan(df.loc[3, "hr"])  
    assert np.isnan(df.loc[2, "spo2"])
    assert np.isnan(df.loc[3, "spo2"])
    assert df.loc[0, "hr"] == 60
    assert df.loc[0, "spo2"] == 95
