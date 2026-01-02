"""Pruebas para el módulo de procesamiento cinemático."""

# LIBRERÍAS ESTÁNDAR
from pathlib import Path
import sys

import pandas as pd # type: ignore

# RUTA AL DIRECTORIO RAÍZ
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# IMPORTS LOCALES
from src.features.kinematics import list_processed_files, process_single_file

# PRUEBAS UNITARIAS
def test_list_processed_files(tmp_path):
    """Listar archivos procesados en un directorio."""
    clean_dir = tmp_path / "processed"
    clean_dir.mkdir()
    (clean_dir / "clean_a.parquet").write_text("")
    (clean_dir / "clean_b.parquet").write_text("")
    (clean_dir / "random.txt").write_text("")

    files = list_processed_files(str(clean_dir))
    assert len(files) == 2
    assert files[0].endswith("clean_a.parquet")

def test_process_single_file_generates_enriched(tmp_path):
    """El procesamiento de un archivo genera el archivo enriquecido."""
    processed_file = tmp_path / "clean_demo.parquet"
    output_dir = tmp_path / "enriched"
    output_dir.mkdir()

    df = pd.DataFrame(
        {
            "relative_time": [0.0, 0.02, 0.04],
            "acc_x": [0.1, 0.2, 0.3],
            "acc_y": [0.0, 0.0, 0.0],
            "acc_z": [0.5, 0.5, 0.5],
            "grav_x": [0.0, 0.0, 0.0],
            "grav_y": [0.0, 0.0, 0.0],
            "grav_z": [1.0, 1.0, 1.0],
            "rot_x": [0.0, 0.0, 0.0],
            "rot_y": [0.0, 0.0, 0.0],
            "rot_z": [0.0, 0.0, 0.0],
            "roll": [0.0, 0.0, 0.0],
            "pitch": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
            "hr": [60, 60, 60],
            "spo2": [98, 98, 98],
        }
    )
    df.to_parquet(processed_file, index=False)

    stats = process_single_file(
        str(processed_file),
        output_dir=str(output_dir),
    )
    assert stats is not None
    enriched_path = output_dir / "enriched_demo.parquet"
    assert enriched_path.exists()
    enriched_df = pd.read_parquet(enriched_path)
    assert "acc_mag" in enriched_df.columns

def test_process_single_file_with_nan_accelerations(tmp_path):
    """El procesamiento de un archivo con NaN en aceleraciones."""
    processed_file = tmp_path / "clean_nan.parquet"
    output_dir = tmp_path / "enriched"
    output_dir.mkdir()

    df = pd.DataFrame(
        {
            "relative_time": [0.0, 0.02, 0.04],
            "acc_x": [0.1, float("nan"), 0.3],
            "acc_y": [0.0, 0.0, 0.0],
            "acc_z": [0.5, float("nan"), 0.5],
            "grav_x": [0.0, 0.0, 0.0],
            "grav_y": [0.0, 0.0, 0.0],
            "grav_z": [1.0, 1.0, 1.0],
            "rot_x": [0.0, 0.0, 0.0],
            "rot_y": [0.0, 0.0, 0.0],
            "rot_z": [0.0, 0.0, 0.0],
            "roll": [0.0, 0.0, 0.0],
            "pitch": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
            "hr": [60, 60, 60],
            "spo2": [98, 98, 98],
        }
    )
    df.to_parquet(processed_file, index=False)

    stats = process_single_file(
        str(processed_file),
        output_dir=str(output_dir),
    )
    assert stats is not None
    enriched_path = output_dir / "enriched_nan.parquet"
    assert enriched_path.exists()
    enriched_df = pd.read_parquet(enriched_path)
    assert "acc_mag" in enriched_df.columns
