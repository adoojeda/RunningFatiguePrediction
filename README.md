# Running Fatigue Prediction Pipeline

This repository processes wearable running signals into modelling- and analysis-ready datasets.

## End-to-end flow

1. **Preprocessing** – `python src/data/preprocess.py`
   - input: `data/raw/*.csv`
   - output: `data/processed/clean_*.parquet`
   - cleans timestamps, interpolates HR/SpO₂, removes obvious outliers
   - optional flags: `--input-dir` and `--output-dir` allow working with alternate folders
   - **Raw CSV column map**

     | Position | Description            | Assigned name |
     |---------:|------------------------|---------------|
     | 1        | Absolute timestamp     | `time`        |
     | 2        | Acceleration X         | `acc_x`       |
     | 3        | Acceleration Y         | `acc_y`       |
     | 4        | Acceleration Z         | `acc_z`       |
     | 5        | Gravity X              | `grav_x`      |
     | 6        | Gravity Y              | `grav_y`      |
     | 7        | Gravity Z              | `grav_z`      |
     | 8        | Rotation X             | `rot_x`       |
     | 9        | Rotation Y             | `rot_y`       |
     | 10       | Rotation Z             | `rot_z`       |
     | 11       | Roll                   | `roll`        |
     | 12       | Pitch                  | `pitch`       |
     | 13       | Yaw                    | `yaw`         |
     | 14       | Heart rate             | `hr`          |
     | 15       | Oxygen saturation      | `spo2`        |

2. **Kinematic enrichment** – `python src/features/kinematics.py`
   - input: `data/processed/clean_*.parquet`
   - output: `data/enriched/enriched_*.parquet`
   - centres accelerations and computes raw/dynamic magnitudes

3. **Session metrics** – `python src/data/metrics.py`
   - input: `data/enriched/enriched_*.parquet`
   - output: `data/results/all_sessions_metrics.parquet`
   - derives translational velocity, jerk and fatigue scores

4. **Feature extraction** – `python src/features/features_extraction.py`
   - input: `data/enriched/enriched_*.parquet` + `data/raw/rpe_file_mapping.csv`
   - output: `data/results/features_dataset.parquet`
   - builds sliding-window features (3 s, 75% overlap) aligned with reported RPE/fatigue targets

5. **Analysis & dashboards** – scripts in `src/analysis/` and `src/app/`
   - generate EDA figures (`eda_features.py`) and interactive views (`dashboard.py`)

## Feature set used for modeling

After auditing coverage, variance and redundancy, the modeling stage consumes a curated list of window-level features, including physiology (`hr_mean`, `spo2_mean`, `fatigue_score`), accelerations per axis (`acc_*centered_*`), magnitudes (`acc_*`), translational velocity (`vtr_*`), jerk (`jerk_*`), and orientation/balance (`roll_*`, `yaw_*`, `grav_*`). The whitelist lives in the modeling scripts (e.g., `src/models/run_experiments.py`) and is applied before training for reproducibility.

## Modeling workflows

- **Experiments**: `python src/models/run_experiments.py --dataset data/results/features_dataset.parquet --target fatigue_score --group runner_id --models gradient_boosting random_forest hist_gradient_boosting elasticnet xgboost catboost --save-predictions --save-models (--fast-grid optional)`
  - runs grouped experiments (runner_id by default), executes GridSearchCV for each selected model and persists metrics, predictions, hashes and serialized models under `data/results/modeling/experiments/`.

## Inference workflow

Use `src/models/run_inference.py` (also exposed in `notebooks/inference_demo.ipynb` and via the dashboard tab) to replay a trained pipeline over any `enriched_*.parquet` session and stream the predictions. This produces per-window predictions identical to those seen during evaluation and is also used by the Dash dashboard (`Fatigue Inference` tab) to visualise the model online.
