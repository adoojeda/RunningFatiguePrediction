# Running Fatigue Prediction Pipeline

This repository processes wearable running signals into modelling- and analysis-ready datasets.

## End-to-end flow

1. **Preprocessing** – `python src/data/preprocess.py`
   - input: `data/raw/*.csv`
   - output: `data/processed/clean_*.parquet`
   - cleans timestamps, interpolates HR/SpO₂, removes obvious outliers

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
   - output: `data/results/features_dataset_3s_50olap.parquet`
   - builds sliding-window features (3 s, 50% overlap) aligned with reported RPE

5. **Analysis & dashboards** – scripts in `src/analysis/` and `src/app/`
   - generate EDA figures (`eda_features.py`) and interactive views (`dashboard.py`)

Environment variables such as `RFP_DEFAULT_FS`, `RFP_HP_CUTOFF` and `RFP_VTR_SMOOTHING`
allow parameter tweaks without touching the code.

## Feature set used for modeling

After auditing coverage, variance and redundancy, the modeling stage consumes a curated list of window-level features, including physiology (`fc_mean`, `spo2_mean`, `fatigue_score`), accelerations per axis (`acc_*centered_*`), magnitudes (`acc_*`), translational velocity (`vtr_*`), jerk (`jerk_*`), and orientation/balance (`roll_*`, `yaw_*`, `grav_*`). The whitelist lives in the modeling scripts (e.g., `src/models/run_experiments.py`) and is applied before training for reproducibility.

### Preprocessing pipeline

During training, every numeric feature is passed through a `SimpleImputer(strategy="median")` followed by a `StandardScaler` inside each scikit-learn `Pipeline`, guaranteeing that cross-validation and test evaluation use the exact same transformations.

## Modeling workflows

- **Experiments**: `python src/models/run_experiments.py --dataset data/results/features_dataset_3s_50olap.parquet --target fatigue_score --group runner_id --models gradient_boosting random_forest hist_gradient_boosting elasticnet xgboost catboost --save-predictions --save-models (--fast-grid optional)`
  - runs grouped experiments (runner_id by default), executes GridSearchCV for each selected model (optionally including XGBoost/CatBoost when installed) and persists metrics, predictions, hashes and serialized models under `data/results/modeling/experiments/`.
