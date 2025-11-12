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
   - builds sliding-window features (3 s, 50 % solape) alineadas con los RPE reportados

5. **Analysis & dashboards** – scripts in `src/analysis/` and `src/app/`
   - generate EDA figures (`eda_features.py`) and interactive views (`dashboard.py`)

Environment variables such as `RFP_DEFAULT_FS`, `RFP_HP_CUTOFF` and `RFP_VTR_SMOOTHING`
allow parameter tweaks without touching the code.

## Feature set used for modeling

After auditing coverage, variance and redundancy, the modeling stage consumes a curated list of window-level features:

- **Physiological**: `FC_mean`, `SpO2_mean`, `Fatigue_Score`, and the fatigue components `Fatigue_component_norm_fc`, `Fatigue_component_norm_acc`, `Fatigue_component_norm_jerk`.
- **Acceleration (per axis)**: mean, standard deviation, MAD and skew/kurtosis for the centred axes (`AccX/Y/Z_centered_*`).
- **Acceleration magnitude**: `Acc_mean`, `Acc_std`, `Acc_mag_mad`, `Acc_mag_skew`, `Acc_mag_kurt`.
- **Translational velocity**: `Vtr_mean`, `Vtr_std`, `Vtr_mad`, `Vtr_skew`, `Vtr_kurt`.
- **Jerk**: `jerk_mean`, `jerk_std`, `jerk_mad`, `jerk_skew`.

This whitelist lives in `src/models/train_baselines.py` and is applied before training. Any new model that imports the dataset automatically uses the same columns, ensuring reproducibility.

### Preprocessing pipeline

During training, every numeric feature is passed through a `SimpleImputer(strategy="median")` followed by a `StandardScaler` (see `build_models` in `src/models/train_baselines.py`). The preprocessing is encapsulated inside each scikit-learn `Pipeline`, guaranteeing that cross-validation and test evaluation use the exact same transformations.
