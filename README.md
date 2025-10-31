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
   - output: `data/results/features_dataset_5s_50olap.parquet`
   - builds sliding-window features aligned with reported RPE values

5. **Analysis & dashboards** – scripts in `src/analysis/` and `src/app/`
   - generate EDA figures (`eda_features.py`) and interactive views (`dashboard.py`)

Environment variables such as `RFP_DEFAULT_FS`, `RFP_HP_CUTOFF` and `RFP_VTR_SMOOTHING`
allow parameter tweaks without touching the code.
