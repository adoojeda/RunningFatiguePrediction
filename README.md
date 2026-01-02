# Pipeline de Predicción de cansancio en Running

Este repositorio procesa señales capturadas por relojes inteligentes para generar conjuntos de datos listos para modelado y análisis.

## Flujo extremo a extremo

1. **Preprocesado** – `python src/data/preprocess.py`
   - entrada: `data/raw/*.csv`
   - salida: `data/processed/clean_*.parquet`
   - limpia marcas temporales, interpola FC/SpO₂ y elimina outliers evidentes
   - banderas opcionales `--input-dir`, `--output-dir` y `--files <lista>` permiten trabajar con carpetas o subconjuntos específicos
   - **Mapa de columnas del CSV bruto**

     | Posición | Descripción            | Nombre asignado |
     |---------:|------------------------|-----------------|
     | 1        | Marca temporal absoluta| `time`          |
     | 2        | Aceleración X          | `acc_x`         |
     | 3        | Aceleración Y          | `acc_y`         |
     | 4        | Aceleración Z          | `acc_z`         |
     | 5        | Gravedad X             | `grav_x`        |
     | 6        | Gravedad Y             | `grav_y`        |
     | 7        | Gravedad Z             | `grav_z`        |
     | 8        | Rotación X             | `rot_x`         |
     | 9        | Rotación Y             | `rot_y`         |
     | 10       | Rotación Z             | `rot_z`         |
     | 11       | Roll                   | `roll`          |
     | 12       | Pitch                  | `pitch`         |
     | 13       | Yaw                    | `yaw`           |
     | 14       | Frecuencia cardíaca    | `hr`            |
     | 15       | Saturación de oxígeno  | `spo2`          |

2. **Enriquecimiento cinemático** – `python src/features/kinematics.py`
   - entrada: `data/processed/clean_*.parquet`
   - salida: `data/enriched/enriched_*.parquet`
   - centra aceleraciones y calcula magnitudes dinámicas

3. **Métricas por sesión** – `python src/data/metrics.py`
   - entrada: `data/enriched/enriched_*.parquet`
   - salida: `data/results/all_sessions_metrics.parquet`
   - deriva velocidad traslacional, jerk y los componentes del índice de fatiga

4. **Extracción de características** – `python src/features/features_extraction.py`
   - entrada: `data/enriched/enriched_*.parquet` + `data/raw/rpe_file_mapping.csv`
   - salida: `data/results/features_dataset.parquet`
   - construye ventanas deslizantes de 3 s (50 % de solape) alineadas con los objetivos de fatiga/RPE

5. **Análisis y dashboards** – scripts en `src/analysis/` y `src/app/`
  - generan figuras de EDA (`eda_features.py`), reportes de ablaciones (`ablation_summary.py`) y vistas interactivas (`dashboard.py`)

## Conjunto de características utilizado en el modelado

Tras auditar cobertura, varianza y redundancia, la etapa de modelado consume una lista curada de atributos por ventana: fisiología (`hr_mean`, `spo2_mean`, `fatigue_score`), aceleraciones por eje centradas (`acc_*centered_*`), magnitudes (`acc_*`), velocidad traslacional (`vtr_*`), jerk (`jerk_*`) y orientación/balance (`roll_*`, `yaw_*`, `grav_*`). La whitelist vive en los scripts de modelado (p. ej., `src/models/run_experiments.py`) y se aplica antes del entrenamiento para garantizar reproducibilidad.

## Flujo de modelado

- **Experimentos**: `python src/models/run_experiments.py --dataset data/results/features_dataset.parquet --target fatigue_score --group runner_id --models gradient_boosting random_forest hist_gradient_boosting elasticnet xgboost catboost [--fast-grid opcional]`
  - ejecuta experimentos agrupados por corredor, lanza GridSearchCV para cada modelo y guarda métricas, predicciones, hashes y modelos serializados en `data/results/modeling/experiments/`.
- **Ablaciones de características**: `python src/models/run_ablation.py --dataset data/results/features_dataset.parquet --target fatigue_score --group runner_id --exclude-blocks orientacion fisiologia`
  - genera versiones del experimento eliminando bloques de señales (fisiología, orientación, jerk, etc.) y guarda los resultados en `data/results/modeling/ablation/<bloques>/`.

## Flujo de inferencia

`src/models/run_inference.py` (también disponible en `notebooks/inference_demo.ipynb` y en la pestaña del dashboard) permite reproducir un pipeline entrenado sobre cualquier `enriched_*.parquet` y emitir las predicciones por ventana. El resultado es idéntico al evaluado durante el entrenamiento y alimenta la pestaña de “Inferencia de fatiga” del panel Dash para visualizar el estado en tiempo (casi) real.
