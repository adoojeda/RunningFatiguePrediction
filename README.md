# RunningFatiguePrediction – Sistema de estimación del cansancio físico en corredores

Este repositorio implementa el pipeline completo para procesar señales de Apple Watch, generar datasets por ventanas y entrenar modelos capaces de estimar el cansancio físico en corredores. Incluye scripts reproducibles, análisis exploratorio, experimentos de modelado, inferencia y un dashboard interactivo.

---

## 🧭 Flujo general (end‑to‑end)

1. **Preprocesado**  
   ```bash
   python src/data/preprocess.py
   ```
   - Entrada: `data/raw/*.csv`
   - Salida: `data/processed/clean_*.parquet`
   - Limpia marcas temporales, interpola FC/SpO₂, filtra valores fuera de rango y corrige inconsistencias.

2. **Enriquecimiento cinemático**  
   ```bash
   python src/features/kinematics.py
   ```
   - Entrada: `data/processed/clean_*.parquet`
   - Salida: `data/enriched/enriched_*.parquet`
   - Calcula aceleración centrada, magnitudes, jerk y velocidad traslacional.

3. **Métricas de sesión**  
   ```bash
   python src/data/metrics.py
   ```
   - Entrada/Salida: `data/enriched/enriched_*.parquet`
   - Añade métricas necesarias para el índice de cansancio físico.

4. **Extracción de características**  
   ```bash
   python src/features/features_extraction.py
   ```
   - Entrada: `data/enriched/enriched_*.parquet` + `data/raw/rpe_file_mapping.csv`
   - Salida: `data/results/features_dataset.parquet`
   - Ventanas de 3 s con solape del 50 %.

5. **Modelado**  
   ```bash
   python src/models/run_experiments.py \
     --dataset data/results/features_dataset.parquet \
     --target physical_fatigue_index \
     --group runner_id \
     --models gradient_boosting random_forest hist_gradient_boosting elasticnet xgboost catboost
   ```

6. **Ablaciones**  
   ```bash
   python src/models/run_ablation.py \
     --dataset data/results/features_dataset.parquet \
     --target physical_fatigue_index \
     --group runner_id \
     --exclude-blocks orientation
   ```

7. **Inferencia**  
   ```bash
   python src/models/run_inference.py --help
   ```
   - Reproduce el pipeline sobre un `enriched_*.parquet`.

---

## 🧠 Índice de cansancio físico

El objetivo principal es un índice continuo (0–1) que integra:

- Variabilidad del *jerk*  
- Variabilidad de la aceleración  
- Frecuencia cardíaca  
- Saturación de oxígeno  

Los pesos se ajustan mediante Optuna en `optimize_fatigue_weights.py`.

---

## 📊 Modelado y resultados

- Métricas: **MAE**, **RMSE**, **R²**
- Validación agrupada por corredor (GroupShuffleSplit + GroupKFold)
- Mejores resultados: modelos de **boosting**

Los resultados se guardan en:
```
data/results/modeling/experiments/runner_id_YYYYMMDD_HHMMSS/
```

---

## 🧪 Análisis exploratorio y auditoría

Scripts disponibles en `src/analysis/`:

- `eda_features.py` → figuras y correlaciones  
- `feature_audit.py` → cobertura, redundancia y ranking de features  
- `ablation_summary.py` → resumen de ablaciones  

---

## 🖥️ Dashboard interactivo

Panel en Dash para explorar señales y ejecutar inferencia:

```bash
python src/app/dashboard.py
```

- Visualización por pestañas (aceleración, jerk, velocidad, orientación, FC, SpO₂…)
- Inferencia ventana a ventana con métricas y resumen

---

## 📁 Estructura principal

```
src/
  data/        # Preprocesado y métricas
  features/    # Enriquecimiento y extracción de ventanas
  models/      # Entrenamiento, ablaciones, inferencia
  analysis/    # EDA y auditorías
  app/         # Dashboard
notebooks/     # Análisis interactivo
data/          # Raw, processed, enriched, results
```

---

## ✅ Requisitos

- Python 3.9+
- Conda/venv recomendado
- Librerías: numpy, pandas, scikit-learn, optuna, plotly, seaborn, etc.

---

## 🔮 Trabajo futuro

- Modelos temporales
- Personalización por corredor
- Inferencia en el dispositivo
