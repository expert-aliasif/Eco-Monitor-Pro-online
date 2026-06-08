# Project Methodology 

This document outlines the end-to-end data pipeline and architectural decisions made for the EcoPulse Pro system.

## 1. Data Collection Pipeline
Our pipeline aggregates data from multiple high-fidelity meteorological sources:
- **Historical Data**: Fetched via the Open-Meteo Archive API (`scripts/fetch_data.py`). We extract daily max/min temperatures, precipitation sums, and wind speeds across 7 target cities spanning 5 years.
- **Real-Time Data**: Current atmospheric conditions and 5-day forecasts are pulled dynamically from the OpenWeatherMap API and Open-Meteo API using the `utils/api.py` module.

## 2. Data Preprocessing & Engineering
Raw environmental data is fundamentally noisy and requires rigorous preprocessing before model ingestion:
- **Missing Value Imputation**: Time-series gaps (NaNs) are handled using Forward Fill (`ffill`) to preserve the chronological context of the weather, followed by dropping any irreparable rows.
- **Temporal Feature Engineering**: The `date` column is decomposed into `month` and `day_of_year`.
- **Cyclical Encoding**: To help the model understand that December (Month 12) is chronologically adjacent to January (Month 1), the month feature is transformed using Sine/Cosine encoding (`month_sin`, `month_cos`).
- **Delta Transformation**: Instead of predicting absolute future temperatures (which is highly error-prone), the daily model is trained on a **Delta Target** (Tomorrow's Temp - Today's Temp). This forces the model to focus purely on relative shifts.

## 3. Machine Learning Architecture
We deploy a **Random Forest Regressor** for its robustness against non-linear interactions and lack of sensitivity to unscaled data.

### Rigorous Evaluation Split
To prevent data leakage and overfitting, we eschew the standard 80/20 split for a rigorous academic standard:
- **70% Training**: Used to build the trees.
- **15% Validation**: Used internally during Hyperparameter Tuning (`RandomizedSearchCV`) to optimize the maximum depth and number of estimators.
- **15% True Test**: Held out entirely until the final evaluation to calculate the true Mean Absolute Error (MAE) and R² score.

### Baseline Benchmarking
To mathematically prove the model's efficacy, we compare our Random Forest MAE against a **Naive Baseline Model**. In meteorology, the standard naive baseline is "Tomorrow's weather will be exactly the same as today's." Our model successfully beats this baseline, proving it has learned underlying atmospheric mechanics.

## 4. Deployment & Visualization
The trained model artifacts (`.pkl`) and extracted insights (`feature_importances.json`, `metrics.json`) are decoupled from the training scripts. The FastAPI backend (`main.py`) loads these frozen assets into memory and serves the data via high-performance RESTful API endpoints. For the presentation layer, we utilize a custom-designed HTML/CSS/JS frontend featuring modern web design principles (glassmorphism, dynamic data fetching) and Plotly.js to generate interactive charts and provide maximum transparency into both the data and the model mechanics.
