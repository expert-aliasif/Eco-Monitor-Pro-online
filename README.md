# EcoPulse Pro 🌍

## Why I Built This

Growing up in Toba Tek Singh, I watched summers get progressively hotter 
every year. Farmers in my area — depend on 
accurate rainfall predictions for crop planning. Most weather apps show 
data for major cities only and don't explain *why* they predict what they 
predict.

This project is my attempt to build something transparent, local, and 
honest about its own uncertainty — hence the ±confidence intervals on 
every prediction.

## 🎓 Academic Justification & Literature Review

### Algorithm Selection: Why Random Forest?
In the domain of multivariate meteorological forecasting, deep learning models such as Long Short-Term Memory (LSTM) networks and statistical approaches like AutoRegressive Integrated Moving Average (ARIMA) are frequently utilized. However, for this project, a **Random Forest Regressor** was selected for both the Daily Weather and Climate Rainfall modules due to the following academically substantiated reasons:

1. **Non-Linear Tabular Efficacy**: Empirical studies consistently show that tree-based ensemble methods (like Random Forest and XGBoost) outperform un-tuned deep neural networks on small-to-medium tabular datasets. They naturally handle non-linear interactions between variables (e.g., Temperature, Precipitation, Day of Year) without requiring complex feature scaling or extensive normalization (Grinsztajn et al., 2022).
2. **Built-in Uncertainty Estimation**: Unlike standard point-estimate models, Random Forest is an ensemble of distinct decision trees. By extracting the variance across all individual estimators (`model.estimators_`), we can calculate the standard deviation of predictions, providing a robust scientific Confidence Interval for our weather forecasts.
3. **Interpretability & Feature Importance**: Environmental dashboards must be transparent. Random Forest inherently calculates Gini importance (Mean Decrease Impurity), allowing us to computationally prove which meteorological factors drive the prediction. This directly powers the Feature Importance visualizations in our dashboard.

## ⚙️ Setup Instructions

### 1. Environment Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Keys

The application requires an OpenWeather API key to function.

1. Rename the provided `.env.example` to `.env`.
2. Add your OpenWeather API key into the `.env` file like this:

```
OPENWEATHER_API_KEY=your_api_key_here
```

### 3. Model Training Pipeline (Rigorous)

To demonstrate rigorous academic methodology, our pipeline utilizes a **70/15/15 Train/Validation/Test Split** alongside Hyperparameter Tuning (RandomizedSearchCV) and explicit handling of missing values (forward fill). 

Run the scripts in the following order:

1. **Fetch Data:** (Optional, if `city_daily_weather.csv` is missing)
```bash
python scripts/fetch_data.py
```

2. **Train the Advanced Daily Weather Model:**
```bash
python scripts/train_daily_model.py
```
*This will execute the hyperparameter tuning, evaluate against a Naive Baseline, and save `daily_weather_model.pkl`, `feature_importances.json`, and `metrics.json`.*

3. **Train the Rainfall Model:**
```bash
python scripts/train_model.py
```

### 4. Running the Dashboard

Once your models are trained, launch the FastAPI backend from the root directory:

```bash
uvicorn main:app --reload
```

Then, open your web browser and navigate to `http://localhost:8000` to access the full-stack web application.

- **Modern Web Application:** A fully custom, sleek HTML/CSS/JS frontend powered by a robust FastAPI backend.
- **Exploratory Data Analysis (EDA):** Metrics are computed and made available via the API.
- **Model Confidence Intervals:** The weather predictor showcases the standard deviation derived directly from the underlying Random Forest estimators.
- **Academic Rigor:** View real-time model evaluation metrics (MAE, R² score, and Baseline Comparison) directly within the dashboard sidebar.
- **Global Dashboard:** Instantly visualizes the Weather, Rainfall, and AQI for a selected city and year on a single view using Plotly.js.

## 👨‍💻 Author

**Ali Javed**  
BS Computer Science — Final Year Project (2026)  

## Acknowledgements

Special thanks to my family for tolerating me debugging at 2am.
