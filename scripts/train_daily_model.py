import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
import json

def main():
    print("Loading dataset...")
    df = pd.read_csv("data/city_daily_weather.csv")
    
    # grabbing data and filling holes (forward fill) so the model doesn't cry
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    # saving the dates and size so we can brag about it in the dashboard
    data_start = str(df['date'].min())[:10]
    data_end = str(df['date'].max())[:10]
    total_samples = len(df)
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
    
    # breaking date into month and day so the model knows what time of year it is
    # month 12 and month 1 should feel "close" to the model, hence sin/cos trick
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year'] = df['date'].dt.dayofyear
    
    print("Engineering features...")
    df = df.sort_values(by=['city', 'date']).reset_index(drop=True)
    
    # what we actually want to guess (tomorrow's weather)
    df['target_temp_max'] = df.groupby('city')['temp_max'].shift(-1)
    df['target_temp_min'] = df.groupby('city')['temp_min'].shift(-1)
    df['target_precip'] = df.groupby('city')['precipitation'].shift(-1)
    
    # Create DELTA targets
    df['target_delta_temp_max'] = df['target_temp_max'] - df['temp_max']
    df['target_delta_temp_min'] = df['target_temp_min'] - df['temp_min']
    df['target_delta_precip'] = df['target_precip'] - df['precipitation']
    
    # Drop NaNs created by shift
    df = df.dropna().reset_index(drop=True)
    
    # Encode city strings
    le = LabelEncoder()
    df['city_encoded'] = le.fit_transform(df['city'])
    
    os.makedirs("model", exist_ok=True)
    with open("model/city_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    
    # Define Features
    features = ['city_encoded', 'month_sin', 'month_cos', 'day_of_year', 'temp_max', 'temp_min', 'precipitation', 'wind_speed']
    X = df[features]
    y = df[['target_delta_temp_max', 'target_delta_temp_min', 'target_delta_precip']]
    
    # dummy check: what if we just guessed tomorrow is exactly like today?
    baseline_preds = np.zeros_like(y)
    baseline_mae = mean_absolute_error(y, baseline_preds)
    
    print(f"Dataset Size: {len(X)}")
    print(f"Naive Baseline MAE: {baseline_mae:.3f}")
    
    # breaking data into chunks so we don't cheat on the test
    # saving 30% for grading
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    # Split the 30% into 15% val and 15% test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    print("Starting Hyperparameter Tuning...")
    # Basic parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=5, cv=3, verbose=1, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    
    print(f"Best parameters found: {search.best_params_}")
    
    best_model = search.best_estimator_
    
    # Evaluate on true test set
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Test MAE: {mae:.3f} (Baseline was {baseline_mae:.3f})")
    print(f"Test R^2: {r2:.3f}")
    
    # Feature Importances extraction
    feature_importances = best_model.feature_importances_
    fi_dict = {feat: float(imp) for feat, imp in zip(features, feature_importances)}
    
    # Save the model
    with open("model/daily_weather_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
        
    # Save metrics
    metrics = {
        "daily_model": {
            "mae": round(mae, 3),
            "r2_score": round(r2, 3),
            "baseline_mae": round(baseline_mae, 3),
            "best_params": search.best_params_
        },
        "dataset": {
            "total_samples": total_samples,
            "date_range": f"{data_start} to {data_end}",
            "split": "70% Train, 15% Val, 15% Test"
        }
    }
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    with open("model/feature_importances.json", "w") as f:
        json.dump(fi_dict, f, indent=4)
        
    print("Delta Model saved to model/daily_weather_model.pkl")
    print("Metrics and Feature Importances saved to model/")

if __name__ == "__main__":
    main()
