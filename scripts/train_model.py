import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

def main():
    """training the big rainfall model so farmers know what's coming."""
    # grabbing the data
    df = pd.read_csv("data/rainfall.csv")
    
    # Clean column names (IMPORTANT)
    df.columns = df.columns.str.strip()
    
    # Rename columns if needed (adjust if different)
    df.rename(columns={
        "Year": "year",
        "Month": "month",
        "Rainfall_mm": "rainfall"
    }, inplace=True)
    
    # the sin/cos trick so month 12 and 1 are neighbors
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Features & Target
    X = df[['year', 'month_sin', 'month_cos']]
    y = df['rainfall']
    
    # 70 / 15 / 15 Split
    # First split off 30% for val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    # Split the 30% into 15% val and 15% test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    # dummy check: what if we just guessed the average rainfall?
    baseline_preds = np.full_like(y_test, y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    
    print("Starting Hyperparameter Tuning...")
    # Parameter grid for Random Forest
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
    
    # Save model
    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    print("Model trained successfully!")
    
    #Test 
    sample_month = 7
    sample = [[2020, np.sin(2 * np.pi * sample_month / 12), np.cos(2 * np.pi * sample_month / 12)]] # july 2020
    prediction = best_model.predict(sample)
    print("Test Prediction (July 2020):", prediction)

if __name__ == "__main__":
    main()
