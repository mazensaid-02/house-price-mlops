import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

TRAIN_PATH      = "data/processed/train.csv"
TEST_PATH       = "data/processed/test.csv"
STATS_PATH      = "data/processed/data_stats.json"
MLFLOW_URI      = "http://localhost:5000"
EXPERIMENT_NAME = "house-price-prediction"
TARGET_COL      = "MedHouseVal"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def load_data():
    print("📂 Loading processed data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    X_train  = train_df.drop(TARGET_COL, axis=1)
    y_train  = train_df[TARGET_COL]
    X_test   = test_df.drop(TARGET_COL, axis=1)
    y_test   = test_df[TARGET_COL]
    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def evaluate(y_true, y_pred):
    return {
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        "r2":   round(r2_score(y_true, y_pred), 4)
    }

def log_run(name, model, params, X_train, y_train, X_test, y_test):
    print(f"\n🧪 Running experiment: {name}")
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        with open(STATS_PATH) as f:
            stats = json.load(f)
        mlflow.log_dict(stats, "data_stats.json")
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows",  len(X_test))
        print(f"   RMSE: {metrics['rmse']}")
        print(f"   MAE:  {metrics['mae']}")
        print(f"   R2:   {metrics['r2']}")
        print(f"   Logged to MLflow")
        return metrics, mlflow.active_run().info.run_id

def main():
    print("=" * 50)
    print("  MODEL TRAINING PIPELINE")
    print("=" * 50)

    X_train, y_train, X_test, y_test = load_data()
    results = []

    # 1. Linear Regression baseline
    metrics, run_id = log_run(
        "LinearRegression-baseline",
        LinearRegression(),
        {"model_type": "LinearRegression"},
        X_train, y_train, X_test, y_test
    )
    results.append({"name": "LinearRegression", "metrics": metrics, "run_id": run_id})

    # 2. Random Forest
    metrics, run_id = log_run(
        "RandomForest-v1",
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 10, "random_state": 42},
        X_train, y_train, X_test, y_test
    )
    results.append({"name": "RandomForest", "metrics": metrics, "run_id": run_id})

    # 3. XGBoost
    metrics, run_id = log_run(
        "XGBoost-v1",
        XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42, verbosity=0),
        {"model_type": "XGBoost", "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8},
        X_train, y_train, X_test, y_test
    )
    results.append({"name": "XGBoost", "metrics": metrics, "run_id": run_id})

    # Summary
    print("\n" + "=" * 50)
    print("  EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    print("-" * 50)
    for r in results:
        m = r["metrics"]
        print(f"{r['name']:<25} {m['rmse']:>8} {m['mae']:>8} {m['r2']:>8}")

    best = min(results, key=lambda x: x["metrics"]["rmse"])
    print(f"\nBest model: {best['name']} (RMSE: {best['metrics']['rmse']})")
    print(f"Run ID: {best['run_id']}")
    print(f"\nView results at: http://localhost:5000")
    print("\nAll experiments logged to MLflow")

if __name__ == "__main__":
    main()
