"""
Model Training Pipeline for Bangalore Traffic AI System.
Trains classification, regression, and forecasting models.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix,
                              mean_squared_error, mean_absolute_error, r2_score)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATA_DIR, MODEL_DIR, save_model, ensure_dirs
from src.feature_engineering import feature_engineering_pipeline
from src.preprocess import encode_categoricals

# Feature columns for models
CLASSIFICATION_FEATURES = [
    "hour", "month", "day_of_week", "is_weekend", "is_holiday",
    "distance_km", "signal_count", "road_capacity", "vehicle_density",
    "avg_speed", "traffic_index", "rainfall_mm", "temperature",
    "humidity", "visibility", "event_flag", "accident_flag",
    "construction_flag", "peak_hour_flag", "office_hour_flag",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "speed_per_km", "delay_ratio", "route_risk_score",
    "rain_impact_score", "event_impact_score",
    "source_area_encoded", "destination_area_encoded",
    "route_type_encoded", "weather_encoded", "time_period",
    "density_per_lane", "speed_ratio"
]

REGRESSION_FEATURES = [
    "hour", "month", "day_of_week", "is_weekend", "is_holiday",
    "distance_km", "signal_count", "road_capacity", "vehicle_density",
    "avg_speed", "traffic_index", "rainfall_mm", "temperature",
    "humidity", "visibility", "event_flag", "accident_flag",
    "construction_flag", "peak_hour_flag",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "speed_per_km", "route_risk_score", "rain_impact_score",
    "source_area_encoded", "destination_area_encoded",
    "route_type_encoded", "weather_encoded", "time_period",
    "density_per_lane"
]


def prepare_data():
    """Load and prepare data for training."""
    print("  [>>] Loading and preparing data...")

    # Load cleaned data
    cleaned_path = os.path.join(DATA_DIR, "cleaned_traffic_data.csv")
    if not os.path.exists(cleaned_path):
        print("  [!] Cleaned data not found. Running preprocessing...")
        from src.preprocess import preprocess_pipeline
        preprocess_pipeline()

    df = pd.read_csv(cleaned_path)

    # Run feature engineering
    df = feature_engineering_pipeline(df)

    # Encode categoricals
    df, encoders = encode_categoricals(df)

    # Sample data for faster training during development
    if len(df) > 20000:
        print(f"  [>>] Sampling 20,000 rows for faster training...")
        df = df.sample(20000, random_state=42)

    print(f"  [OK] Data prepared: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df, encoders


def train_classification_models(df):
    """Train congestion level classification models."""
    print("\n" + "=" * 60)
    print("  CLASSIFICATION MODEL TRAINING")
    print("  Target: congestion_level")
    print("=" * 60)

    available_features = [f for f in CLASSIFICATION_FEATURES if f in df.columns]
    X = df[available_features]
    y = df["congestion_level_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=15, min_samples_split=5,
            n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=50, max_depth=8, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            n_jobs=-1, random_state=42, verbosity=0
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=50, max_depth=6, learning_rate=0.1,
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }

        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1 Score:  {f1:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name

    print(f"\n  [**] Best Classification Model: {best_name} (Accuracy: {best_accuracy:.4f})")
    save_model(best_model, "congestion_model.pkl")

    # Save feature list
    feature_info = {"features": available_features, "best_model": best_name}
    with open(os.path.join(MODEL_DIR, "classification_features.json"), "w") as f:
        json.dump(feature_info, f, indent=2)

    # Save all results
    with open(os.path.join(MODEL_DIR, "classification_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return best_model, results


def train_regression_models(df):
    """Train travel time regression models."""
    print("\n" + "=" * 60)
    print("  REGRESSION MODEL TRAINING")
    print("  Target: travel_time_minutes")
    print("=" * 60)

    available_features = [f for f in REGRESSION_FEATURES if f in df.columns]
    X = df[available_features]
    y = df["travel_time_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=50, max_depth=15, n_jobs=-1, random_state=42
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=50, max_depth=8, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbosity=0
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=50, max_depth=6, learning_rate=0.1,
            random_state=42
        ),
        "Linear Regression": LinearRegression(n_jobs=-1),
    }

    results = {}
    best_model = None
    best_r2 = -999
    best_name = ""

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2_score": round(r2, 4),
        }

        print(f"    RMSE:     {rmse:.4f}")
        print(f"    MAE:      {mae:.4f}")
        print(f"    R² Score: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    print(f"\n  [**] Best Regression Model: {best_name} (R2: {best_r2:.4f})")
    save_model(best_model, "eta_model.pkl")

    feature_info = {"features": available_features, "best_model": best_name}
    with open(os.path.join(MODEL_DIR, "regression_features.json"), "w") as f:
        json.dump(feature_info, f, indent=2)

    with open(os.path.join(MODEL_DIR, "regression_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return best_model, results


def train_forecast_model(df):
    """Train a traffic index forecasting model using XGBoost time-series approach."""
    print("\n" + "=" * 60)
    print("  FORECASTING MODEL TRAINING")
    print("  Target: traffic_index (next hour)")
    print("=" * 60)

    # Sort by date and time
    df_sorted = df.sort_values(["date", "time"]).reset_index(drop=True)

    # Create lag features
    df_sorted["traffic_index_lag1"] = df_sorted["traffic_index"].shift(1)
    df_sorted["traffic_index_lag2"] = df_sorted["traffic_index"].shift(2)
    df_sorted["traffic_index_lag3"] = df_sorted["traffic_index"].shift(3)
    df_sorted["traffic_index_rolling_mean"] = df_sorted["traffic_index"].rolling(5, min_periods=1).mean()
    df_sorted = df_sorted.dropna()

    forecast_features = [
        "hour", "month", "day_of_week", "is_weekend", "is_holiday",
        "hour_sin", "hour_cos", "rainfall_mm", "event_flag",
        "traffic_index_lag1", "traffic_index_lag2", "traffic_index_lag3",
        "traffic_index_rolling_mean", "peak_hour_flag"
    ]

    available = [f for f in forecast_features if f in df_sorted.columns]
    X = df_sorted[available]
    y = df_sorted["traffic_index"]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = XGBRegressor(
        n_estimators=50, max_depth=8, learning_rate=0.1,
        n_jobs=-1, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # MAPE
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

    print(f"    RMSE:  {rmse:.4f}")
    print(f"    MAE:   {mae:.4f}")
    print(f"    R2:    {r2:.4f}")
    print(f"    MAPE:  {mape:.2f}%")

    save_model(model, "forecast_model.pkl")

    feature_info = {"features": available, "metrics": {
        "rmse": round(rmse, 4), "mae": round(mae, 4),
        "r2": round(r2, 4), "mape": round(mape, 2)
    }}
    with open(os.path.join(MODEL_DIR, "forecast_features.json"), "w") as f:
        json.dump(feature_info, f, indent=2)

    return model


def train_all():
    """Run the complete training pipeline."""
    print("=" * 60)
    print("  BANGALORE TRAFFIC AI - MODEL TRAINING")
    print("=" * 60)

    ensure_dirs()
    df, encoders = prepare_data()

    # Save encoders
    save_model(encoders, "label_encoders.pkl")

    # Train all models
    clf_model, clf_results = train_classification_models(df)
    reg_model, reg_results = train_regression_models(df)
    fc_model = train_forecast_model(df)

    # Save summary
    summary = {
        "classification": clf_results,
        "regression": reg_results,
        "total_training_rows": len(df),
    }
    with open(os.path.join(MODEL_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 60)

    return clf_model, reg_model, fc_model


if __name__ == "__main__":
    train_all()
