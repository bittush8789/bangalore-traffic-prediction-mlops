import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json

def train_pipeline():
    # 1. Load Data
    data_path = "data/traffic_data.csv"
    if not os.path.exists(data_path):
        print("Data not found. Generating data...")
        from generate_data import generate_bangalore_traffic_data
        generate_bangalore_traffic_data()
        
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape}")

    # 2. Feature Engineering
    # Extract hour from time
    df['hour'] = df['time'].apply(lambda x: int(x.split(':')[0]))
    
    # Label Encoding for categorical columns
    categorical_cols = ["area_name", "day_of_week", "weather", "road_type"]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    # Target encoding
    target_le = LabelEncoder()
    df['congestion_level'] = target_le.fit_transform(df['congestion_level'])
    encoders['target'] = target_le
    
    # Save encoders
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, "models/encoders.pkl")

    # Define features and target
    features = [
        "area_name", "hour", "day_of_week", "holiday", "weather", 
        "rainfall", "road_type", "event_nearby", "accident_reported", 
        "traffic_volume", "route_distance"
    ]
    X = df[features]
    y = df['congestion_level']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Model Training & MLflow Tracking
    mlflow.set_experiment("Bangalore_Traffic_Prediction")

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42)
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            mlflow.log_params(model.get_params() if hasattr(model, 'get_params') else {})
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            
            print(f"{name} Accuracy: {acc:.4f}")
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_model_name = name

    # 4. Save Best Model
    if best_model:
        print(f"Saving best model: {best_model_name} with accuracy {best_accuracy:.4f}")
        joblib.dump(best_model, "models/model.pkl")
        
        # Log best model artifact
        with mlflow.start_run(run_name="Best_Model_Final"):
            mlflow.log_param("model_type", best_model_name)
            mlflow.log_metric("best_accuracy", best_accuracy)
            mlflow.sklearn.log_model(best_model, "model")
            
    # Save feature list for inference
    with open("models/features.json", "w") as f:
        json.dump(features, f)

if __name__ == "__main__":
    train_pipeline()
