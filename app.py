from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import json
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

app = FastAPI(title="Bangalore Traffic Prediction API", version="1.0.0")

# Load model and encoders
MODEL_PATH = "models/model.pkl"
ENCODER_PATH = "models/encoders.pkl"
FEATURES_PATH = "models/features.json"

model = None
encoders = None
features = None

def load_artifacts():
    global model, encoders, features
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(ENCODER_PATH):
        encoders = joblib.load(ENCODER_PATH)
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r") as f:
            features = json.load(f)

load_artifacts()

# Prometheus Metrics
PREDICTION_COUNTER = Counter("traffic_predictions_total", "Total traffic predictions made")
LATENCY_HISTOGRAM = Histogram("prediction_latency_seconds", "Latency of traffic predictions")

class TrafficInput(BaseModel):
    area_name: str
    hour: int
    day_of_week: str
    holiday: int
    weather: str
    rainfall: float
    road_type: str
    event_nearby: int
    accident_reported: int
    traffic_volume: int
    route_distance: float

@app.get("/")
def home():
    return {"message": "Bangalore Traffic Prediction API is running!"}

@app.get("/health")
def health():
    if model is None or encoders is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(data: TrafficInput):
    start_time = time.time()
    
    if model is None or encoders is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Prepare data
        input_data = data.model_dump()
        df = pd.DataFrame([input_data])
        
        # Apply encoding
        for col, le in encoders.items():
            if col in df.columns:
                # Handle unseen labels by defaulting to 0 or first class
                try:
                    df[col] = le.transform(df[col])
                except:
                    df[col] = 0
        
        # Ensure feature order matches training
        X = df[features]
        
        # Prediction
        prediction_idx = model.predict(X)[0]
        prediction_label = encoders['target'].inverse_transform([prediction_idx])[0]
        
        # Probability if available
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(X)[0]))
        
        # Estimate travel time based on congestion (simple logic for demo)
        # In a real scenario, this would be another regression model
        base_speed = 40
        if prediction_label == "Severe": base_speed = 10
        elif prediction_label == "High": base_speed = 20
        elif prediction_label == "Medium": base_speed = 30
        
        estimated_travel_time = (data.route_distance / base_speed) * 60
        
        latency = time.time() - start_time
        LATENCY_HISTOGRAM.observe(latency)
        PREDICTION_COUNTER.inc()
        
        return {
            "congestion_level": prediction_label,
            "confidence": round(confidence, 2),
            "estimated_travel_time_minutes": round(estimated_travel_time, 2),
            "route_suggestion": "Take alternative route" if prediction_label in ["High", "Severe"] else "Optimal route selected"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
