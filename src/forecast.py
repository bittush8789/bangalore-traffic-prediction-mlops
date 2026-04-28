"""
Forecasting module for Bangalore Traffic AI System.
"""
import os, sys, json, numpy as np, pandas as pd
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import MODEL_DIR, DATA_DIR, load_model

def forecast_traffic(models, params):
    """Forecast traffic index for next hours."""
    hour = params.get("hour", 12)
    month = params.get("month", 6)
    dow = params.get("day_of_week", 2)
    is_we = params.get("is_weekend", 0)
    is_hol = params.get("is_holiday", 0)
    rain = params.get("rainfall_mm", 0)
    ev = params.get("event_flag", 0)
    current_ti = params.get("current_traffic_index", 50)
    forecasts = []
    prev = [current_ti, current_ti * 0.95, current_ti * 0.9]
    for offset in [1, 2, 3, 4]:
        fh = (hour + offset) % 24
        hs = np.sin(2 * np.pi * fh / 24)
        hc = np.cos(2 * np.pi * fh / 24)
        ph = 1 if (8 <= fh <= 11 or 17 <= fh <= 20) else 0
        rm = np.mean(prev[-3:])
        fd = {"hour": fh, "month": month, "day_of_week": dow, "is_weekend": is_we,
              "is_holiday": is_hol, "hour_sin": hs, "hour_cos": hc, "rainfall_mm": rain,
              "event_flag": ev, "traffic_index_lag1": prev[-1], "traffic_index_lag2": prev[-2] if len(prev) >= 2 else prev[-1],
              "traffic_index_lag3": prev[-3] if len(prev) >= 3 else prev[-1],
              "traffic_index_rolling_mean": rm, "peak_hour_flag": ph}
        features = models.get("fc_features", list(fd.keys()))
        row = {f: fd.get(f, 0) for f in features}
        X = pd.DataFrame([row])
        try:
            pred_ti = models["forecast"].predict(X)[0]
        except:
            pred_ti = current_ti + np.random.uniform(-5, 5)
        pred_ti = np.clip(pred_ti, 5, 100)
        prev.append(pred_ti)
        if pred_ti < 30: level = "Low"
        elif pred_ti < 55: level = "Medium"
        elif pred_ti < 75: level = "High"
        else: level = "Severe"
        forecasts.append({"hour": int(fh), "traffic_index": float(round(float(pred_ti), 1)), "congestion_level": level,
                          "time_label": f"{fh:02d}:00"})
    return {"forecasts": forecasts, "current_traffic_index": float(round(float(current_ti), 1))}

def get_analytics_data():
    """Get analytics data from the dataset for dashboard charts."""
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_traffic_data.csv"))
        # Sample for dashboard performance if dataset is large
        if len(df) > 50000:
            df = df.sample(50000, random_state=42)
    except Exception as e:
        print(f"Error loading analytics data: {e}")
        return {}

    if "time" in df.columns:
        df["hour"] = df["time"].apply(lambda x: int(str(x).split(":")[0]))
    
    hourly = df.groupby("hour")["traffic_index"].mean().to_dict()
    hourly = {str(k): float(round(v, 1)) for k, v in hourly.items()}
    
    top_routes = df.groupby(["source_area", "destination_area"])["traffic_index"].mean()
    top_routes = top_routes.sort_values(ascending=False).head(10)
    top_routes_list = [{"source": idx[0], "destination": idx[1], "avg_traffic": float(round(val, 1))} for idx, val in top_routes.items()]
    
    weather_impact = df.groupby("weather")["traffic_index"].mean().to_dict()
    weather_impact = {str(k): float(round(v, 1)) for k, v in weather_impact.items()}
    
    weekday = df.groupby("day_of_week")["traffic_index"].mean().to_dict()
    weekday = {str(k): float(round(v, 1)) for k, v in weekday.items()}
    
    cong_dist = {str(k): int(v) for k, v in df["congestion_level"].value_counts().to_dict().items()}
    
    area_traffic_raw = df.groupby("source_area")["traffic_index"].mean().sort_values(ascending=False).head(10).to_dict()
    area_traffic = {str(k): float(round(v, 1)) for k, v in area_traffic_raw.items()}
    
    stats = {
        "total_records": int(len(df)), 
        "avg_traffic_index": float(round(df["traffic_index"].mean(), 1)),
        "avg_travel_time": float(round(df["travel_time_minutes"].mean(), 1)),
        "avg_speed": float(round(df["avg_speed"].mean(), 1)),
        "severe_pct": float(round(len(df[df["congestion_level"] == "Severe"]) / len(df) * 100, 1))
    }
    
    return {
        "hourly_traffic": hourly, 
        "top_congested_routes": top_routes_list, 
        "weather_impact": weather_impact,
        "weekday_traffic": weekday, 
        "congestion_distribution": cong_dist, 
        "area_traffic": area_traffic, 
        "stats": stats
    }
