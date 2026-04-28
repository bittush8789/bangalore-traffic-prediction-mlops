"""
Prediction module for Bangalore Traffic AI System.
"""
import os, sys, json, numpy as np, pandas as pd
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import MODEL_DIR, LOCATIONS, LOCATION_NAMES, CONGESTION_LABELS, load_model, get_distance, IT_AREAS

CONGESTION_MAP = {0: "Low", 1: "Medium", 2: "High", 3: "Severe"}

def load_models():
    models = {}
    files = {
        "congestion": "congestion_model.pkl",
        "eta": "eta_model.pkl",
        "forecast": "forecast_model.pkl",
        "encoders": "label_encoders.pkl"
    }
    
    for key, filename in files.items():
        try:
            models[key] = load_model(filename)
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")

    try:
        with open(os.path.join(MODEL_DIR, "classification_features.json")) as f:
            models["clf_features"] = json.load(f)["features"]
        with open(os.path.join(MODEL_DIR, "regression_features.json")) as f:
            models["reg_features"] = json.load(f)["features"]
        with open(os.path.join(MODEL_DIR, "forecast_features.json")) as f:
            models["fc_features"] = json.load(f)["features"]
    except Exception as e:
        print(f"Warning: Could not load feature configs: {e}")
        
    return models

def encode_value(encoders, col, value):
    if col in encoders:
        le = encoders[col]
        if value in le.classes_:
            return le.transform([value])[0]
    return 0

def build_features(params, encoders, feature_list):
    hour = params.get("hour", 12)
    month = params.get("month", 6)
    dow = params.get("day_of_week", 2)
    is_we = params.get("is_weekend", 0)
    is_hol = params.get("is_holiday", 0)
    src = params.get("source", "Koramangala")
    dst = params.get("destination", "Whitefield")
    weather = params.get("weather", "Clear")
    rain = params.get("rainfall_mm", 0)
    temp = params.get("temperature", 28)
    hum = params.get("humidity", 65)
    ev = params.get("event_flag", 0)
    acc = params.get("accident_flag", 0)
    con = params.get("construction_flag", 0)
    dist = get_distance(src, dst)
    sig = np.random.randint(3, 12)
    cap = np.random.choice([4, 6, 8])
    base = 30
    if 8 <= hour <= 11: base += 25
    elif 17 <= hour <= 20: base += 30
    if is_we: base -= 10
    if rain > 5: base += 15
    elif rain > 0: base += 8
    if ev: base += 15
    if src in IT_AREAS or dst in IT_AREAS:
        if 8 <= hour <= 10 or 17 <= hour <= 19: base += 10
    ti = np.clip(base, 5, 100)
    if ti >= 80: spd = np.random.uniform(5, 12)
    elif ti >= 60: spd = np.random.uniform(12, 22)
    elif ti >= 40: spd = np.random.uniform(22, 35)
    else: spd = np.random.uniform(35, 55)
    mx = spd + np.random.uniform(5, 15)
    mn = max(2, spd - np.random.uniform(5, 10))
    vd = int(ti * np.random.uniform(15, 25))
    vis = 10 if weather == "Clear" else (5 if rain > 5 else 7)
    ph = 1 if (8 <= hour <= 11 or 17 <= hour <= 20) else 0
    oh = 1 if (9 <= hour <= 18 and not is_we) else 0
    hs = np.sin(2 * np.pi * hour / 24)
    hc = np.cos(2 * np.pi * hour / 24)
    ms = np.sin(2 * np.pi * month / 12)
    mc = np.cos(2 * np.pi * month / 12)
    spk = spd / dist if dist > 0 else 0
    sr = spd / mx if mx > 0 else 0
    it = (dist / 45) * 60
    dr = ((dist / spd) * 60) / it if it > 0 and spd > 0 else 1
    rrs = min(100, ti * 0.3 + sig * 2 + acc * 20 + con * 10 + ev * 15 + rain * 0.3)
    ris = (0.6 if rain > 3 else 0.3 if rain > 0 else 0) * ti * 0.5
    eis = ev * ti * 0.4
    dpl = vd / cap if cap > 0 else vd
    if 5 <= hour <= 8: tp = 1
    elif 9 <= hour <= 11: tp = 2
    elif 12 <= hour <= 16: tp = 3
    elif 17 <= hour <= 20: tp = 4
    elif 21 <= hour <= 23: tp = 5
    else: tp = 0
    se = encode_value(encoders, "source_area", src)
    de = encode_value(encoders, "destination_area", dst)
    rt = np.random.choice(["Highway", "Main Road", "Inner Road", "Service Road"])
    re = encode_value(encoders, "route_type", rt)
    we = encode_value(encoders, "weather", weather)
    fd = {"hour": hour, "month": month, "day_of_week": dow, "is_weekend": is_we, "is_holiday": is_hol,
          "distance_km": dist, "signal_count": sig, "road_capacity": cap, "vehicle_density": vd,
          "avg_speed": spd, "traffic_index": ti, "rainfall_mm": rain, "temperature": temp,
          "humidity": hum, "visibility": vis, "event_flag": ev, "accident_flag": acc,
          "construction_flag": con, "peak_hour_flag": ph, "office_hour_flag": oh,
          "hour_sin": hs, "hour_cos": hc, "month_sin": ms, "month_cos": mc,
          "speed_per_km": spk, "speed_range": mx - mn, "speed_ratio": sr, "delay_ratio": dr,
          "route_risk_score": rrs, "rain_impact_score": ris, "event_impact_score": eis,
          "source_area_encoded": se, "destination_area_encoded": de,
          "route_type_encoded": re, "weather_encoded": we, "time_period": tp,
          "density_per_lane": dpl, "density_speed_ratio": vd / spd if spd > 0 else 0}
    row = {f: fd.get(f, 0) for f in feature_list}
    return pd.DataFrame([row]), fd

def predict_congestion(models, params):
    if "encoders" not in models or "clf_features" not in models or "congestion" not in models:
        raise ValueError("Congestion model or encoders not loaded. Please train models first.")
    X, fd = build_features(params, models["encoders"], models["clf_features"])
    pred = int(models["congestion"].predict(X)[0])
    cong = CONGESTION_MAP.get(pred, "Unknown")
    conf = 85.0
    if hasattr(models["congestion"], "predict_proba"):
        conf = float(round(max(models["congestion"].predict_proba(X)[0]) * 100, 1))
    return {"congestion_level": cong, "confidence": conf, "traffic_index": float(round(fd["traffic_index"], 1))}

def predict_eta(models, params):
    if "encoders" not in models or "reg_features" not in models or "eta" not in models:
        raise ValueError("ETA model or encoders not loaded. Please train models first.")
    X, fd = build_features(params, models["encoders"], models["reg_features"])
    eta = max(1.0, float(round(models["eta"].predict(X)[0], 1)))
    return {"eta_minutes": eta, "distance_km": float(fd["distance_km"]), "avg_speed": float(round(fd["avg_speed"], 1))}

def get_best_departure(models, params):
    if "encoders" not in models or "clf_features" not in models or "congestion" not in models:
        return {"error": "Models not loaded"}
    best_h, best_t = params.get("hour", 12), 100
    results = []
    for off in range(5):
        th = (params.get("hour", 12) + off) % 24
        tp = {**params, "hour": th}
        X, fd = build_features(tp, models["encoders"], models["clf_features"])
        p = int(models["congestion"].predict(X)[0])
        t = float(fd["traffic_index"])
        results.append({"hour": int(th), "congestion": CONGESTION_MAP.get(p, "Unknown"), "traffic_index": float(round(t, 1))})
        if t < best_t:
            best_t = t
            best_h = th
    return {"best_departure_hour": int(best_h), "best_traffic_index": float(round(best_t, 1)), "hourly_forecast": results}

def get_route_suggestions(models, params):
    if "encoders" not in models or "reg_features" not in models or "eta" not in models:
        return {"routes": []}
    routes = []
    for rt in ["Highway", "Main Road", "Inner Road"]:
        tp = {**params}
        X, fd = build_features(tp, models["encoders"], models["reg_features"])
        eta = max(1.0, float(round(models["eta"].predict(X)[0] + np.random.uniform(-5, 5), 1)))
        cp = int(models["congestion"].predict(build_features(tp, models["encoders"], models["clf_features"])[0])[0])
        routes.append({"route_type": rt, "eta_minutes": eta, "congestion": CONGESTION_MAP.get(cp, "Medium"),
                       "distance_km": float(fd["distance_km"]), "risk_score": float(round(fd["route_risk_score"], 1)), "recommended": False})
    routes.sort(key=lambda x: x["eta_minutes"])
    if routes:
        routes[0]["recommended"] = True
    return {"routes": routes}
