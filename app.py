"""
Bangalore Traffic Prediction & Smart Route Intelligence System
Flask Backend Application
"""
import os, sys, json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import LOCATION_NAMES, WEATHER_OPTIONS, LOCATIONS
from src.predict import load_models, predict_congestion, predict_eta, get_best_departure, get_route_suggestions
from src.forecast import forecast_traffic, get_analytics_data

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.json.cls = NumpyEncoder
app.config["SECRET_KEY"] = "blr-traffic-ai-2024"

# Load models at startup
print("Loading trained models...")
MODELS = load_models()
if MODELS:
    print("Models loaded successfully!")
else:
    print("WARNING: Models not loaded. Train models first.")


@app.route("/")
def index():
    return render_template("index.html", locations=LOCATION_NAMES)


@app.route("/predictor")
def predictor():
    return render_template("predictor.html", locations=LOCATION_NAMES, weathers=WEATHER_OPTIONS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form
        now = datetime.now()
        hour = int(data.get("hour", now.hour))
        month = int(data.get("month", now.month))
        dow = now.weekday()
        params = {
            "source": data.get("source", "Koramangala"),
            "destination": data.get("destination", "Whitefield"),
            "hour": hour, "month": month, "day_of_week": dow,
            "is_weekend": 1 if dow >= 5 else 0,
            "is_holiday": int(data.get("is_holiday", 0)),
            "weather": data.get("weather", "Clear"),
            "rainfall_mm": float(data.get("rainfall", 0)),
            "temperature": float(data.get("temperature", 28)),
            "humidity": float(data.get("humidity", 65)),
            "event_flag": int(data.get("event_flag", 0)),
            "accident_flag": int(data.get("accident_flag", 0)),
            "construction_flag": int(data.get("construction_flag", 0)),
        }
        cong = predict_congestion(MODELS, params)
        eta = predict_eta(MODELS, params)
        dep = get_best_departure(MODELS, params)
        routes = get_route_suggestions(MODELS, params)
        fc = forecast_traffic(MODELS, {**params, "current_traffic_index": cong["traffic_index"]})
        src_coords = LOCATIONS.get(params["source"], (12.97, 77.59))
        dst_coords = LOCATIONS.get(params["destination"], (12.97, 77.69))
        result = {
            "congestion": cong, "eta": eta, "departure": dep, "routes": routes,
            "forecast": fc, "source_coords": list(src_coords), "dest_coords": list(dst_coords),
            "source": params["source"], "destination": params["destination"],
        }
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/analytics")
def analytics():
    try:
        data = get_analytics_data()
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/map")
def map_page():
    return render_template("map.html", locations=LOCATION_NAMES, locations_data=json.dumps(LOCATIONS))


@app.route("/api/locations")
def api_locations():
    return jsonify({"locations": LOCATIONS, "names": LOCATION_NAMES})


@app.route("/business")
def business():
    return render_template("business.html")


@app.route("/about")
def about():
    try:
        summary_path = os.path.join("models", "training_summary.json")
        with open(summary_path) as f:
            training_summary = json.load(f)
    except:
        training_summary = {}
    return render_template("about.html", training_summary=training_summary)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
