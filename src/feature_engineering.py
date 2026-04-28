"""
Advanced Feature Engineering for Bangalore Traffic AI System.
Creates derived features for improved model performance.
"""
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATA_DIR


def extract_hour(df):
    """Extract hour from time column."""
    if "time" in df.columns:
        df["hour"] = df["time"].apply(lambda x: int(str(x).split(":")[0]))
    return df


def add_cyclical_features(df):
    """Add sin/cos cyclical encoding for hour and month."""
    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_speed_features(df):
    """Add speed-derived features."""
    df["speed_per_km"] = np.where(
        df["distance_km"] > 0,
        df["avg_speed"] / df["distance_km"],
        0
    )
    df["speed_range"] = df["max_speed"] - df["min_speed"]
    df["speed_ratio"] = np.where(
        df["max_speed"] > 0,
        df["avg_speed"] / df["max_speed"],
        0
    )
    return df


def add_delay_features(df):
    """Add delay and congestion ratio features."""
    # Ideal travel time at free-flow speed (~45 km/h)
    df["ideal_travel_time"] = (df["distance_km"] / 45) * 60
    df["delay_ratio"] = np.where(
        df["ideal_travel_time"] > 0,
        df["travel_time_minutes"] / df["ideal_travel_time"],
        1
    )
    df["excess_delay"] = df["travel_time_minutes"] - df["ideal_travel_time"]
    df["waiting_ratio"] = np.where(
        df["travel_time_minutes"] > 0,
        df["waiting_time_minutes"] / df["travel_time_minutes"],
        0
    )
    return df


def add_route_risk_score(df):
    """Compute a route risk score based on multiple factors."""
    df["route_risk_score"] = (
        df["traffic_index"] * 0.3 +
        df["signal_count"] * 2 +
        df["accident_flag"] * 20 +
        df["construction_flag"] * 10 +
        df["event_flag"] * 15 +
        (100 - df["visibility"]) * 0.5 +
        df["rainfall_mm"] * 0.3
    )
    df["route_risk_score"] = df["route_risk_score"].clip(0, 100)
    return df


def add_rain_impact_score(df):
    """Calculate rain impact on traffic."""
    rain_factor = df["rainfall_mm"].apply(
        lambda x: 0 if x == 0 else
                  0.3 if x < 3 else
                  0.6 if x < 10 else
                  0.85 if x < 25 else 1.0
    )
    df["rain_impact_score"] = rain_factor * df["traffic_index"] * 0.5
    return df


def add_event_impact_score(df):
    """Calculate event impact on traffic."""
    df["event_impact_score"] = df["event_flag"] * df["traffic_index"] * 0.4
    return df


def add_density_features(df):
    """Add vehicle density derived features."""
    df["density_per_lane"] = np.where(
        df["road_capacity"] > 0,
        df["vehicle_density"] / df["road_capacity"],
        df["vehicle_density"]
    )
    df["density_speed_ratio"] = np.where(
        df["avg_speed"] > 0,
        df["vehicle_density"] / df["avg_speed"],
        0
    )
    return df


def add_time_features(df):
    """Add time-based categorical features."""
    if "hour" in df.columns:
        conditions = [
            df["hour"].between(5, 8),
            df["hour"].between(9, 11),
            df["hour"].between(12, 16),
            df["hour"].between(17, 20),
            df["hour"].between(21, 23),
        ]
        choices = [1, 2, 3, 4, 5]  # early_morning, morning_peak, afternoon, evening_peak, night
        df["time_period"] = np.select(conditions, choices, default=0)
    return df


def feature_engineering_pipeline(df=None):
    """Run the full feature engineering pipeline."""
    print("=" * 60)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    if df is None:
        cleaned_path = os.path.join(DATA_DIR, "cleaned_traffic_data.csv")
        df = pd.read_csv(cleaned_path)
        print(f"  [OK] Loaded cleaned data: {len(df):,} rows")

    df = extract_hour(df)
    print("  [OK] Extracted hour")

    df = add_cyclical_features(df)
    print("  [OK] Added cyclical features (hour_sin, hour_cos, month_sin, month_cos)")

    df = add_speed_features(df)
    print("  [OK] Added speed features (speed_per_km, speed_range, speed_ratio)")

    df = add_delay_features(df)
    print("  [OK] Added delay features (delay_ratio, excess_delay, waiting_ratio)")

    df = add_route_risk_score(df)
    print("  [OK] Added route_risk_score")

    df = add_rain_impact_score(df)
    print("  [OK] Added rain_impact_score")

    df = add_event_impact_score(df)
    print("  [OK] Added event_impact_score")

    df = add_density_features(df)
    print("  [OK] Added density features")

    df = add_time_features(df)
    print("  [OK] Added time_period feature")

    # Save engineered data
    eng_path = os.path.join(DATA_DIR, "engineered_traffic_data.csv")
    df.to_csv(eng_path, index=False)
    print(f"\n  [OK] Engineered dataset saved: {eng_path}")
    print(f"  [OK] Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("=" * 60)

    return df


if __name__ == "__main__":
    feature_engineering_pipeline()
