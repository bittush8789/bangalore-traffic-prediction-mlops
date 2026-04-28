"""
Bangalore Traffic Synthetic Data Generator
Generates 250,000+ rows of realistic traffic data for model training.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    LOCATIONS, LOCATION_NAMES, IT_AREAS, SHOPPING_AREAS, SCHOOL_ZONES,
    WEATHER_OPTIONS, EVENT_TYPES, ROUTE_TYPES, HOLIDAYS,
    get_distance, ensure_dirs, DATA_DIR
)

np.random.seed(42)
random.seed(42)

NUM_ROWS = 300000  # Generate 300K rows


def generate_timestamps(n):
    """Generate realistic timestamps spanning 3 years."""
    start = datetime(2023, 1, 1, 0, 0)
    end = datetime(2025, 12, 31, 23, 0)
    total_hours = int((end - start).total_seconds() / 3600)
    timestamps = []
    for _ in range(n):
        random_hour = random.randint(0, total_hours)
        ts = start + timedelta(hours=random_hour)
        # Bias towards peak hours
        hour = ts.hour
        if random.random() < 0.4:
            # Push towards morning or evening peak
            if random.random() < 0.5:
                ts = ts.replace(hour=random.choice([7, 8, 9, 10, 11]))
            else:
                ts = ts.replace(hour=random.choice([17, 18, 19, 20]))
        timestamps.append(ts)
    return timestamps


def get_weather_for_month(month):
    """Realistic Bangalore weather by month."""
    # Bangalore rainy: Jun-Oct, dry: Dec-Mar
    if month in [6, 7, 8, 9]:
        weights = [0.15, 0.20, 0.25, 0.20, 0.05, 0.15]
    elif month in [10, 11]:
        weights = [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]
    elif month in [12, 1, 2, 3]:
        weights = [0.55, 0.25, 0.05, 0.02, 0.10, 0.03]
    else:
        weights = [0.40, 0.30, 0.15, 0.05, 0.05, 0.05]
    return np.random.choice(WEATHER_OPTIONS, p=weights)


def get_temperature(month, hour):
    """Realistic Bangalore temperature."""
    base = {1: 22, 2: 24, 3: 27, 4: 30, 5: 29, 6: 25,
            7: 24, 8: 24, 9: 24, 10: 24, 11: 23, 12: 22}
    t = base.get(month, 25)
    if 6 <= hour <= 9:
        t -= random.uniform(2, 5)
    elif 11 <= hour <= 15:
        t += random.uniform(1, 4)
    elif 18 <= hour <= 22:
        t -= random.uniform(1, 3)
    else:
        t -= random.uniform(3, 6)
    return round(t + random.uniform(-1, 1), 1)


def compute_traffic_index(hour, day_of_week, is_weekend, is_holiday, weather,
                           source, destination, event_flag, accident_flag,
                           construction_flag, month):
    """Compute a realistic traffic index (0-100)."""
    base = 30

    # Time of day impact
    if 8 <= hour <= 11:
        base += random.uniform(20, 35)  # Morning peak
    elif 17 <= hour <= 20:
        base += random.uniform(25, 40)  # Evening peak (worse)
    elif 12 <= hour <= 16:
        base += random.uniform(5, 15)   # Afternoon moderate
    elif 21 <= hour <= 23:
        base += random.uniform(-5, 5)   # Night
    else:
        base += random.uniform(-15, -5) # Late night

    # Weekend effect
    if is_weekend:
        if source in SHOPPING_AREAS or destination in SHOPPING_AREAS:
            base += random.uniform(5, 15)  # Shopping rush
        else:
            base -= random.uniform(10, 20) # Less office traffic
    
    # Holiday effect
    if is_holiday:
        base -= random.uniform(5, 15)
        if source in SHOPPING_AREAS or destination in SHOPPING_AREAS:
            base += random.uniform(10, 20)

    # IT area effect
    if source in IT_AREAS or destination in IT_AREAS:
        if not is_weekend and not is_holiday:
            if 8 <= hour <= 10 or 17 <= hour <= 19:
                base += random.uniform(10, 25)

    # School zone morning
    if source in SCHOOL_ZONES or destination in SCHOOL_ZONES:
        if 7 <= hour <= 9 and not is_weekend and not is_holiday:
            base += random.uniform(5, 12)

    # Weather impact
    weather_impact = {
        "Clear": 0, "Cloudy": 2,
        "Light Rain": random.uniform(8, 15),
        "Heavy Rain": random.uniform(15, 30),
        "Fog": random.uniform(10, 18),
        "Thunderstorm": random.uniform(20, 35)
    }
    base += weather_impact.get(weather, 0)

    # Events
    if event_flag:
        base += random.uniform(10, 25)

    # Accidents
    if accident_flag:
        base += random.uniform(15, 30)

    # Construction
    if construction_flag:
        base += random.uniform(5, 15)

    # Silk Board junction - always high
    if source == "Silk Board" or destination == "Silk Board":
        base += random.uniform(5, 15)

    # ORR corridor
    if source == "Outer Ring Road" or destination == "Outer Ring Road":
        if 8 <= hour <= 10 or 17 <= hour <= 20:
            base += random.uniform(5, 12)

    return np.clip(base, 5, 100)


def compute_speeds(traffic_index, distance_km):
    """Compute realistic speeds based on traffic."""
    if traffic_index >= 80:
        avg = random.uniform(5, 12)
    elif traffic_index >= 60:
        avg = random.uniform(12, 22)
    elif traffic_index >= 40:
        avg = random.uniform(22, 35)
    else:
        avg = random.uniform(35, 55)

    max_speed = avg + random.uniform(5, 20)
    min_speed = max(2, avg - random.uniform(5, 15))
    return round(avg, 1), round(max_speed, 1), round(min_speed, 1)


def compute_congestion_level(traffic_index):
    """Map traffic index to congestion label."""
    if traffic_index < 30:
        return "Low"
    elif traffic_index < 55:
        return "Medium"
    elif traffic_index < 75:
        return "High"
    else:
        return "Severe"


def generate_dataset():
    """Generate the complete synthetic dataset."""
    print("=" * 60)
    print("  BANGALORE TRAFFIC SYNTHETIC DATA GENERATOR")
    print("=" * 60)
    print(f"  Generating {NUM_ROWS:,} rows of realistic traffic data...")
    print()

    ensure_dirs()

    timestamps = generate_timestamps(NUM_ROWS)
    print(f"  [OK] Generated {NUM_ROWS:,} timestamps")

    records = []

    for i, ts in enumerate(timestamps):
        if (i + 1) % 50000 == 0:
            print(f"  [>>] Processing row {i+1:,} / {NUM_ROWS:,} ...")

        date = ts.date()
        time_str = ts.strftime("%H:%M")
        year = ts.year
        month = ts.month
        day = ts.day
        hour = ts.hour
        day_of_week = ts.weekday()  # 0=Mon, 6=Sun
        day_name = ts.strftime("%A")
        is_weekend = 1 if day_of_week >= 5 else 0
        is_holiday = 1 if f"{month:02d}-{day:02d}" in HOLIDAYS else 0

        # Source and destination (no self-routes)
        src = random.choice(LOCATION_NAMES)
        dst = random.choice([l for l in LOCATION_NAMES if l != src])
        distance_km = get_distance(src, dst)

        route_type = random.choice(ROUTE_TYPES)
        signal_count = random.randint(2, 18) if route_type != "Highway" else random.randint(0, 3)

        road_capacity = random.choice([2, 4, 6, 8])  # Lanes

        # Weather
        weather = get_weather_for_month(month)
        temperature = get_temperature(month, hour)
        humidity = round(random.uniform(40, 95), 1)

        rainfall_mm = 0.0
        if weather in ["Light Rain", "Heavy Rain", "Thunderstorm"]:
            if weather == "Light Rain":
                rainfall_mm = round(random.uniform(0.5, 5), 1)
            elif weather == "Heavy Rain":
                rainfall_mm = round(random.uniform(5, 30), 1)
            else:
                rainfall_mm = round(random.uniform(10, 50), 1)

        visibility = round(random.uniform(8, 15), 1) if weather == "Clear" else \
                     round(random.uniform(5, 10), 1) if weather in ["Cloudy", "Light Rain"] else \
                     round(random.uniform(1, 5), 1)

        # Events
        event_flag = 1 if random.random() < 0.05 else 0
        event_type = random.choice(EVENT_TYPES[1:]) if event_flag else "None"

        # Accidents & construction
        accident_flag = 1 if random.random() < 0.03 else 0
        construction_flag = 1 if random.random() < 0.08 else 0

        # Traffic index
        traffic_index = compute_traffic_index(
            hour, day_of_week, is_weekend, is_holiday, weather,
            src, dst, event_flag, accident_flag, construction_flag, month
        )
        traffic_index = round(traffic_index, 1)

        # Speeds
        avg_speed, max_speed, min_speed = compute_speeds(traffic_index, distance_km)

        # Vehicle density
        vehicle_density = int(traffic_index * random.uniform(15, 30))

        # Travel and waiting time
        if avg_speed > 0:
            travel_time = (distance_km / avg_speed) * 60  # minutes
        else:
            travel_time = distance_km * 10
        travel_time += signal_count * random.uniform(0.5, 2)
        travel_time = round(travel_time, 1)

        waiting_time = round(signal_count * random.uniform(0.3, 1.5) +
                            (traffic_index / 100) * random.uniform(2, 10), 1)

        # Flags
        peak_hour_flag = 1 if (8 <= hour <= 11 or 17 <= hour <= 20) else 0
        office_hour_flag = 1 if (9 <= hour <= 18 and not is_weekend) else 0

        # Fuel waste estimate (liters)
        fuel_waste = round((waiting_time / 60) * random.uniform(0.8, 1.5), 2)

        # Congestion level
        congestion_level = compute_congestion_level(traffic_index)

        records.append({
            "date": str(date),
            "time": time_str,
            "year": year,
            "month": month,
            "day": day,
            "day_of_week": day_of_week,
            "day_name": day_name,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "source_area": src,
            "destination_area": dst,
            "distance_km": distance_km,
            "route_type": route_type,
            "signal_count": signal_count,
            "road_capacity": road_capacity,
            "vehicle_density": vehicle_density,
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "min_speed": min_speed,
            "traffic_index": traffic_index,
            "weather": weather,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall_mm": rainfall_mm,
            "visibility": visibility,
            "event_flag": event_flag,
            "event_type": event_type,
            "accident_flag": accident_flag,
            "construction_flag": construction_flag,
            "fuel_waste_estimate": fuel_waste,
            "travel_time_minutes": travel_time,
            "waiting_time_minutes": waiting_time,
            "peak_hour_flag": peak_hour_flag,
            "office_hour_flag": office_hour_flag,
            "congestion_level": congestion_level,
        })

    df = pd.DataFrame(records)

    # Save raw data
    raw_path = os.path.join(DATA_DIR, "raw_traffic_data.csv")
    df.to_csv(raw_path, index=False)
    print(f"\n  [OK] Dataset saved: {raw_path}")
    print(f"  [OK] Total rows: {len(df):,}")
    print(f"  [OK] Total columns: {len(df.columns)}")
    print(f"\n  Congestion Distribution:")
    print(df["congestion_level"].value_counts().to_string())
    print(f"\n  Date Range: {df['date'].min()} -> {df['date'].max()}")
    print(f"\n  Sample locations: {df['source_area'].nunique()} sources, {df['destination_area'].nunique()} destinations")
    print("=" * 60)

    return df


if __name__ == "__main__":
    generate_dataset()
