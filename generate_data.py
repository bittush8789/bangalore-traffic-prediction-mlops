import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_bangalore_traffic_data(num_rows=50000):
    print(f"Generating {num_rows} rows of Bangalore traffic data...")
    
    np.random.seed(42)
    
    areas = [
        "Silk Board", "Whitefield", "Marathahalli", "Hebbal", "KR Puram", 
        "Electronic City", "Koramangala", "MG Road", "Indiranagar", 
        "Outer Ring Road", "Bellandur", "HSR Layout", "BTM Layout", 
        "Jayanagar", "Majestic", "Yeshwanthpur", "Manyata Tech Park"
    ]
    
    weather_options = ["Clear", "Cloudy", "Rainy", "Thunderstorm", "Fog"]
    road_types = ["Highway", "Main Road", "Inner Road", "Service Road"]
    
    # Generate dates over the last year
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(num_rows)]
    
    data = []
    for date in dates:
        area = np.random.choice(areas)
        hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        time_str = f"{hour:02d}:{minute:02d}"
        
        day_of_week = date.strftime("%A")
        is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
        
        # Simple holiday logic
        is_holiday = 1 if np.random.random() < 0.05 else 0
        
        weather = np.random.choice(weather_options, p=[0.5, 0.2, 0.15, 0.05, 0.1])
        rainfall = 0.0 if weather == "Clear" else np.random.uniform(0, 50)
        
        road_type = np.random.choice(road_types)
        route_distance = np.random.uniform(2, 25)
        
        event_nearby = 1 if np.random.random() < 0.1 else 0
        accident_reported = 1 if np.random.random() < 0.03 else 0
        
        # Traffic volume logic
        base_volume = 500
        if 8 <= hour <= 11 or 17 <= hour <= 21: # Peak hours
            base_volume *= np.random.uniform(2, 4)
        if is_weekend:
            base_volume *= 0.7
        if is_holiday:
            base_volume *= 0.5
        
        traffic_volume = int(base_volume * np.random.uniform(0.8, 1.2))
        
        # Average speed logic
        max_speed = 60 if road_type == "Highway" else 40
        speed_reduction = (traffic_volume / 2000) * 30
        if weather in ["Rainy", "Thunderstorm"]:
            speed_reduction += 15
        if accident_reported:
            speed_reduction += 25
            
        avg_speed = max(5, max_speed - speed_reduction + np.random.normal(0, 2))
        
        # Travel time calculation
        travel_time_minutes = (route_distance / avg_speed) * 60
        
        # Congestion level
        if avg_speed < 15:
            congestion_level = "Severe"
        elif avg_speed < 25:
            congestion_level = "High"
        elif avg_speed < 40:
            congestion_level = "Medium"
        else:
            congestion_level = "Low"
            
        data.append([
            area, date.strftime("%Y-%m-%d"), time_str, day_of_week, 
            is_holiday, weather, rainfall, road_type, event_nearby, 
            accident_reported, round(avg_speed, 2), traffic_volume, 
            round(route_distance, 2), congestion_level, round(travel_time_minutes, 2)
        ])
        
    columns = [
        "area_name", "date", "time", "day_of_week", "holiday", "weather", 
        "rainfall", "road_type", "event_nearby", "accident_reported", 
        "avg_speed", "traffic_volume", "route_distance", "congestion_level", 
        "travel_time_minutes"
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    os.makedirs("data", exist_ok=True)
    output_path = "data/traffic_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_bangalore_traffic_data()
