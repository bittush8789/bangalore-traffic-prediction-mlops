"""
Utility functions for Bangalore Traffic AI System.
"""
import os
import pickle
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Bangalore locations with approximate coordinates
LOCATIONS = {
    "Silk Board": (12.9170, 77.6230),
    "Whitefield": (12.9698, 77.7500),
    "Marathahalli": (12.9591, 77.6974),
    "Hebbal": (13.0358, 77.5970),
    "KR Puram": (13.0012, 77.6960),
    "Electronic City": (12.8440, 77.6602),
    "Koramangala": (12.9352, 77.6245),
    "MG Road": (12.9756, 77.6064),
    "Indiranagar": (12.9719, 77.6412),
    "Outer Ring Road": (12.9300, 77.6840),
    "Bellandur": (12.9260, 77.6762),
    "HSR Layout": (12.9116, 77.6389),
    "BTM Layout": (12.9166, 77.6101),
    "Jayanagar": (12.9308, 77.5838),
    "Airport Road": (13.0990, 77.5940),
    "Manyata Tech Park": (13.0470, 77.6210),
    "Sarjapur Road": (12.9100, 77.6870),
    "Yeshwanthpur": (13.0220, 77.5510),
    "Banashankari": (12.9255, 77.5468),
    "Majestic": (12.9770, 77.5710),
}

LOCATION_NAMES = list(LOCATIONS.keys())

# Realistic distances between common routes (km)
ROUTE_DISTANCES = {}

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two lat/lon points in km."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def get_distance(src, dst):
    """Get realistic road distance between two locations."""
    key = (src, dst)
    if key in ROUTE_DISTANCES:
        return ROUTE_DISTANCES[key]
    rev_key = (dst, src)
    if rev_key in ROUTE_DISTANCES:
        return ROUTE_DISTANCES[rev_key]
    lat1, lon1 = LOCATIONS[src]
    lat2, lon2 = LOCATIONS[dst]
    straight = haversine(lat1, lon1, lat2, lon2)
    road_factor = np.random.uniform(1.3, 1.6)
    dist = round(straight * road_factor, 1)
    ROUTE_DISTANCES[key] = dist
    return dist

# Indian holidays (approximate)
HOLIDAYS = [
    "01-26",  # Republic Day
    "03-08",  # Maha Shivaratri
    "03-25",  # Holi
    "04-14",  # Ambedkar Jayanti
    "05-01",  # May Day
    "08-15",  # Independence Day
    "08-26",  # Janmashtami
    "09-07",  # Ganesh Chaturthi
    "10-02",  # Gandhi Jayanti
    "10-12",  # Dussehra
    "10-24",  # Diwali
    "11-01",  # Kannada Rajyotsava
    "11-14",  # Children's Day
    "12-25",  # Christmas
]

IT_AREAS = ["Whitefield", "Electronic City", "Manyata Tech Park", "Outer Ring Road", "Marathahalli", "Bellandur", "Sarjapur Road"]
SHOPPING_AREAS = ["MG Road", "Koramangala", "Indiranagar", "Jayanagar", "Majestic", "Banashankari"]
SCHOOL_ZONES = ["Jayanagar", "Koramangala", "Indiranagar", "HSR Layout", "BTM Layout", "Banashankari"]

WEATHER_OPTIONS = ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Fog", "Thunderstorm"]
EVENT_TYPES = ["None", "IPL Match", "Concert", "Festival", "Political Rally", "Marathon"]
ROUTE_TYPES = ["Highway", "Main Road", "Inner Road", "Service Road"]

CONGESTION_LABELS = ["Low", "Medium", "High", "Severe"]

def save_model(model, filename):
    """Save a trained model to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved: {filepath}")

def load_model(filename):
    """Load a trained model from disk."""
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, "rb") as f:
        return pickle.load(f)

def ensure_dirs():
    """Create required directories."""
    for d in [DATA_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)
