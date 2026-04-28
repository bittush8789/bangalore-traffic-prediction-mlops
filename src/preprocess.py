"""
Data Preprocessing Pipeline for Bangalore Traffic AI System.
Cleans, validates, encodes, and splits data for model training.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATA_DIR, ensure_dirs


def load_raw_data():
    """Load the raw generated dataset."""
    path = os.path.join(DATA_DIR, "raw_traffic_data.csv")
    df = pd.read_csv(path)
    print(f"  [OK] Loaded raw data: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def clean_data(df):
    """Clean and validate the dataset."""
    print("  [>>] Cleaning data...")
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"      Removed {before - len(df)} duplicates")

    # Handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Clip outliers
    df["avg_speed"] = df["avg_speed"].clip(2, 80)
    df["traffic_index"] = df["traffic_index"].clip(0, 100)
    df["travel_time_minutes"] = df["travel_time_minutes"].clip(1, 300)
    df["waiting_time_minutes"] = df["waiting_time_minutes"].clip(0, 60)
    df["temperature"] = df["temperature"].clip(10, 45)
    df["humidity"] = df["humidity"].clip(20, 100)
    df["rainfall_mm"] = df["rainfall_mm"].clip(0, 100)

    print(f"  [OK] Data cleaned: {df.shape[0]:,} rows")
    return df


def encode_categoricals(df):
    """Encode categorical variables."""
    print("  [>>] Encoding categorical features...")

    label_encoders = {}

    cat_cols = ["source_area", "destination_area", "route_type", "weather",
                "event_type", "day_name"]

    for col in cat_cols:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Congestion level encoding
    congestion_map = {"Low": 0, "Medium": 1, "High": 2, "Severe": 3}
    df["congestion_level_encoded"] = df["congestion_level"].map(congestion_map)

    print(f"  [OK] Encoded {len(cat_cols)} categorical columns")
    return df, label_encoders


def split_data(df, test_size=0.2):
    """Split into train and test sets."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42,
                                          stratify=df["congestion_level"])

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"  [OK] Train set: {len(train_df):,} rows -> {train_path}")
    print(f"  [OK] Test set:  {len(test_df):,} rows -> {test_path}")
    return train_df, test_df


def preprocess_pipeline():
    """Run the complete preprocessing pipeline."""
    print("=" * 60)
    print("  DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    ensure_dirs()
    df = load_raw_data()
    df = clean_data(df)

    # Save cleaned data
    cleaned_path = os.path.join(DATA_DIR, "cleaned_traffic_data.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"  [OK] Cleaned data saved: {cleaned_path}")

    df, encoders = encode_categoricals(df)
    train_df, test_df = split_data(df)

    print("=" * 60)
    return df, train_df, test_df, encoders


if __name__ == "__main__":
    preprocess_pipeline()
