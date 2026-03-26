import pandas as pd
import os
import sys
from config.config import Config

def load_and_preprocess_data(file_path=Config.DATA_FILE):
    """
    Loads aggregated multi-sensor creek data with weather features.
    """
    # Robust File Check
    if not os.path.exists(file_path):
        print(f"CRITICAL ERROR: Data file '{file_path}' not found.")
        print("Make sure the CSV is in the root directory of the project.")
        sys.exit(1)

    # Loading and Initial Setup
    df = pd.read_csv(file_path)
    
    # Ensure datetime is handled correctly
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()

    # Metadata Logging
    print(f"--- Data Loading Report ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Range: {df.index.min()} to {df.index.max()}")
    
    # Feature Selection Logic
    physics_features = ['conductivity', 'depth', 'temperature']
    weather_features = ['rain_mm', 'SolRad_Avg', 'Temp2m_Avg']
    metadata = ['location']
    
    all_features = physics_features + weather_features + metadata
    
    # A check to see if all required columns exist in the CSV
    missing_cols = [col for col in all_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    df_featured = df[all_features].copy()

    print(f"Physics features: {', '.join(physics_features)}")
    print(f"Weather context: {', '.join(weather_features)}")
    print(f"---------------------------\n")

    return df_featured, df, Config.LOCATIONS