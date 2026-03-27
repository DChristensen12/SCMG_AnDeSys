import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from config.config import Config
from src.ingest.api_client import fetch_network_snapshot

def load_and_preprocess_data(file_path=Config.DATA_FILE, force_download=False, days=30):
    """
    Loads creek data. If the local file is missing or force_download is True, 
    it fetches data from the Strawberry Creek API for the specified number of days.
    """
    
    if not os.path.exists(file_path) or force_download:
        if force_download:
            print(f"Refresh requested. Fetching last {days} days of data...")
        else:
            print(f"Local file '{file_path}' not found. Initializing API pull...")
        
        # Calculate time window based on the provided 'days' argument
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df_api = fetch_network_snapshot(
            start_time=start_date.isoformat(),
            end_time=end_date.isoformat()
        )
        
        if df_api.empty:
            print("CRITICAL ERROR: No data retrieved from API.")
            if not os.path.exists(file_path):
                sys.exit(1)
            print("Falling back to existing local file.")
        else:
            # Map API keys to internal names used by the GNN
            column_mapping = {
                'Meter_Hydros21_Cond': 'conductivity',
                'Meter_Hydros21_Depth': 'depth',
                'Meter_Hydros21_Temp': 'temperature',
                'TE_TR_525USW_Precip_5minTotal': 'rain_mm',
                'Sensirion_SHT40_Temperature': 'Temp2m_Avg',
                'timestamp': 'datetime',
                'station_id': 'location'
            }
            
            df_api = df_api.rename(columns=column_mapping)
            
            # Save the latest pull to disk
            df_api.to_csv(file_path, index=False)
            print(f"Data successfully cached to {file_path}")

    # Load from the cached or existing file
    df = pd.read_csv(file_path)
    
    # Preprocessing and Indexing
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()

    print(f"--- Data Loading Report ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Range: {df.index.min()} to {df.index.max()}")
    
    physics_features = ['conductivity', 'depth', 'temperature']
    weather_features = ['rain_mm']
    metadata = ['location']
    
    all_features = physics_features + weather_features + metadata
    
    # Check for required columns
    missing_cols = [col for col in all_features if col not in df.columns]
    if missing_cols:
        print(f"Warning: Columns missing from dataset: {missing_cols}")
        all_features = [c for c in all_features if c in df.columns]

    df_featured = df[all_features].copy()

    print(f"Active features: {', '.join([f for f in all_features if f != 'location'])}")
    print(f"---------------------------\n")

    return df_featured, df, Config.LOCATIONS