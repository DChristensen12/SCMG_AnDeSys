import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from config.config import Config
from src.ingest.api_client import fetch_network_snapshot

def load_and_preprocess_data(file_path=Config.DATA_FILE, force_download=False, days=30, data_source="api"):
    """
    Loads creek data. If the local file is missing or force_download is True,
    it fetches data from the configured source for the specified number of days.

    data_source: "api" (default) pulls from the Strawberry Creek REST API.
                 "sql" pulls from the SQL database configured in .env.
    """

    if not os.path.exists(file_path) or force_download:
        source_label = "API" if data_source == "api" else "SQL database"
        if force_download:
            print(f"Refresh requested. Fetching last {days} days of data from {source_label}...")
        else:
            print(f"Local file '{file_path}' not found. Initializing {source_label} pull...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        if data_source == "sql":
            from src.ingest.sql_client import fetch_network_snapshot_sql
            df_raw = fetch_network_snapshot_sql(
                start_time=start_date.isoformat(),
                end_time=end_date.isoformat(),
            )
        else:
            df_raw = fetch_network_snapshot(
                start_time=start_date.isoformat(),
                end_time=end_date.isoformat(),
            )

        if df_raw.empty:
            print(f"CRITICAL ERROR: No data retrieved from {source_label}.")
            if not os.path.exists(file_path):
                sys.exit(1)
            print("Falling back to existing local file.")
        else:
            # Map raw source column names to the internal names used by the GNN.
            # For the SQL path the sensor columns are already renamed by sql_client;
            # only 'timestamp' -> 'datetime' and 'station_id' -> 'location' remain.
            column_mapping = {
                'Meter_Hydros21_Cond': 'conductivity',
                'Meter_Hydros21_Depth': 'depth',
                'Meter_Hydros21_Temp': 'temperature',
                'TE_TR_525USW_Precip_5minTotal': 'rain_mm',
                'Sensirion_SHT40_Temperature': 'Temp2m_Avg',
                'timestamp': 'datetime',
                'station_id': 'location',
            }

            df_raw = df_raw.rename(columns=column_mapping)

            if Config.USE_NWS_RAIN:
                df_raw = _merge_nws_rain(df_raw, start_date, end_date)

            df_raw.to_csv(file_path, index=False)
            print(f"Data successfully cached to {file_path}")

    # Load from the cached or existing file
    df = pd.read_csv(file_path)
    
    # Preprocessing and Indexing
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()

    print(f"--- Data Loading Report ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Range: {df.index.min()} to {df.index.max()}")

    # Pass every numeric sensor column through — no hardcoded list.
    # New parameters from the API or SQL are picked up automatically.
    # Only 'rain_mm' and 'depth' carry semantic meaning downstream
    # (rain-flag logic and absent-sensor detection); everything else is
    # just a model feature and its name doesn't matter.
    _non_feature = {'Unnamed: 0', 'delta', 'Precip_Max', 'EnviroDIY_Mayfly_Batt'}
    feature_cols = [
        col for col in df.columns
        if col != 'location'
        and col not in _non_feature
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        print("CRITICAL ERROR: No numeric feature columns found in dataset.")
        sys.exit(1)

    df_featured = df[['location'] + feature_cols].copy()

    print(f"Active features ({len(feature_cols)}): {', '.join(feature_cols)}")
    if 'rain_mm' not in feature_cols:
        print("Warning: 'rain_mm' not found — rain-adjusted detection will be disabled.")
    print(f"---------------------------\n")

    return df_featured, df, Config.LOCATIONS


def _merge_nws_rain(df_raw, start_date, end_date):
    """
    Fetch NWS hourly precipitation for the same window and merge it into df_raw.

    Strategy: take max(creek_rain_mm, nws_rain_mm) at each timestamp so that
    if the creek's tipping-bucket gauge fails or reads zero during a real storm,
    the NWS reading fills the gap.  NWS values (hourly) are aligned by flooring
    each creek timestamp to the nearest hour.
    """
    from src.ingest.weather_client import fetch_nws_precipitation

    print(f"Fetching NWS precipitation for station {Config.NWS_STATION_ID}...")
    df_nws = fetch_nws_precipitation(start_date.isoformat(), end_date.isoformat())

    if df_nws.empty:
        print("NWS rain data unavailable — using creek sensor rain only.")
        return df_raw

    # Align: floor each creek timestamp to the hour, then look up the NWS
    # hourly observation for that hour (precipitationLastHour covers the
    # preceding 60 minutes, so the 14:00 reading = rain from 13:00–14:00).
    creek_dt = pd.to_datetime(df_raw["datetime"], utc=True)
    hourly_keys = creek_dt.dt.floor("h")

    nws_lookup = df_nws["rain_mm"].fillna(0)
    nws_per_row = hourly_keys.map(nws_lookup).fillna(0).values

    creek_rain = pd.to_numeric(df_raw.get("rain_mm", 0), errors="coerce").fillna(0).values
    df_raw["rain_mm"] = pd.Series(
        [max(c, n) for c, n in zip(creek_rain, nws_per_row)],
        index=df_raw.index,
    )

    nws_contributed = int((nws_per_row > creek_rain).sum())
    print(f"NWS rain merged — supplemented {nws_contributed} rows where creek gauge read lower.")
    return df_raw