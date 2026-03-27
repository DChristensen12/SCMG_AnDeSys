import requests
import pandas as pd
import sys
from config.config import Config

def fetch_creek_data(site, start_time, end_time, variables=None):
    """
    Query the Strawberry Creek API for a specific site and time range.
    Returns a pandas DataFrame.
    """
    
    headers = {
        "Authorization": f"Token {Config.API_TOKEN}"
    }
    
    # Base parameters required by the GET endpoint
    params = {
        "site": site,
        "start": start_time,
        "end": end_time
    }
    
    # Handle optional variable filtering
    if variables:
        params["vars"] = variables

    try:
        response = requests.get(
            Config.API_BASE_URL, 
            headers=headers, 
            params=params,
            timeout=60
        )
        
        # Check for 401 Unauthorized or 404 Not Found
        if response.status_code != 200:
            print(f"Error: API returned status {response.status_code}")
            print(f"Details: {response.text}")
            return pd.DataFrame()
            
        data = response.json()
        
        if not data:
            print(f"No records found for {site} in the specified range.")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # Standardize timestamp for downstream processing
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df

    except requests.exceptions.Timeout:
        print("The request timed out. The server might be busy.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

def fetch_network_snapshot(start_time, end_time):
    """
    Iterates through all sites defined in Config to pull a full dataset.
    Useful for building the graph-based training set.
    """
    frames = []
    
    for site in Config.LOCATIONS:
        print(f"Requesting data: {site}...")
        df_site = fetch_creek_data(site, start_time, end_time)
        
        if not df_site.empty:
            df_site['station_id'] = site
            frames.append(df_site)
            
    if not frames:
        print("No data retrieved for any site.")
        return pd.DataFrame()
        
    # Combine all sites into a single flat file for the DataLoader
    return pd.concat(frames, ignore_index=True)