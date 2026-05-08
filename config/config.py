import yaml
import os
import torch
from dotenv import load_dotenv

# Load in the sensitive environment variables from .env 
load_dotenv()

class Config:
    # Grab SCMG API and Sensitive Credentials (from .env)
    API_TOKEN = os.getenv("SCMG_API_TOKEN")
    API_BASE_URL = os.getenv("SCMG_API_BASE_URL", "https://www.strawberrycreek.org/api/creek-data/")

    # NWS Weather (Lawrence Berkeley National Lab Weather Data), this has no API key required, just a User-Agent)
    # Used to supplement the creek's own rain gauge with an independent measurement.
    # Set USE_NWS_RAIN=false in .env to disable.
    NWS_STATION_ID = os.getenv("NWS_STATION_ID", "LBNL1")
    NWS_USER_AGENT = os.getenv("NWS_USER_AGENT", "SCMG-AnDeSys/1.0")
    USE_NWS_RAIN   = os.getenv("USE_NWS_RAIN", "true").lower() == "true"

    # SQL Database (this is optional, it is only needed when using --data-source sql)
    SQL_CONNECTION_STRING = os.getenv("SQL_CONNECTION_STRING")
    SQL_TABLE_NAME        = os.getenv("SQL_TABLE_NAME", "creek_data")
    # Column names in your SQL table, you should override these if your schema differs
    SQL_TIMESTAMP_COL     = os.getenv("SQL_TIMESTAMP_COL", "timestamp")
    SQL_LOCATION_COL      = os.getenv("SQL_LOCATION_COL", "station_id")
    SQL_CONDUCTIVITY_COL  = os.getenv("SQL_CONDUCTIVITY_COL", "Meter_Hydros21_Cond")
    SQL_DEPTH_COL         = os.getenv("SQL_DEPTH_COL", "Meter_Hydros21_Depth")
    SQL_TEMP_COL          = os.getenv("SQL_TEMP_COL", "Meter_Hydros21_Temp")
    SQL_RAIN_COL          = os.getenv("SQL_RAIN_COL", "TE_TR_525USW_Precip_5minTotal")
    SQL_TEMP2M_COL        = os.getenv("SQL_TEMP2M_COL", "Sensirion_SHT40_Temperature")

    # Path Setup for YAML
    _current_dir = os.path.dirname(__file__)
    _yaml_path = os.path.join(_current_dir, 'settings.yaml')

    # Load YAML
    try:
        with open(_yaml_path, 'r') as f:
            _s = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: settings.yaml not found at {_yaml_path}. Using hardcoded defaults.")
        _s = {}

    # Map YAML to Class Variables
    HIDDEN_DIM      = _s.get('model_architecture', {}).get('hidden_dim', 16)
    GNN_LAYERS      = _s.get('model_architecture', {}).get('gnn_layers', 1)
    TEMPORAL_LAYERS = _s.get('model_architecture', {}).get('temporal_layers', 1)
    DROPOUT         = _s.get('model_architecture', {}).get('dropout', 0.1)
    GNN_TYPE        = _s.get('model_architecture', {}).get('gnn_type', 'GCN')

    EPOCHS          = _s.get('training', {}).get('epochs', 10)
    BATCH_SIZE      = _s.get('training', {}).get('batch_size', 128)
    LEARNING_RATE   = _s.get('training', {}).get('learning_rate', 0.001)
    PATIENCE        = _s.get('training', {}).get('patience', 5)
    TRAIN_SPLIT     = _s.get('training', {}).get('train_split', 0.8)

    THRESHOLD_PERCENTILE      = _s.get('detection', {}).get('threshold_percentile', 99)
    RAIN_WINDOW_HOURS         = _s.get('detection', {}).get('rain_window_hours', 12)
    RAIN_THRESHOLD_MULTIPLIER = _s.get('detection', {}).get('rain_multiplier', 2.0)
    RAIN_AMOUNT_THRESHOLD     = _s.get('detection', {}).get('rain_amount_threshold', 0.1)

    IMPUTATION_LIMIT_HOURS = _s.get('preprocessing', {}).get('imputation_limit_hours', 3)

    # Constants
    SEQUENCE_LENGTH = 24
    DATA_FILE = 'full_creek_gnn.csv'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LOCATIONS = ['footbridge', 'north_fork_0', 'south_fork_2', 'south_fork_1', 'oxford']
    LOCATION_TO_IDX = {loc: idx for idx, loc in enumerate(LOCATIONS)}
    NUM_FEATURES = 10 # Adjust this based on the actual column count