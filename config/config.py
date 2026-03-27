import yaml
import os
import torch
from dotenv import load_dotenv

# Load sensitive environment variables from .env
load_dotenv()

class Config:
    # API and Sensitive Credentials (from .env)
    API_TOKEN = os.getenv("SCMG_API_TOKEN")
    API_BASE_URL = os.getenv("SCMG_API_BASE_URL", "https://www.strawberrycreek.org/api/creek-data/")

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

    # Constants
    SEQUENCE_LENGTH = 24
    DATA_FILE = 'full_creek_gnn.csv'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LOCATIONS = ['footbridge', 'north_fork_0', 'south_fork_2', 'south_fork_1', 'oxford']
    LOCATION_TO_IDX = {loc: idx for idx, loc in enumerate(LOCATIONS)}
    NUM_FEATURES = 10 # Adjust this based on the actual column count