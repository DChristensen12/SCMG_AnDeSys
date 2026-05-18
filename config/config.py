import yaml
import os
import torch
from dotenv import load_dotenv

# Load sensitive environment variables from .env
load_dotenv()


class Config:
    # ─── SCMG REST API ──────────────────────────────────────────────────────
    API_TOKEN     = os.getenv("SCMG_API_TOKEN")
    API_BASE_URL  = os.getenv(
        "SCMG_API_BASE_URL",
        "https://www.strawberrycreek.org/api/creek-data/",
    )

    # ─── NWS Weather (LBNL station) ─────────────────────────────────────────
    # No API key needed; only a descriptive User-Agent.
    # Set USE_NWS_RAIN=false in .env to disable rain-merge entirely.
    NWS_STATION_ID = os.getenv("NWS_STATION_ID", "LBNL1")
    NWS_USER_AGENT = os.getenv("NWS_USER_AGENT", "SCMG-AnDeSys/1.0")
    USE_NWS_RAIN   = os.getenv("USE_NWS_RAIN", "true").lower() == "true"
    USE_NWS_WEATHER = os.getenv("USE_NWS_WEATHER", "true").lower() == "true"

    # ─── MySQL ───────────────────────────────────────
    # Only needed when running with --data-source sql.
    # Schema: one table per site (named after site_code, lowercased), columns
    # are managed dynamically by Night Heron's get_creek_data.py. The sql_client
    # uses SHOW COLUMNS to discover what's available — no column overrides needed.
    #
    # Variable names match Night Heron's .env so the same credentials work for both.
    MYSQL_HOST     = os.getenv("MYSQL_HOST")
    MYSQL_USER     = os.getenv("MYSQL_DATABASE_USER")
    MYSQL_PASSWORD = os.getenv("MYSQL_DATABASE_PASSWORD")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE_NAME")
    MYSQL_PORT     = int(os.getenv("MYSQL_PORT", "3306"))

    # ─── YAML settings ──────────────────────────────────────────────────────
    _current_dir = os.path.dirname(__file__)
    _yaml_path   = os.path.join(_current_dir, "settings.yaml")

    try:
        with open(_yaml_path, "r") as f:
            _s = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: settings.yaml not found at {_yaml_path}. Using hardcoded defaults.")
        _s = {}

    # Model architecture
    HIDDEN_DIM      = _s.get("model_architecture", {}).get("hidden_dim", 16)
    GNN_LAYERS      = _s.get("model_architecture", {}).get("gnn_layers", 1)
    TEMPORAL_LAYERS = _s.get("model_architecture", {}).get("temporal_layers", 1)
    DROPOUT         = _s.get("model_architecture", {}).get("dropout", 0.1)
    GNN_TYPE        = _s.get("model_architecture", {}).get("gnn_type", "GCN")

    # Training
    EPOCHS        = _s.get("training", {}).get("epochs", 10)
    BATCH_SIZE    = _s.get("training", {}).get("batch_size", 128)
    LEARNING_RATE = _s.get("training", {}).get("learning_rate", 0.001)
    PATIENCE      = _s.get("training", {}).get("patience", 5)
    TRAIN_SPLIT   = _s.get("training", {}).get("train_split", 0.8)

    # Detection
    THRESHOLD_PERCENTILE      = _s.get("detection", {}).get("threshold_percentile", 99)
    RAIN_WINDOW_HOURS         = _s.get("detection", {}).get("rain_window_hours", 12)
    RAIN_THRESHOLD_MULTIPLIER = _s.get("detection", {}).get("rain_multiplier", 2.0)
    RAIN_AMOUNT_THRESHOLD     = _s.get("detection", {}).get("rain_amount_threshold", 0.1)

    # Preprocessing
    IMPUTATION_LIMIT_HOURS = _s.get("preprocessing", {}).get("imputation_limit_hours", 3)

    # ─── Constants ──────────────────────────────────────────────────────────
    SEQUENCE_LENGTH = 24
    DATA_FILE       = "full_creek_gnn.csv"
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOCATIONS       = ["footbridge", "north_fork_0", "south_fork_2", "south_fork_1", "oxford"]
    LOCATION_TO_IDX = {loc: idx for idx, loc in enumerate(LOCATIONS)}

    # NUM_FEATURES is not used anywhere — the model reads sequences.shape[3]
    # at runtime, which is the right thing. Leaving the line out intentionally;
    # if anything tries to import it, that's a stale reference to clean up.