"""
Anomaly-detection integration tests for the SCMG Temporal GNN.

Each test loads a labeled sensor window from data/anomalies/, runs it through
the trained model, and checks whether the reconstruction error is elevated
(anomalous) or flat (true negative).

--- How partial-parameter files are handled ---
The anomaly CSVs only contain the Hydros21 sensor columns (conductivity, depth,
temperature).  The trained model may have been built with additional parameters
(rain_mm, Temp2m_Avg, etc.).  Missing columns are filled with 0.0, which equals
the normalised mean — a neutral, "background" value.  The GNN can still detect
anomalies that manifest in the *available* channels; a conductivity spike stays
visible as a spike in the reconstruction error even when other channels are flat.

--- Station mapping ---
Test-file suffixes map to Config.LOCATIONS station names.  If a station is not
present in the trained model's graph (e.g. sf0, nf1 if they were never in
Config.LOCATIONS), that test is skipped with a clear message.

--- Running ---
    pytest tests/test_anomaly_detection.py -v

Requires a trained model:
    python main.py --mode train
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.config import Config

ANOMALY_DIR = ROOT / "data" / "anomalies"

# Maps the short suffix used in file names to Config.LOCATIONS station names.
STATION_MAP = {
    "nf0": "north_fork_0",
    "nf1": "north_fork_1",
    "sf0": "south_fork_0",
    "sf1": "south_fork_1",
    "sf2": "south_fork_2",
}

# Raw API column names → internal names (mirrors data_loader.py column_mapping)
COLUMN_MAP = {
    "DateTimeUTC":                   "datetime",
    "Meter_Hydros21_Cond":           "conductivity",
    "Meter_Hydros21_Depth":          "depth",
    "Meter_Hydros21_Temp":           "temperature",
    "TE_TR_525USW_Precip_5minTotal": "rain_mm",
    "Sensirion_SHT40_Temperature":   "Temp2m_Avg",
}

# ── Ground-truth catalog ──────────────────────────────────────────────────────
# (filename, station_suffix_or_None, is_anomalous, event_group)
#
# is_anomalous=True  → model should produce elevated reconstruction error
# is_anomalous=False → true negative; model should NOT flag it
EVENT_CATALOG = [
    # June 2025 mystery spill — propagating downstream across all south-fork sensors
    ("anomaly_2025_06_12_spill_sf0.csv",        "sf0", True,  "jun25_spill"),
    ("anomaly_2025_06_12_spill_sf1.csv",        "sf1", True,  "jun25_spill"),
    ("anomaly_2025_06_12_spill_sf2.csv",        "sf2", True,  "jun25_spill"),
    # September 2025 overnight conductivity spike across south fork
    ("anomaly_2025_09_10_overnight_sf0.csv",    "sf0", True,  "sep25_overnight"),
    ("anomaly_2025_09_10_overnight_sf1.csv",    "sf1", True,  "sep25_overnight"),
    ("anomaly_2025_09_10_overnight_sf2.csv",    "sf2", True,  "sep25_overnight"),
    # November 2025 foam/chemical event — nf1 anomalous, nf0 contrast/comparison
    ("anomaly_2025_11_05_foam_nf1.csv",         "nf1", True,  "nov25_foam"),
    ("anomaly_2025_11_05_foam_nf0.csv",         "nf0", False, "nov25_foam"),
    # November 2025 storm — nf1 anomalous spike, nf0 is the normal rain response
    ("anomaly_2025_11_13_rain_nf1.csv",         "nf1", True,  "nov25_rain"),
    ("anomaly_2025_11_13_rain_nf0.csv",         "nf0", False, "nov25_rain"),
    # April 2026 rainfall — all four sensors, all true negatives
    ("anomaly_2026_04_01_rainfall_nf0.csv",     "nf0", False, "apr26_rainfall"),
    ("anomaly_2026_04_01_rainfall_nf1.csv",     "nf1", False, "apr26_rainfall"),
    ("anomaly_2026_04_01_rainfall_sf0.csv",     "sf0", False, "apr26_rainfall"),
    ("anomaly_2026_04_01_rainfall_sf2.csv",     "sf2", False, "apr26_rainfall"),
    # Standalone: true-negative rain event (model should not over-flag)
    ("anomaly_2025_05_12_rain_nf1.csv",         "nf1", False, "may25_rain_tn"),
    # Standalone: actuator malfunction at botanical garden
    ("anomaly_2026_01_botanical_actuator.csv",  None,  True,  "jan26_actuator"),
    # Baseline for the same sensor/location — must NOT be flagged
    ("normal_2026_01_botanical_baseline.csv",   None,  False, "jan26_actuator_baseline"),
    # Standalone: fire-hydrant spill at north fork 0
    ("anomaly_2026_03_20_hydrant_nf0.csv",      "nf0", True,  "mar26_hydrant"),
]

# Threshold multiplier: peak error must exceed this multiple of the window
# baseline (first 20 % of timesteps) to be called anomalous.
ANOMALY_MULTIPLIER = 2.0


# ── Core helper ───────────────────────────────────────────────────────────────

def _reconstruction_errors(csv_path, station_suffix, model, model_metadata, edge_index):
    """
    Load a single-sensor CSV window and return per-timestep reconstruction
    errors (mean absolute error, averaged across all feature channels) for
    the target node.

    Columns absent from the CSV are filled with 0.0 (= normalised mean), so
    the model still receives a full-rank feature vector.  The anomaly signal
    in the channels that ARE present remains intact.
    """
    feature_cols     = model_metadata["feature_cols"]
    scaler           = model_metadata["scaler"]
    location_to_idx  = model_metadata["location_to_idx"]
    num_nodes        = len(location_to_idx)
    num_features     = len(feature_cols)

    # Resolve station
    station_name = STATION_MAP.get(station_suffix) if station_suffix else None
    if station_name is not None and station_name not in location_to_idx:
        pytest.skip(
            f"Station '{station_name}' (suffix '{station_suffix}') is not in "
            f"the trained model's graph.  Add it to Config.LOCATIONS and retrain."
        )
    # For files without a station suffix (botanical actuator), use node 0 as a
    # proxy — we're checking whether *any* anomalous signal elevates the error.
    node_idx = location_to_idx.get(station_name, 0)

    # Load CSV
    df = pd.read_csv(csv_path)
    df = df.rename(columns=COLUMN_MAP)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()

    # Build a full feature matrix: real values where available, 0 elsewhere
    feature_matrix = pd.DataFrame(0.0, index=df.index, columns=feature_cols)
    for col in feature_cols:
        if col in df.columns:
            feature_matrix[col] = df[col]

    # Normalise using training scaler
    normalised = scaler.transform(feature_matrix.fillna(0.0).values)

    seq_len = Config.SEQUENCE_LENGTH
    if len(normalised) <= seq_len:
        pytest.skip(
            f"{csv_path.name} has only {len(normalised)} rows — need > {seq_len} "
            f"for a single sequence.  File too short to test."
        )

    errors = []
    with torch.no_grad():
        for i in range(len(normalised) - seq_len):
            # Target node gets real normalised data; all other nodes stay at 0
            seq    = np.zeros((seq_len,  num_nodes, num_features))
            target = np.zeros((num_nodes, num_features))

            seq[:,    node_idx, :] = normalised[i : i + seq_len]
            target[   node_idx, :] = normalised[i + seq_len]

            seq_t = torch.FloatTensor(seq).unsqueeze(0).to(Config.DEVICE)
            pred  = model(seq_t, edge_index, batch_size=1, num_nodes=num_nodes)

            err = torch.abs(
                pred[0, node_idx] -
                torch.FloatTensor(target[node_idx]).to(Config.DEVICE)
            )
            errors.append(err.mean().item())   # scalar: mean over features

    return np.array(errors)   # shape (T,)


def _is_elevated(errors: np.ndarray) -> bool:
    """
    True if the peak error in the window is more than ANOMALY_MULTIPLIER times
    the baseline (the first 20 % of timesteps, which should be pre-event).
    """
    split      = max(1, len(errors) // 5)
    baseline   = errors[:split].mean()
    peak       = errors.max()
    if baseline == 0:
        return peak > 0
    return peak / baseline > ANOMALY_MULTIPLIER


# ── Parametrised tests ────────────────────────────────────────────────────────

ANOMALOUS_CASES    = [(f, s, g) for f, s, a, g in EVENT_CATALOG if a]
TRUE_NEGATIVE_CASES = [(f, s, g) for f, s, a, g in EVENT_CATALOG if not a]


@pytest.mark.parametrize("filename,station,group", ANOMALOUS_CASES,
                         ids=[g + "/" + f.replace(".csv", "")
                              for f, s, a, g in EVENT_CATALOG if a])
def test_anomaly_detected(filename, station, group, trained_model, model_metadata, edge_index):
    """Reconstruction error must spike for known anomalous events."""
    errors = _reconstruction_errors(
        ANOMALY_DIR / filename, station, trained_model, model_metadata, edge_index
    )
    assert _is_elevated(errors), (
        f"[{group}] {filename}: expected anomalous spike but peak error "
        f"({errors.max():.4f}) did not exceed {ANOMALY_MULTIPLIER}× baseline "
        f"({errors[:max(1,len(errors)//5)].mean():.4f})."
    )


@pytest.mark.parametrize("filename,station,group", TRUE_NEGATIVE_CASES,
                         ids=[g + "/" + f.replace(".csv", "")
                              for f, s, a, g in EVENT_CATALOG if not a])
def test_true_negative_not_flagged(filename, station, group, trained_model, model_metadata, edge_index):
    """Reconstruction error must remain flat for normal / rain events."""
    errors = _reconstruction_errors(
        ANOMALY_DIR / filename, station, trained_model, model_metadata, edge_index
    )
    assert not _is_elevated(errors), (
        f"[{group}] {filename}: expected no anomaly but peak error "
        f"({errors.max():.4f}) exceeded {ANOMALY_MULTIPLIER}× baseline "
        f"({errors[:max(1,len(errors)//5)].mean():.4f}).  "
        f"Model may be over-flagging rain events."
    )


# ── Within-group relative tests ───────────────────────────────────────────────
# These don't require a calibrated threshold — they just check ordering.

def test_foam_event_nf1_more_anomalous_than_nf0(trained_model, model_metadata, edge_index):
    """
    November 2025 foam event: nf1 (anomalous) should have a higher peak error
    than nf0 (contrast/comparison case).
    """
    err_nf1 = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2025_11_05_foam_nf1.csv",
        "nf1", trained_model, model_metadata, edge_index
    )
    err_nf0 = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2025_11_05_foam_nf0.csv",
        "nf0", trained_model, model_metadata, edge_index
    )
    assert err_nf1.max() > err_nf0.max(), (
        f"Foam event: nf1 peak ({err_nf1.max():.4f}) should exceed "
        f"nf0 peak ({err_nf0.max():.4f})."
    )


def test_rain_storm_nf1_more_anomalous_than_nf0(trained_model, model_metadata, edge_index):
    """
    November 2025 storm: nf1 (anomalous conductivity spike during rain) should
    have a higher peak error than nf0 (normal rain response / true negative).
    """
    err_nf1 = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2025_11_13_rain_nf1.csv",
        "nf1", trained_model, model_metadata, edge_index
    )
    err_nf0 = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2025_11_13_rain_nf0.csv",
        "nf0", trained_model, model_metadata, edge_index
    )
    assert err_nf1.max() > err_nf0.max(), (
        f"Nov storm: nf1 peak ({err_nf1.max():.4f}) should exceed "
        f"nf0 peak ({err_nf0.max():.4f})."
    )


def test_spill_propagation_all_south_fork_flagged(trained_model, model_metadata, edge_index):
    """
    June 2025 spill: all three south-fork sensors should independently show
    elevated reconstruction error as the spill propagates downstream.
    """
    results = {}
    for suffix, fname in [("sf0", "anomaly_2025_06_12_spill_sf0.csv"),
                          ("sf1", "anomaly_2025_06_12_spill_sf1.csv"),
                          ("sf2", "anomaly_2025_06_12_spill_sf2.csv")]:
        errors = _reconstruction_errors(
            ANOMALY_DIR / fname, suffix, trained_model, model_metadata, edge_index
        )
        results[suffix] = errors

    failures = [s for s, e in results.items() if not _is_elevated(e)]
    assert not failures, (
        f"Jun 2025 spill: expected all three sf sensors to be flagged, "
        f"but {failures} were not elevated."
    )


def test_botanical_actuator_more_anomalous_than_baseline(trained_model, model_metadata, edge_index):
    """
    The actuator malfunction window should have a meaningfully higher peak
    reconstruction error than the same sensor's normal baseline window,
    recorded at the same time of year (January 2026).
    """
    err_actuator = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2026_01_botanical_actuator.csv",
        None, trained_model, model_metadata, edge_index
    )
    err_baseline = _reconstruction_errors(
        ANOMALY_DIR / "normal_2026_01_botanical_baseline.csv",
        None, trained_model, model_metadata, edge_index
    )
    assert err_actuator.max() > err_baseline.max(), (
        f"Botanical actuator: anomaly peak ({err_actuator.max():.4f}) should "
        f"exceed baseline peak ({err_baseline.max():.4f})."
    )


def test_april_rainfall_no_false_positives(trained_model, model_metadata, edge_index):
    """
    April 2026 rain event: none of the four sensors should be flagged.
    Validates the model doesn't over-fire on expected rainfall behaviour.
    """
    files = [
        ("nf0", "anomaly_2026_04_01_rainfall_nf0.csv"),
        ("nf1", "anomaly_2026_04_01_rainfall_nf1.csv"),
        ("sf0", "anomaly_2026_04_01_rainfall_sf0.csv"),
        ("sf2", "anomaly_2026_04_01_rainfall_sf2.csv"),
    ]
    false_positives = []
    for suffix, fname in files:
        errors = _reconstruction_errors(
            ANOMALY_DIR / fname, suffix, trained_model, model_metadata, edge_index
        )
        if _is_elevated(errors):
            false_positives.append(suffix)

    assert not false_positives, (
        f"Apr 2026 rainfall: sensors {false_positives} were flagged as anomalous "
        f"but should be true negatives."
    )
