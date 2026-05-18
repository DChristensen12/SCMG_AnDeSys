import numpy as np
import pandas as pd
import torch
from config.config import Config

def compute_anomaly_scores(model, sequences, targets, edge_index, device):
    """
    Compute reconstruction errors (anomaly scores) for sequences.
    
    Returns:
        errors: np.ndarray (absolute prediction error per sensor per feature)
        predictions: np.ndarray (raw model outputs)
    """
    model.eval()
    model = model.to(device)
    edge_index = edge_index.to(device)

    with torch.no_grad():
        # Convert to tensor and move to device
        seq_tensor = torch.FloatTensor(sequences).to(device)
        target_tensor = torch.FloatTensor(targets).to(device)

        predictions = model(
            seq_tensor,
            edge_index,
            batch_size=len(sequences),
            num_nodes=sequences.shape[2]
        )

        # Compute absolute reconstruction error (MAE)
        errors = torch.abs(predictions - target_tensor)

    return errors.cpu().numpy(), predictions.cpu().numpy()

def detect_spills_with_rain_adjustment(
    system_anomaly_scores,
    timestamps,
    df_original,
    locations,
    threshold_percentile=Config.THRESHOLD_PERCENTILE,
    rain_window_hours=Config.RAIN_WINDOW_HOURS,
    rain_threshold_multiplier=Config.RAIN_THRESHOLD_MULTIPLIER,
    rain_amount_threshold=Config.RAIN_AMOUNT_THRESHOLD
):
    """
    Rain-aware spill detection with adaptive thresholding.
    Identifies rain-affected periods with a lookback window, calculates a base
    threshold from the anomaly score distribution, then scales it up during rain
    to reduce false positives from natural runoff.
    """
    # Skip rain adjustment if there's no rain_mm column or it's all NaN.
    if 'rain_mm' not in df_original.columns or df_original['rain_mm'].isna().all():
        print("[INFO] No rain_mm data available — running without rain adjustment.")
        base_threshold = np.percentile(system_anomaly_scores, threshold_percentile)
        adjusted_thresholds = np.full(len(timestamps), base_threshold)
        spill_flags = system_anomaly_scores > adjusted_thresholds
        rain_flags = np.zeros(len(timestamps), dtype=bool)
        print(f"--- Detection Summary ---")
        print(f"Total Spills Detected: {spill_flags.sum()}")
        print(f"Rain-Affected Spills: 0 (no rain data)")
        print(f"Dry-Weather Spills: {spill_flags.sum()}")
        print(f"-------------------------\n")
        return spill_flags, rain_flags, adjusted_thresholds

    # Use the first location as the weather reference for rain data
    rain_data = df_original[df_original['location'] == locations[0]][['rain_mm']].copy()

    # Flag each timestamp where it rained in the past rain_window_hours
    rain_flags = np.zeros(len(timestamps), dtype=bool)

    for i, ts in enumerate(timestamps):
        lookback_start = ts - pd.Timedelta(hours=rain_window_hours)
        recent_rain = rain_data[(rain_data.index >= lookback_start) & (rain_data.index <= ts)]
        
        if recent_rain['rain_mm'].sum() > rain_amount_threshold:
            rain_flags[i] = True

    # Calculate Adaptive Thresholds
    base_threshold = np.percentile(system_anomaly_scores, threshold_percentile)
    
    # Apply the multiplier only where rain_flags is True
    adjusted_thresholds = np.where(
        rain_flags,
        base_threshold * rain_threshold_multiplier,
        base_threshold
    )

    # Generate Spill Flags
    spill_flags = system_anomaly_scores > adjusted_thresholds
    
    # Summary Logging
    print(f"--- Detection Summary ---")
    print(f"Total Spills Detected: {spill_flags.sum()}")
    print(f"Rain-Affected Spills: {(spill_flags & rain_flags).sum()}")
    print(f"Dry-Weather Spills: {(spill_flags & ~rain_flags).sum()}")
    print(f"-------------------------\n")

    return spill_flags, rain_flags, adjusted_thresholds