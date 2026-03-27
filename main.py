import torch
import numpy as np
import os
import sys
from config.config import Config
from src.ingest.data_loader import load_and_preprocess_data
from src.utils.graph_utils import create_graph_topology
from src.preprocessing.data_processor import prepare_sequences_normalized
from src.models.model import TemporalGNN
from src.training.trainer import train_temporal_gnn
from src.anomalies.anomaly_detector import compute_anomaly_scores, detect_spills_with_rain_adjustment
from src.utils.visualizations import plot_static_dashboard

def main(mode="update"):
    """
    Control logic for the GNN pipeline.
    """
    model_dir = "models"
    model_path = os.path.join(model_dir, "gnn_weights.pt")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("--- SCMG Anomaly Detection System ---")
    print(f"Execution Mode: {mode.upper()}")
    print(f"Device: {Config.DEVICE}")
    
    # DATA LOADING LOGIC 
    if mode == "inference":
        # Pull only 2 days for speed during live monitoring
        df_featured, df_original, locations = load_and_preprocess_data(force_download=True, days=2)
    else:
        # Pull 30 days for training/updating
        df_featured, df_original, locations = load_and_preprocess_data(force_download=False, days=30)
    # ----------------------------------
    
    edge_index, _, location_to_idx = create_graph_topology()
    
    sequences, targets, timestamps, scaler = prepare_sequences_normalized(
        df_featured, 
        location_to_idx, 
        Config.SEQUENCE_LENGTH
    )
    
    if mode == "inference":
        train_seq, train_tgt = None, None
        test_seq, test_tgt = sequences, targets
        test_timestamps = timestamps
    else:
        split_idx = int(len(sequences) * Config.TRAIN_SPLIT)
        train_seq, test_seq = sequences[:split_idx], sequences[split_idx:]
        train_tgt, test_tgt = targets[:split_idx], targets[split_idx:]
        test_timestamps = timestamps[split_idx:]
    
    num_node_features = sequences.shape[3] 
    model = TemporalGNN(num_node_features=num_node_features).to(Config.DEVICE)
    
    if mode in ["update", "inference"]:
        if os.path.exists(model_path):
            print(f"Loading weights from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        else:
            print(f"No weight file found at {model_path}. Switching to fresh train.")
            mode = "train"

    if mode in ["train", "update"]:
        print("Commencing model optimization...")
        train_temporal_gnn(
            model, 
            train_seq, 
            train_tgt, 
            edge_index, 
            val_sequences=test_seq, 
            val_targets=test_tgt
        )
        torch.save(model.state_dict(), model_path)
        print(f"Optimization complete. Weights saved to {model_path}")
    else:
        print("Skipping training phase. Entering evaluation mode.")

    model.eval()
    errors, predictions = compute_anomaly_scores(
        model, 
        test_seq, 
        test_tgt, 
        edge_index, 
        Config.DEVICE
    )
    
    system_scores = np.mean(errors, axis=(1, 2))
    
    spill_flags, rain_flags, adjusted_thresholds = detect_spills_with_rain_adjustment(
        system_anomaly_scores=system_scores,
        timestamps=test_timestamps,
        df_original=df_original,
        locations=locations
    )
    
    base_threshold = np.percentile(system_scores, Config.THRESHOLD_PERCENTILE)
    spill_count = np.sum(spill_flags)
    print(f"Detection cycle finished. Anomalies identified: {spill_count}")
    
    plot_static_dashboard(
        timestamps=test_timestamps,
        system_anomaly_scores=system_scores,
        normalized_anomaly_scores=errors,
        adjusted_thresholds=adjusted_thresholds,
        base_threshold=base_threshold,
        spill_flags=spill_flags,
        rain_flags=rain_flags,
        df_original=df_original,
        locations=locations,
        threshold_percentile=Config.THRESHOLD_PERCENTILE
    )

    if mode == "inference" and spill_count > 0:
        try:
            from src.utils.notifier import send_spill_alert
            # Identify which locations had flags
            # Find any timestamp where a spill occurred, then find which sensors were above threshold
            affected_indices = np.unique(np.where(spill_flags)[0])
            # For simplicity in alerts, we notify based on nodes that peaked
            loc_peaks = np.any(spill_flags, axis=0)
            affected_locations = [locations[i] for i, peak in enumerate(loc_peaks) if peak]
            
            send_spill_alert(int(spill_count), affected_locations)
        except Exception as e:
            print(f"Alerting failed: {e}")

    if mode != "inference":
        from src.utils.visualizations import plot_interactive_plotly
        plot_interactive_plotly(
            timestamps=test_timestamps,
            system_anomaly_scores=system_scores,
            adjusted_thresholds=adjusted_thresholds,
            base_threshold=base_threshold,
            spill_flags=spill_flags,
            rain_flags=rain_flags,
            rain_threshold_multiplier=Config.RAIN_THRESHOLD_MULTIPLIER,
            rain_window_hours=Config.RAIN_WINDOW_HOURS,
            threshold_percentile=Config.THRESHOLD_PERCENTILE
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SCMG GNN Pipeline")
    parser.add_argument('--mode', type=str, default='update', choices=['train', 'update', 'inference'])
    args = parser.parse_args()
    main(mode=args.mode)