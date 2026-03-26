import torch
import numpy as np
from config.config import Config
from src.ingest.data_loader import load_and_preprocess_data
from src.utils.graph_utils import create_graph_topology
from src.preprocessing.data_processor import prepare_sequences_normalized
from src.models.model import TemporalGNN
from src.training.trainer import train_temporal_gnn
from src.anomalies.anomaly_detector import compute_anomaly_scores, detect_spills_with_rain_adjustment
from src.utils.visualizations import plot_static_dashboard

def main():
    # Initialize Environment
    print(f"   Starting SCMG Anomaly Detection System")
    print(f"   Target Device: {Config.DEVICE}")
    
    # Load Raw Data
    # df_featured: only the columns for the model
    # df_original: the full dataframe (needed for rain timestamps later)
    df_featured, df_original, locations = load_and_preprocess_data()
    
    # Define Graph Topology
    edge_index, _, location_to_idx = create_graph_topology()
    
    # Preprocess into Normalized Sequences
    sequences, targets, timestamps, scaler = prepare_sequences_normalized(
        df_featured, 
        location_to_idx, 
        Config.SEQUENCE_LENGTH
    )
    
    # Split Data (80% Train, 20% Test/Validation)
    split_idx = int(len(sequences) * Config.TRAIN_SPLIT)
    train_seq, test_seq = sequences[:split_idx], sequences[split_idx:]
    train_tgt, test_tgt = targets[:split_idx], targets[split_idx:]
    test_timestamps = timestamps[split_idx:]
    
    # Initialize Model
    # num_node_features is the last dimension of our sequences
    num_node_features = sequences.shape[3] 
    model = TemporalGNN(num_node_features=num_node_features)
    
    # Train the Model (Learning the 'Clean' Physics)
    train_losses, val_losses = train_temporal_gnn(
        model, 
        train_seq, 
        train_tgt, 
        edge_index, 
        val_sequences=test_seq, 
        val_targets=test_tgt
    )
    
    # Run Inference & Compute Anomaly Scores
    # We run this on the test set to find spills
    errors, predictions = compute_anomaly_scores(
        model, 
        test_seq, 
        test_tgt, 
        edge_index, 
        Config.DEVICE
    )
    
    # Calculate "System Score" (Mean error across all sensors and features)
    system_scores = np.mean(errors, axis=(1, 2))
    
    # Rain-Aware Spill Detection
    spill_flags, rain_flags, adjusted_thresholds = detect_spills_with_rain_adjustment(
        system_anomaly_scores=system_scores,
        timestamps=test_timestamps,
        df_original=df_original,
        locations=locations
    )
    
    # Visualization
    base_threshold = np.percentile(system_scores, Config.THRESHOLD_PERCENTILE)
    
    plot_static_dashboard(
        timestamps=test_timestamps,
        system_anomaly_scores=system_scores,
        normalized_anomaly_scores=errors, # Error per sensor used for bar chart
        adjusted_thresholds=adjusted_thresholds,
        base_threshold=base_threshold,
        spill_flags=spill_flags,
        rain_flags=rain_flags,
        df_original=df_original,
        locations=locations,
        threshold_percentile=Config.THRESHOLD_PERCENTILE
    )

if __name__ == "__main__":
    main()