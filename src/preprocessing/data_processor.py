import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config.config import Config

def prepare_sequences_normalized(df_featured, location_to_idx, sequence_length=Config.SEQUENCE_LENGTH):
    """
    Prepare temporal sequences with Z-score normalization.
    Handles Oxford sensor's missing depth data by filling with 0 (neutral value post-scaling).
    """
    
    # Identify feature columns
    exclude_cols = ['location', 'Unnamed: 0', 'delta', 'Precip_Max', 'EnviroDIY_Mayfly_Batt']
    # Select only numeric columns and filter out excluded ones
    feature_cols = [col for col in df_featured.select_dtypes(include=[np.number]).columns.tolist() 
                    if col not in exclude_cols]

    print(f"[INFO] Using {len(feature_cols)} features: {', '.join(feature_cols)}")

    # Get specific indices for the Oxford depth handling
    oxford_idx = location_to_idx['oxford']
    try:
        depth_feature_idx = feature_cols.index('depth')
    except ValueError:
        raise ValueError("Column 'depth' must be present in features for Oxford sensor handling.")

    # Z-score normalization
    # Fit scaler on all available data combined to ensure consistent scaling across nodes
    all_data = []
    for location in location_to_idx.keys():
        loc_data = df_featured[df_featured['location'] == location][feature_cols].values
        all_data.append(loc_data)

    all_data = np.vstack(all_data)
    scaler = StandardScaler()
    scaler.fit(all_data)

    # Apply normalization to a copy of the dataframe
    df_normalized = df_featured.copy()
    for location in location_to_idx.keys():
        loc_mask = df_featured['location'] == location
        df_normalized.loc[loc_mask, feature_cols] = scaler.transform(
            df_featured.loc[loc_mask, feature_cols].values
        )

    # Pivot to 3D array: (timesteps, nodes, features)
    print("--- Building 3D Array ---")
    timestamps_all = sorted(df_normalized.index.unique())
    num_nodes = len(location_to_idx)
    num_features = len(feature_cols)

    data_3d = np.full((len(timestamps_all), num_nodes, num_features), np.nan)

    for t_idx, timestamp in enumerate(tqdm(timestamps_all, desc="Pivoting data")):
        t_data = df_normalized.loc[timestamp]
        
        # Only process if we have data for all nodes at this timestamp
        if len(t_data) == num_nodes:
            for _, row in t_data.iterrows():
                node_idx = location_to_idx[row['location']]
                data_3d[t_idx, node_idx, :] = row[feature_cols].values

    # Validating timesteps (Allowing NaN only for Oxford depth)
    def is_valid_timestep(t_idx):
        t_data = data_3d[t_idx].copy()
        t_data[oxford_idx, depth_feature_idx] = 0  # Temporarily ignore Oxford depth for check
        return not np.isnan(t_data).any()

    valid_mask = np.array([is_valid_timestep(i) for i in range(len(timestamps_all))])
    print(f"[INFO] Valid timesteps: {valid_mask.sum():,} / {len(valid_mask):,}")

    # Create sliding window sequences
    print(f"--- Creating Sequences (Length: {sequence_length}) ---")
    sequences = []
    targets = []
    sequence_timestamps = []

    for i in tqdm(range(len(timestamps_all) - sequence_length), desc="Sliding window"):
        # Ensure the entire window + the target step are valid
        if valid_mask[i:i+sequence_length+1].all():
            seq = data_3d[i:i+sequence_length].copy()
            target = data_3d[i+sequence_length].copy()

            # Fill Oxford depth with 0 (Model learns to ignore this channel for this node)
            seq[:, oxford_idx, depth_feature_idx] = 0
            target[oxford_idx, depth_feature_idx] = 0

            sequences.append(seq)
            targets.append(target)
            sequence_timestamps.append(timestamps_all[i+sequence_length])

    print(f"[INFO] Final Sequence Count: {len(sequences):,}")

    return np.array(sequences), np.array(targets), sequence_timestamps, scaler