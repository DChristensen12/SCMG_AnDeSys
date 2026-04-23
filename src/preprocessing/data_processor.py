import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config.config import Config


def _impute_short_gaps(df, feature_cols, limit_hours):
    """
    For each (location, feature) pair, linearly interpolate over gaps shorter
    than limit_hours.  Longer gaps are left as NaN so the validity check
    downstream still excludes them.  limit_area='inside' prevents extrapolation
    beyond the first/last real observation.
    """
    timestamps = df.index.sort_values().unique()
    if len(timestamps) < 2:
        return df

    median_interval = pd.Series(timestamps.astype('int64')).diff().dropna().median()
    median_interval = pd.Timedelta(median_interval)
    rows_per_hour = pd.Timedelta('1h') / median_interval
    limit_rows = max(1, int(limit_hours * rows_per_hour))

    print(f"--- Imputing Short Sensor Gaps (limit: {limit_hours}h / {limit_rows} rows) ---")
    df = df.copy()
    total_filled = 0

    for location in df['location'].unique():
        mask = df['location'] == location
        before = int(df.loc[mask, feature_cols].isna().sum().sum())

        df.loc[mask, feature_cols] = (
            df.loc[mask, feature_cols]
            .interpolate(method='time', limit=limit_rows, limit_area='inside')
        )

        filled = before - int(df.loc[mask, feature_cols].isna().sum().sum())
        if filled > 0:
            print(f"  [{location}] filled {filled} missing values")
        total_filled += filled

    print(f"[INFO] Imputation complete — {total_filled} values filled across all locations.\n")
    return df


def prepare_sequences_normalized(df_featured, location_to_idx, sequence_length=Config.SEQUENCE_LENGTH):
    """
    Prepare temporal sequences with Z-score normalization.
    Short sensor gaps are interpolated before normalization.
    Oxford's permanently absent depth channel is zeroed so the model learns
    to ignore it rather than treat it as signal.
    """

    # Identify feature columns
    exclude_cols = ['location', 'Unnamed: 0', 'delta', 'Precip_Max', 'EnviroDIY_Mayfly_Batt']
    feature_cols = [col for col in df_featured.select_dtypes(include=[np.number]).columns.tolist()
                    if col not in exclude_cols]

    print(f"[INFO] Using {len(feature_cols)} features: {', '.join(feature_cols)}")

    # Fill short sensor-dropout gaps before anything else.
    # Gaps longer than IMPUTATION_LIMIT_HOURS remain NaN and are caught by the
    # validity check below, so they never enter a training sequence.
    df_featured = _impute_short_gaps(df_featured, feature_cols, Config.IMPUTATION_LIMIT_HOURS)

    # Detect permanently absent (node, feature) pairs — channels where a sensor
    # simply doesn't exist at a location (e.g. Oxford has no depth sensor).
    # These are zeroed in every sequence so the model learns a fixed "no signal"
    # value rather than seeing NaN, and they're exempted from the validity check
    # so their timesteps aren't discarded.  This generalises the original
    # Oxford-depth special case to work for any future absent sensor automatically.
    print("--- Detecting Permanently Absent Sensor Channels ---")
    absent_pairs = set()  # {(node_idx, feat_idx), ...}
    for location, node_idx in location_to_idx.items():
        loc_mask = df_featured['location'] == location
        for feat_idx, feat in enumerate(feature_cols):
            if df_featured.loc[loc_mask, feat].isna().all():
                absent_pairs.add((node_idx, feat_idx))
                print(f"  {location}/{feat}: no data — will be zeroed in sequences")
    if not absent_pairs:
        print("  (none)")
    print()

    # Z-score normalization.
    # Fit only on rows that are fully valid so NaN values from long outages
    # don't corrupt the scaler's mean/std statistics.
    all_data = []
    for location in location_to_idx.keys():
        loc_data = df_featured[df_featured['location'] == location][feature_cols].values
        all_data.append(loc_data)

    all_data = np.vstack(all_data)
    valid_rows = ~np.isnan(all_data).any(axis=1)
    scaler = StandardScaler()
    scaler.fit(all_data[valid_rows])

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

    # Validating timesteps — exempt permanently absent channels from the NaN check
    def is_valid_timestep(t_idx):
        t_data = data_3d[t_idx].copy()
        for node_idx, feat_idx in absent_pairs:
            t_data[node_idx, feat_idx] = 0
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

            # Zero out permanently absent channels so the model sees a fixed
            # "no signal" value rather than NaN
            for node_idx, feat_idx in absent_pairs:
                seq[:, node_idx, feat_idx] = 0
                target[node_idx, feat_idx] = 0

            sequences.append(seq)
            targets.append(target)
            sequence_timestamps.append(timestamps_all[i+sequence_length])

    print(f"[INFO] Final Sequence Count: {len(sequences):,}")

    return np.array(sequences), np.array(targets), sequence_timestamps, scaler, feature_cols