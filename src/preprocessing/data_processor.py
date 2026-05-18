import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config.config import Config
from src.ingest.data_loader import _NON_FEATURE_COLUMNS


def _impute_short_gaps(df, feature_cols, limit_hours):
    """
    For each (location, feature) pair, linearly interpolate over gaps shorter
    than limit_hours. Longer gaps are left as NaN so the validity check
    downstream still excludes them. limit_area='inside' prevents extrapolation
    beyond the first/last real observation.
    """
    timestamps = pd.DatetimeIndex(sorted(df.index.unique()))
    if len(timestamps) < 2:
        return df

    # Use pandas-native Timedelta arithmetic. Avoids int64-nanosecond round-trip
    # bugs that previously made median_interval ~1 second instead of 15 minutes.
    median_interval = pd.Series(timestamps).diff().median()
    if pd.isna(median_interval) or median_interval <= pd.Timedelta(0):
        print("[WARN] Could not determine sampling cadence; skipping imputation.")
        return df

    rows_per_hour = pd.Timedelta("1h") / median_interval
    limit_rows = max(1, int(limit_hours * rows_per_hour))

    # Safety: pandas' interpolation breaks if limit_rows >= the per-site array length.
    # Cap to the smallest per-site row count minus a safety margin.
    min_site_rows = df.groupby("location").size().min()
    max_safe_limit = max(1, min_site_rows - 2)
    if limit_rows > max_safe_limit:
        print(
            f"[INFO] limit_rows={limit_rows} exceeds smallest site size "
            f"({min_site_rows} rows); capping to {max_safe_limit}."
        )
        limit_rows = max_safe_limit

    print(f"--- Imputing Short Sensor Gaps (limit: {limit_hours}h / {limit_rows} rows) ---")
    df = df.copy()
    total_filled = 0

    for location in df["location"].unique():
        mask = df["location"] == location
        before = int(df.loc[mask, feature_cols].isna().sum().sum())

        # Skip if this site has too few rows for interpolation to be meaningful.
        site_row_count = mask.sum()
        if site_row_count <= 2:
            print(f"  [{location}] only {site_row_count} rows — skipping interpolation")
            continue

        df.loc[mask, feature_cols] = (
            df.loc[mask, feature_cols]
            .interpolate(method="time", limit=limit_rows, limit_area="inside")
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
    Permanently absent (location, feature) channels (e.g. Oxford's depth)
    are zeroed in every sequence so the model learns to ignore them rather
    than treat NaN as signal.
    """

    # Identify feature columns. Pull the exclude set from data_loader so the
    # two stay in sync — when someone adds a new non-feature column, they
    # only update _NON_FEATURE_COLUMNS in one place.
    exclude_cols = _NON_FEATURE_COLUMNS | {"location"}
    feature_cols = [
        col for col in df_featured.select_dtypes(include=[np.number]).columns.tolist()
        if col not in exclude_cols
    ]

    print(f"[INFO] Using {len(feature_cols)} features: {', '.join(feature_cols)}")

    # Fill short sensor-dropout gaps before anything else.
    df_featured = _impute_short_gaps(df_featured, feature_cols, Config.IMPUTATION_LIMIT_HOURS)

    # ─── rest of the function is unchanged ──────────────────────────────────
    print("--- Detecting Permanently Absent Sensor Channels ---")
    absent_pairs = set()
    for location, node_idx in location_to_idx.items():
        loc_mask = df_featured["location"] == location
        for feat_idx, feat in enumerate(feature_cols):
            if df_featured.loc[loc_mask, feat].isna().all():
                absent_pairs.add((node_idx, feat_idx))
                print(f"  {location}/{feat}: no data — will be zeroed in sequences")
    if not absent_pairs:
        print("  (none)")
    print()

    all_data = []
    for location in location_to_idx.keys():
        loc_data = df_featured[df_featured["location"] == location][feature_cols].values
        all_data.append(loc_data)

    all_data = np.vstack(all_data)
    valid_rows = ~np.isnan(all_data).any(axis=1)
    scaler = StandardScaler()
    scaler.fit(all_data[valid_rows])

    df_normalized = df_featured.copy()
    for location in location_to_idx.keys():
        loc_mask = df_featured["location"] == location
        if not loc_mask.any():
            # Location has zero rows in this dataset (sensor offline this window).
            # It's still a node in the graph; its sequences will be filled with
            # NaN/zero downstream via the absent-pairs mechanism.
            print(f"  [{location}] no rows in this window — skipping normalization")
            continue
        df_normalized.loc[loc_mask, feature_cols] = scaler.transform(
            df_featured.loc[loc_mask, feature_cols].values
        )

# ─── rest of the function ──────────────────────────────────────────────
    print("--- Detecting Permanently Absent Sensor Channels ---")
    absent_pairs = set()
    for location, node_idx in location_to_idx.items():
        loc_mask = df_featured["location"] == location
        for feat_idx, feat in enumerate(feature_cols):
            if df_featured.loc[loc_mask, feat].isna().all():
                absent_pairs.add((node_idx, feat_idx))
                print(f"  {location}/{feat}: no data — will be zeroed in sequences")
    if not absent_pairs:
        print("  (none)")
    print()

    all_data = []
    for location in location_to_idx.keys():
        loc_data = df_featured[df_featured["location"] == location][feature_cols].values
        all_data.append(loc_data)

    all_data = np.vstack(all_data)
    valid_rows = ~np.isnan(all_data).any(axis=1)
    scaler = StandardScaler()
    scaler.fit(all_data[valid_rows])

    df_normalized = df_featured.copy()
    for location in location_to_idx.keys():
        loc_mask = df_featured["location"] == location
        if not loc_mask.any():
            # Location has zero rows in this dataset (sensor offline this window).
            # It's still a node in the graph; its sequences will be filled with
            # NaN/zero downstream via the absent-pairs mechanism.
            print(f"  [{location}] no rows in this window — skipping normalization")
            continue
        df_normalized.loc[loc_mask, feature_cols] = scaler.transform(
            df_featured.loc[loc_mask, feature_cols].values
        )

    print("--- Building 3D Array ---")
    timestamps_all = sorted(df_normalized.index.unique())
    num_nodes = len(location_to_idx)
    num_features = len(feature_cols)

    data_3d = np.full((len(timestamps_all), num_nodes, num_features), np.nan)

    for t_idx, timestamp in enumerate(tqdm(timestamps_all, desc="Pivoting data")):
        t_data = df_normalized.loc[timestamp]

        # Handle single-row return: when only one site has data at this timestamp,
        # .loc returns a Series instead of a DataFrame. Normalize to a DataFrame.
        if isinstance(t_data, pd.Series):
            t_data = t_data.to_frame().T

        # Populate whichever nodes have data at this timestamp. Missing nodes
        # stay as NaN and are handled by the absent-pairs mechanism in
        # is_valid_timestep below — they're zeroed rather than treated as
        # "missing signal."
        for _, row in t_data.iterrows():
            node_idx = location_to_idx[row["location"]]
            data_3d[t_idx, node_idx, :] = row[feature_cols].values

    def is_valid_timestep(t_idx):
        t_data = data_3d[t_idx].copy()
        for node_idx, feat_idx in absent_pairs:
            t_data[node_idx, feat_idx] = 0
        return not np.isnan(t_data).any()

    valid_mask = np.array([is_valid_timestep(i) for i in range(len(timestamps_all))])
    print(f"[INFO] Valid timesteps: {valid_mask.sum():,} / {len(valid_mask):,}")

    print(f"--- Creating Sequences (Length: {sequence_length}) ---")
    sequences = []
    targets = []
    sequence_timestamps = []

    for i in tqdm(range(len(timestamps_all) - sequence_length), desc="Sliding window"):
        if valid_mask[i:i+sequence_length+1].all():
            seq = data_3d[i:i+sequence_length].copy()
            target = data_3d[i+sequence_length].copy()
            for node_idx, feat_idx in absent_pairs:
                seq[:, node_idx, feat_idx] = 0
                target[node_idx, feat_idx] = 0
            sequences.append(seq)
            targets.append(target)
            sequence_timestamps.append(timestamps_all[i+sequence_length])

    print(f"[INFO] Final Sequence Count: {len(sequences):,}")

    return np.array(sequences), np.array(targets), sequence_timestamps, scaler, feature_cols