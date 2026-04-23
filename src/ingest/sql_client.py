import pandas as pd
from sqlalchemy import create_engine, text
from config.config import Config


def _get_engine():
    if not Config.SQL_CONNECTION_STRING:
        raise ValueError(
            "SQL_CONNECTION_STRING is not set. "
            "Add it to your .env file before using --data-source sql."
        )
    return create_engine(Config.SQL_CONNECTION_STRING)


def fetch_creek_data_sql(site, start_time, end_time):
    """
    Query a SQL database for a specific site and time range.
    Returns a DataFrame with internal column names (conductivity, depth, etc.)
    plus 'timestamp' and 'station_id', mirroring the API client's output shape.
    """
    # Map the user-configured SQL column names to the internal names expected
    # by data_loader's column_mapping ('timestamp' -> 'datetime', 'station_id' -> 'location').
    sql_to_internal = {
        Config.SQL_CONDUCTIVITY_COL: "conductivity",
        Config.SQL_DEPTH_COL: "depth",
        Config.SQL_TEMP_COL: "temperature",
        Config.SQL_RAIN_COL: "rain_mm",
        Config.SQL_TEMP2M_COL: "Temp2m_Avg",
        Config.SQL_TIMESTAMP_COL: "timestamp",
        Config.SQL_LOCATION_COL: "station_id",
    }

    col_list = ", ".join(f'"{c}"' for c in sql_to_internal)

    query = text(
        f"SELECT {col_list} FROM {Config.SQL_TABLE_NAME} "  # noqa: S608
        f"WHERE {Config.SQL_LOCATION_COL} = :site "
        f"  AND {Config.SQL_TIMESTAMP_COL} >= :start_time "
        f"  AND {Config.SQL_TIMESTAMP_COL} <= :end_time "
        f"ORDER BY {Config.SQL_TIMESTAMP_COL}"
    )

    try:
        engine = _get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "site": site,
                "start_time": start_time,
                "end_time": end_time,
            })

        if df.empty:
            print(f"No records found in SQL for {site} in the specified range.")
            return df

        df = df.rename(columns=sql_to_internal)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    except Exception as e:
        print(f"SQL query failed for site '{site}': {e}")
        return pd.DataFrame()


def fetch_network_snapshot_sql(start_time, end_time):
    """
    Iterates through all sites defined in Config and pulls a full dataset from SQL.
    Returns a combined DataFrame in the same shape as fetch_network_snapshot().
    """
    frames = []

    for site in Config.LOCATIONS:
        print(f"Querying SQL database: {site}...")
        df_site = fetch_creek_data_sql(site, start_time, end_time)
        if not df_site.empty:
            frames.append(df_site)

    if not frames:
        print("No data retrieved from SQL database for any site.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
