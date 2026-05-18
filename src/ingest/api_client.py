"""
api_client.py
─────────────────────────────────────────────────────────────
Public Strawberry Creek REST API data source for Pulse.

Used when --data-source=api (the default).

Important API quirks worked around in this client:

1) The API rejects requests missing the 'vars' parameter, despite the docs
   saying it's optional.

2) The API has a server-side bug where multiple 'vars' parameters are not
   accumulated — only the LAST one in the URL is honored. To get multiple
   sensor columns we have to make one request per column and merge
   client-side on the timestamp.

3) The timestamp column is always included implicitly; we don't need to
   ask for it. It comes back as 'DateTimeUTC'.
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import reduce
from typing import List, Optional

import pandas as pd
import requests

from config.config import Config

logger = logging.getLogger(__name__)


# Sensor columns we pull for each site by default. Each is requested in its
# own HTTP call because of API quirk #2 above. The minimum useful set for
# Pulse downstream is cond/depth/temp; batt is included for health monitoring.
_DEFAULT_VARS = [
    "Meter_Hydros21_Cond",
    "Meter_Hydros21_Depth",
    "Meter_Hydros21_Temp",
    "EnviroDIY_Mayfly_Batt",
]


def _fetch_single_var(
    site: str,
    start_str: str,
    end_str: str,
    var_name: str,
    headers: dict,
) -> pd.DataFrame:
    """
    Pull one (site, sensor) pair from the API. Returns a DataFrame with
    'timestamp' and one sensor column, or empty on any failure.

    Splitting one request per sensor is required by API quirk #2.
    """
    params = [
        ("site", site),
        ("start", start_str),
        ("end", end_str),
        ("vars", var_name),
    ]
    try:
        response = requests.get(
            Config.API_BASE_URL, headers=headers, params=params, timeout=60
        )
    except requests.exceptions.Timeout:
        logger.error(f"[{site}/{var_name}] API request timed out after 60s")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        logger.error(f"[{site}/{var_name}] API request failed: {e}")
        return pd.DataFrame()

    if response.status_code != 200:
        logger.error(
            f"[{site}/{var_name}] API returned {response.status_code}: "
            f"{response.text[:200]}"
        )
        return pd.DataFrame()

    data = response.json()
    if not data:
        # Site doesn't expose this column, or no observations in window.
        # Either way: not an error, just nothing to merge.
        logger.debug(f"[{site}/{var_name}] no rows")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "DateTimeUTC" in df.columns:
        df = df.rename(columns={"DateTimeUTC": "timestamp"})
    if "timestamp" not in df.columns:
        logger.error(f"[{site}/{var_name}] no timestamp in response")
        return pd.DataFrame()

    return df


def fetch_creek_data(
    site: str,
    start_time,
    end_time,
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Query the Strawberry Creek API for one site over [start_time, end_time].

    Internally makes one HTTP request per sensor column (API quirk #2) and
    merges them on timestamp. Returns a DataFrame with 'timestamp',
    'station_id', and one column per available sensor at this site.

    Sensors that the site doesn't expose are simply absent from the
    resulting DataFrame — there's no error, the column just won't be there.
    """
    headers = {}
    if Config.API_TOKEN:
        headers["Authorization"] = f"Token {Config.API_TOKEN}"

    start_str = (
        start_time.strftime("%Y-%m-%dT%H:%M:%S")
        if isinstance(start_time, datetime) else str(start_time)
    )
    end_str = (
        end_time.strftime("%Y-%m-%dT%H:%M:%S")
        if isinstance(end_time, datetime) else str(end_time)
    )

    vars_to_request = variables if variables else _DEFAULT_VARS

    # Pull each sensor in its own request, collect non-empty frames.
    frames = []
    for var_name in vars_to_request:
        df = _fetch_single_var(site, start_str, end_str, var_name, headers)
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.info(f"[{site}] no data for any requested variable in window")
        return pd.DataFrame()

    # Merge all sensor frames on the timestamp column.
    merged = reduce(
        lambda left, right: pd.merge(left, right, on="timestamp", how="outer"),
        frames,
    )

    # Normalize timestamp and sort.
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
    merged = (
        merged.dropna(subset=["timestamp"])
              .sort_values("timestamp")
              .reset_index(drop=True)
    )
    merged["station_id"] = site

    logger.info(
        f"[{site}] fetched {len(merged):,} rows × {len(merged.columns) - 2} sensors"
    )
    return merged


def fetch_network_snapshot(start_time, end_time) -> pd.DataFrame:
    """
    Pull data for every site in Config.LOCATIONS and concatenate.
    Same shape as sql_client.fetch_network_snapshot_sql.
    """
    frames = []
    for site in Config.LOCATIONS:
        print(f"Requesting data: {site}...")
        df_site = fetch_creek_data(site, start_time, end_time)
        if not df_site.empty:
            frames.append(df_site)

    if not frames:
        print("No data retrieved for any site.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
