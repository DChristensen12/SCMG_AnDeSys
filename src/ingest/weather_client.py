import requests
import pandas as pd
from config.config import Config


def fetch_nws_precipitation(start_time, end_time):
    """
    Fetch hourly precipitation observations from the NWS for the configured station.

    Uses the free NWS observations API (no API key required — only a User-Agent).
    Returns a DataFrame with a UTC DatetimeIndex and a single 'rain_mm' column.
    Returns an empty DataFrame on any failure.
    """
    base_url = f"https://api.weather.gov/stations/{Config.NWS_STATION_ID}/observations"
    headers = {
        "User-Agent": Config.NWS_USER_AGENT,
        "Accept": "application/geo+json",
    }
    params = {
        "start": _ensure_utc_suffix(start_time),
        "end": _ensure_utc_suffix(end_time),
    }

    all_features = []
    next_url = base_url

    while next_url:
        try:
            resp = requests.get(
                next_url,
                headers=headers,
                params=(params if next_url == base_url else None),
                timeout=30,
            )
            if resp.status_code != 200:
                print(f"NWS API error {resp.status_code} for station {Config.NWS_STATION_ID}: {resp.text[:200]}")
                break
            data = resp.json()
            all_features.extend(data.get("features", []))
            next_url = data.get("pagination", {}).get("next")
        except Exception as e:
            print(f"NWS fetch failed: {e}")
            break

    if not all_features:
        print(f"No NWS observations returned for station {Config.NWS_STATION_ID}.")
        return pd.DataFrame()

    records = []
    for feat in all_features:
        props = feat.get("properties", {})
        ts = props.get("timestamp")
        precip = props.get("precipitationLastHour") or {}
        val = precip.get("value")  # NWS reports in meters; None = not observed
        if ts is not None:
            records.append({
                "datetime": ts,
                # Convert m → mm; keep None so we can distinguish "no report" from "zero rain"
                "rain_mm": val * 1000.0 if val is not None else None,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce")
    return df


def _ensure_utc_suffix(t):
    """Make sure a timestamp string is recognisable as UTC by the NWS API."""
    s = str(t)
    if not (s.endswith("Z") or "+00" in s or "-00" in s):
        return s + "Z"
    return s
