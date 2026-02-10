"""
Fetch historical game-day temperatures for NBA arenas using Open-Meteo API.
Free, no API key required. Uses arena coordinates from arena_coordinates.py.
"""

import os
import time
import pandas as pd
import requests as req
from arena_coordinates import ARENA_COORDS, TEAM_NAME_TO_ABBREV

# Open-Meteo historical weather API
HIST_URL = "https://archive-api.open-meteo.com/v1/archive"
# Open-Meteo forecast API (for upcoming games)
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_historical_temp(lat, lon, date_str):
    """
    Fetch daily mean temperature (Celsius) for a location and date.
    Returns temperature or None on failure.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "daily": "temperature_2m_mean",
        "temperature_unit": "celsius",
        "timezone": "America/New_York",  # NBA games mostly in US timezones
    }
    try:
        resp = req.get(HIST_URL, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            temps = data.get("daily", {}).get("temperature_2m_mean", [])
            if temps and temps[0] is not None:
                return round(temps[0], 1)
    except Exception as e:
        print(f"  Weather error for ({lat},{lon}) on {date_str}: {e}")
    return None


def fetch_forecast_temp(lat, lon, date_str):
    """
    Fetch forecast daily mean temperature (Celsius) for upcoming games.
    Open-Meteo provides up to 7 days ahead.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "daily": "temperature_2m_mean",
        "temperature_unit": "celsius",
        "timezone": "America/New_York",
    }
    try:
        resp = req.get(FORECAST_URL, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            temps = data.get("daily", {}).get("temperature_2m_mean", [])
            if temps and temps[0] is not None:
                return round(temps[0], 1)
    except Exception as e:
        print(f"  Forecast error for ({lat},{lon}) on {date_str}: {e}")
    return None


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    print("Loading game data...")
    games = pd.read_csv(os.path.join(data_dir, "game_pairs.csv"))
    games["GAME_DATE_home"] = pd.to_datetime(games["GAME_DATE_home"])

    # Build unique (home_team_abbrev, date) pairs to minimise API calls
    from arena_coordinates import TEAM_ID_TO_ABBREV
    games["home_abbrev"] = games["TEAM_ID_home"].map(TEAM_ID_TO_ABBREV).fillna(games["TEAM_ABBREVIATION_home"])
    games["date_str"] = games["GAME_DATE_home"].dt.strftime("%Y-%m-%d")

    location_dates = games[["home_abbrev", "date_str"]].drop_duplicates()
    print(f"Unique (arena, date) pairs: {len(location_dates)}")

    # Check for existing progress
    progress_path = os.path.join(data_dir, "weather_cache.csv")
    if os.path.exists(progress_path):
        cache = pd.read_csv(progress_path)
        cached_keys = set(zip(cache["team"], cache["date"]))
        print(f"Resuming: {len(cached_keys)} entries cached")
    else:
        cache = pd.DataFrame(columns=["team", "date", "temp_c"])
        cached_keys = set()

    records = cache.to_dict("records")
    fetched = 0

    for _, row in location_dates.iterrows():
        team = row["home_abbrev"]
        date = row["date_str"]

        if (team, date) in cached_keys:
            continue

        coords = ARENA_COORDS.get(team)
        if not coords:
            print(f"  No coords for {team}, skipping")
            continue

        temp = fetch_historical_temp(coords[0], coords[1], date)
        records.append({"team": team, "date": date, "temp_c": temp})
        cached_keys.add((team, date))
        fetched += 1

        if fetched % 100 == 0:
            print(f"  Fetched {fetched} temps... (latest: {team} {date} = {temp}°C)")
            # Save checkpoint
            pd.DataFrame(records).to_csv(progress_path, index=False)

        # Rate limit: Open-Meteo allows ~600 req/min, be gentle
        time.sleep(0.15)

    # Final save
    df = pd.DataFrame(records)
    df.to_csv(progress_path, index=False)

    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(df)} weather records")
    print(f"  Nulls: {df['temp_c'].isna().sum()}")
    print(f"  Saved: {progress_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
