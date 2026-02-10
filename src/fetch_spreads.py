"""
Fetch historical FanDuel spreads from The Odds API.
Pulls one snapshot per unique game date, close to tip-off time.
"""

import os
import time
import json
import pandas as pd
import requests as req
from datetime import datetime, timedelta

# Load .env file if present
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

API_KEY = os.environ.get("ODDS_API_KEY", "")
BASE_URL = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds"


def fetch_spreads_for_date(date_str, api_key):
    """
    Fetch FanDuel spreads for all NBA games on a given date.
    Uses a timestamp close to game time (noon ET = 17:00 UTC).
    """
    # Request at ~5PM UTC (noon ET) to get close-to-tipoff lines
    timestamp = f"{date_str}T17:00:00Z"

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads",
        "bookmakers": "fanduel",
        "oddsFormat": "american",
        "date": timestamp,
    }

    resp = req.get(BASE_URL, params=params)

    if resp.status_code == 422:
        # Date might be out of range, try without time
        return [], resp.headers
    elif resp.status_code != 200:
        print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
        return [], resp.headers

    data = resp.json()
    games = data.get("data", [])
    return games, resp.headers


def extract_spreads(games, game_date):
    """Extract FanDuel spread for each game."""
    records = []

    for game in games:
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        commence = game.get("commence_time", "")

        # Only include games that start on or near our target date
        if commence:
            game_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            target_dt = datetime.fromisoformat(f"{game_date}T00:00:00+00:00")
            # Games within ~36 hours of target date (to catch evening games)
            if abs((game_dt - target_dt).total_seconds()) > 36 * 3600:
                continue

        bookmakers = game.get("bookmakers", [])
        for bk in bookmakers:
            if bk.get("key") != "fanduel":
                continue
            for market in bk.get("markets", []):
                if market.get("key") != "spreads":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == away_team:
                        records.append({
                            "game_date": game_date,
                            "commence_time": commence,
                            "home_team": home_team,
                            "away_team": away_team,
                            "f_d_sprd_away": outcome.get("point"),
                        })

    return records


def main():
    api_key = API_KEY
    if not api_key:
        api_key = input("Enter Odds API key: ").strip()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Load game dates
    games = pd.read_csv(os.path.join(data_dir, "game_pairs.csv"))
    games["GAME_DATE_home"] = pd.to_datetime(games["GAME_DATE_home"])
    # FanDuel data starts from 2021-22 season
    games = games[games["GAME_DATE_home"] >= "2021-10-01"]
    unique_dates = sorted(games["GAME_DATE_home"].dt.strftime("%Y-%m-%d").unique())

    print(f"Total unique game dates: {len(unique_dates)}")

    # Check for existing progress (resume support)
    progress_path = os.path.join(data_dir, "spreads_raw.json")
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            all_records = json.load(f)
        fetched_dates = {r["game_date"] for r in all_records}
        print(f"Resuming: {len(fetched_dates)} dates already fetched")
    else:
        all_records = []
        fetched_dates = set()

    remaining_dates = [d for d in unique_dates if d not in fetched_dates]
    print(f"Dates remaining: {len(remaining_dates)}")

    for i, date_str in enumerate(remaining_dates):
        print(f"[{i+1}/{len(remaining_dates)}] Fetching {date_str}...", end=" ")

        try:
            games_data, headers = fetch_spreads_for_date(date_str, api_key)
            remaining = headers.get("x-requests-remaining", "?")

            spreads = extract_spreads(games_data, date_str)
            all_records.extend(spreads)

            print(f"{len(spreads)} spreads (credits left: {remaining})")

            # Save progress every 25 dates
            if (i + 1) % 25 == 0:
                with open(progress_path, "w") as f:
                    json.dump(all_records, f)
                print(f"  [Checkpoint saved: {len(all_records)} total records]")

            # Rate limit: ~1 req/sec
            time.sleep(1.1)

        except Exception as e:
            print(f"ERROR: {e}")
            # Save progress on error
            with open(progress_path, "w") as f:
                json.dump(all_records, f)
            print(f"  [Progress saved on error: {len(all_records)} records]")
            time.sleep(5)
            continue

    # Final save
    with open(progress_path, "w") as f:
        json.dump(all_records, f)

    # Convert to CSV
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(data_dir, "spreads.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(all_records)} spread records saved")
    print(f"  JSON: {progress_path}")
    print(f"  CSV:  {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
