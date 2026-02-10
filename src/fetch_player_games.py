"""
Fetch player-level box scores for all games across 5 seasons.
Used to compute:
  1. DNP (absent player) impact per game
  2. Rust/ramp-up factor for returning players
  3. Simplified PER per player per game

Uses nba_api PlayerGameLogs (by season, pulled per team to get full rosters).
"""

import os
import time
import pandas as pd
from nba_api.stats.endpoints import PlayerGameLogs
from arena_coordinates import TEAM_ID_TO_ABBREV

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]


def fetch_season_player_logs(season):
    """
    Fetch all player game logs for a season (regular season).
    Returns a DataFrame with one row per player per game.
    """
    print(f"  Fetching {season}...")
    time.sleep(1)

    logs = PlayerGameLogs(
        season_nullable=season,
        season_type_nullable="Regular Season",
    )
    df = logs.get_data_frames()[0]
    print(f"    {len(df)} player-game records")
    return df


def compute_game_per(row):
    """
    Compute a simplified PER-like efficiency for a single player-game.

    Uses the simplified PER formula (Hollinger approximation):
      PER ≈ (PTS + REB + AST + STL + BLK - TOV - missed FG - missed FT) / MIN

    This is a per-minute efficiency, not the full PER (which needs league pace
    adjustment), but it's consistent within our dataset and good enough for
    measuring relative impact.

    Returns None if minutes = 0 (DNP).
    """
    mins = row.get("MIN", 0)
    if pd.isna(mins) or mins == 0:
        return None

    pts = row.get("PTS", 0) or 0
    reb = row.get("REB", 0) or 0
    ast = row.get("AST", 0) or 0
    stl = row.get("STL", 0) or 0
    blk = row.get("BLK", 0) or 0
    tov = row.get("TOV", 0) or 0
    fga = row.get("FGA", 0) or 0
    fgm = row.get("FGM", 0) or 0
    fta = row.get("FTA", 0) or 0
    ftm = row.get("FTM", 0) or 0

    missed_fg = fga - fgm
    missed_ft = fta - ftm

    efficiency = (pts + reb + ast + stl + blk - tov - missed_fg - missed_ft) / mins
    return round(efficiency, 4)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, "player_game_logs.csv")

    # Check for existing data (resume support)
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        done_seasons = set(existing["SEASON_YEAR"].unique())
        print(f"Existing data: {len(existing)} records from seasons {done_seasons}")
    else:
        existing = pd.DataFrame()
        done_seasons = set()

    all_dfs = [existing] if len(existing) > 0 else []

    for season in SEASONS:
        if season in done_seasons:
            print(f"  {season} already fetched, skipping")
            continue

        try:
            df = fetch_season_player_logs(season)
            all_dfs.append(df)
            time.sleep(2)  # Be nice to the API
        except Exception as e:
            print(f"  ERROR fetching {season}: {e}")
            time.sleep(10)

    if not all_dfs:
        print("No data fetched!")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Parse minutes from "MM:SS" format if needed
    if combined["MIN"].dtype == object:
        def parse_min(val):
            if pd.isna(val):
                return 0
            if isinstance(val, str) and ":" in val:
                parts = val.split(":")
                return int(parts[0]) + int(parts[1]) / 60
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0
        combined["MIN"] = combined["MIN"].apply(parse_min)

    # Compute simplified PER
    print("\nComputing simplified PER for each player-game...")
    combined["SPER"] = combined.apply(compute_game_per, axis=1)

    # Save
    combined.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"PLAYER GAME LOGS SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(combined)}")
    print(f"Seasons: {sorted(combined['SEASON_YEAR'].unique())}")
    print(f"Unique players: {combined['PLAYER_ID'].nunique()}")
    print(f"DNP records (0 min): {(combined['MIN'] == 0).sum()}")
    print(f"SPER computed: {combined['SPER'].notna().sum()}")
    print(f"Saved: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
