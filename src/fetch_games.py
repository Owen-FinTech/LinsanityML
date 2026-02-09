"""
Fetch historical NBA game data using nba_api.
Pulls regular season + playoff games for specified seasons.
"""

import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.static import teams

# Seasons to fetch (last 5 complete + current)
SEASONS = [
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

# Season types
SEASON_TYPES = ["Regular Season", "Playoffs"]


def fetch_all_games():
    """Fetch all games for the specified seasons."""
    all_games = []

    for season in SEASONS:
        for season_type in SEASON_TYPES:
            print(f"Fetching {season} {season_type}...")
            try:
                finder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable=season_type,
                    league_id_nullable="00",  # NBA
                )
                games = finder.get_data_frames()[0]
                games["SEASON"] = season
                games["SEASON_TYPE"] = season_type
                all_games.append(games)
                print(f"  Found {len(games)} team-game rows")
                time.sleep(1)  # Be respectful to the API
            except Exception as e:
                print(f"  Error: {e}")
                time.sleep(3)

    if not all_games:
        print("No data fetched!")
        return pd.DataFrame()

    df = pd.concat(all_games, ignore_index=True)
    print(f"\nTotal team-game rows: {len(df)}")
    return df


def build_game_pairs(df):
    """
    Convert team-level rows into game-level rows with home/away teams.
    Each game appears twice in the raw data (once per team).
    We pair them into single rows.
    """
    # Sort by game date and game ID
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    # Split into home and away based on MATCHUP field
    # Home games have "vs." in MATCHUP, away games have "@"
    home = df[df["MATCHUP"].str.contains("vs.")].copy()
    away = df[df["MATCHUP"].str.contains("@")].copy()

    # Merge on GAME_ID
    games = home.merge(
        away,
        on="GAME_ID",
        suffixes=("_home", "_away"),
    )

    print(f"Paired into {len(games)} games")
    return games


def main():
    import os

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)

    # Fetch raw game data
    print("=" * 60)
    print("FETCHING NBA GAME DATA")
    print("=" * 60)
    df = fetch_all_games()

    if df.empty:
        return

    # Save raw data
    raw_path = os.path.join(output_dir, "raw_games.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nSaved raw data to {raw_path}")

    # Build game pairs (home vs away)
    print("\n" + "=" * 60)
    print("BUILDING GAME PAIRS")
    print("=" * 60)
    games = build_game_pairs(df)

    # Save paired data
    paired_path = os.path.join(output_dir, "game_pairs.csv")
    games.to_csv(paired_path, index=False)
    print(f"Saved paired data to {paired_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Seasons: {SEASONS}")
    print(f"Total games: {len(games)}")
    print(f"Date range: {games['GAME_DATE_home'].min()} to {games['GAME_DATE_home'].max()}")
    print(f"\nColumns: {list(games.columns)}")


if __name__ == "__main__":
    main()
