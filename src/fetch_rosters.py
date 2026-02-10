"""
Fetch team roster data (age, height, weight) for all teams across all seasons.
Also fetches country data for international player counting.
"""

import os
import time
import re
import pandas as pd
from nba_api.stats.endpoints import CommonTeamRoster, CommonPlayerInfo
from arena_coordinates import TEAM_ID_TO_ABBREV

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]


def height_to_cm(height_str):
    """Convert '6-8' format to centimeters."""
    if not height_str or pd.isna(height_str):
        return None
    match = re.match(r"(\d+)-(\d+)", str(height_str))
    if match:
        feet, inches = int(match.group(1)), int(match.group(2))
        return round(feet * 30.48 + inches * 2.54, 1)
    return None


def weight_to_kg(weight_str):
    """Convert pounds to kg."""
    try:
        lbs = float(weight_str)
        return round(lbs * 0.453592, 1)
    except (ValueError, TypeError):
        return None


def fetch_all_rosters():
    """Fetch roster data for all teams across all seasons."""
    records = []
    team_ids = list(TEAM_ID_TO_ABBREV.keys())

    for season in SEASONS:
        print(f"  Fetching rosters for {season}...")
        for tid in team_ids:
            try:
                roster = CommonTeamRoster(team_id=tid, season=season)
                players = roster.get_data_frames()[0]
                team = TEAM_ID_TO_ABBREV[tid]

                for _, p in players.iterrows():
                    records.append({
                        "SEASON": season,
                        "TEAM": team,
                        "PLAYER_ID": p["PLAYER_ID"],
                        "PLAYER": p["PLAYER"],
                        "AGE": p.get("AGE"),
                        "HEIGHT_CM": height_to_cm(p.get("HEIGHT")),
                        "WEIGHT_KG": weight_to_kg(p.get("WEIGHT")),
                        "POSITION": p.get("POSITION"),
                        "EXP": p.get("EXP"),
                    })
                time.sleep(0.6)
            except Exception as e:
                print(f"    Error {TEAM_ID_TO_ABBREV[tid]} {season}: {e}")
                time.sleep(3)

    return pd.DataFrame(records)


def fetch_player_countries(player_ids):
    """Fetch country for each unique player. Cached to avoid repeat calls."""
    countries = {}
    total = len(player_ids)
    print(f"  Fetching country for {total} unique players...")

    for i, pid in enumerate(player_ids):
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{total} players...")
        try:
            info = CommonPlayerInfo(player_id=pid)
            df = info.get_data_frames()[0]
            if len(df) > 0:
                countries[pid] = df.iloc[0].get("COUNTRY", "USA")
            time.sleep(0.6)
        except Exception as e:
            countries[pid] = None
            time.sleep(2)

    return countries


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    roster_path = os.path.join(data_dir, "rosters.csv")
    countries_path = os.path.join(data_dir, "player_countries.csv")

    # Fetch rosters
    if os.path.exists(roster_path):
        print("Loading existing roster data...")
        rosters = pd.read_csv(roster_path)
    else:
        print("Fetching roster data from NBA API...")
        rosters = fetch_all_rosters()
        rosters.to_csv(roster_path, index=False)

    print(f"  {len(rosters)} player-team-season records")

    # Fetch countries
    if os.path.exists(countries_path):
        print("Loading existing country data...")
        countries_df = pd.read_csv(countries_path)
    else:
        unique_players = rosters["PLAYER_ID"].unique()
        print(f"\nFetching country data for {len(unique_players)} unique players...")
        countries = fetch_player_countries(unique_players)
        countries_df = pd.DataFrame([
            {"PLAYER_ID": pid, "COUNTRY": c} for pid, c in countries.items()
        ])
        countries_df.to_csv(countries_path, index=False)

    # Merge
    rosters = rosters.merge(countries_df, on="PLAYER_ID", how="left")
    rosters["IS_INTERNATIONAL"] = (
        rosters["COUNTRY"].notna() &
        (rosters["COUNTRY"] != "USA") &
        (rosters["COUNTRY"] != "")
    ).astype(int)

    # Save enriched roster
    enriched_path = os.path.join(data_dir, "rosters_enriched.csv")
    rosters.to_csv(enriched_path, index=False)

    print(f"\n{'='*60}")
    print("ROSTER DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(rosters)}")
    print(f"Unique players: {rosters['PLAYER_ID'].nunique()}")
    print(f"International players: {rosters[rosters['IS_INTERNATIONAL']==1]['PLAYER_ID'].nunique()}")
    print(f"Mean age: {rosters['AGE'].mean():.1f}")
    print(f"Mean height: {rosters['HEIGHT_CM'].mean():.1f} cm")
    print(f"Mean weight: {rosters['WEIGHT_KG'].mean():.1f} kg")
    print(f"\nSaved: {enriched_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
