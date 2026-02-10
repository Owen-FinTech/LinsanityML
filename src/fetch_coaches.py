"""
Fetch head coach data for all teams across all seasons.
Then compute each coach's historical record vs each opponent (across all teams coached).
"""

import os
import time
import pandas as pd
from nba_api.stats.endpoints import CommonTeamRoster
from arena_coordinates import TEAM_ID_TO_ABBREV

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

# Also fetch a few prior seasons for coach history context
HISTORY_SEASONS = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20"] + SEASONS


def fetch_all_coaches():
    """Fetch head coach for each team-season."""
    records = []
    team_ids = list(TEAM_ID_TO_ABBREV.keys())

    for season in HISTORY_SEASONS:
        print(f"  Fetching coaches for {season}...")
        for tid in team_ids:
            try:
                roster = CommonTeamRoster(team_id=tid, season=season)
                coaches_df = roster.get_data_frames()[1]
                head = coaches_df[coaches_df["IS_ASSISTANT"] == 1]  # 1 = Head Coach
                if len(head) > 0:
                    coach = head.iloc[0]
                    records.append({
                        "SEASON": season,
                        "TEAM_ID": tid,
                        "TEAM": TEAM_ID_TO_ABBREV[tid],
                        "COACH_ID": coach["COACH_ID"],
                        "COACH_NAME": coach["COACH_NAME"],
                    })
                time.sleep(0.6)  # Rate limit
            except Exception as e:
                print(f"    Error {TEAM_ID_TO_ABBREV[tid]} {season}: {e}")
                time.sleep(3)

    return pd.DataFrame(records)


def build_coach_vs_opponent(coaches_df, game_pairs):
    """
    For each game, compute the head coach's historical record vs the opponent
    (across all teams they've coached).
    
    Returns features:
      - home_coach_vs_opp_wl: home coach's win rate vs away team (career)
      - away_coach_vs_opp_wl: away coach's win rate vs home team (career)
      - home_coach_vs_opp_n: number of games in that history
      - away_coach_vs_opp_n: number of games in that history
    """
    gp = game_pairs.copy()
    gp["GAME_DATE_home"] = pd.to_datetime(gp["GAME_DATE_home"])
    gp["home_abbrev"] = gp["TEAM_ID_home"].map(TEAM_ID_TO_ABBREV)
    gp["away_abbrev"] = gp["TEAM_ID_away"].map(TEAM_ID_TO_ABBREV)

    # Map season string to games
    # Build coach lookup: (team, season) -> coach_id
    coach_lookup = {}
    for _, row in coaches_df.iterrows():
        coach_lookup[(row["TEAM"], row["SEASON"])] = row["COACH_ID"]

    # Add coach_id to each game
    gp["home_coach_id"] = gp.apply(
        lambda r: coach_lookup.get((r["home_abbrev"], r["SEASON_home"])), axis=1
    )
    gp["away_coach_id"] = gp.apply(
        lambda r: coach_lookup.get((r["away_abbrev"], r["SEASON_home"])), axis=1
    )

    # Sort by date
    gp = gp.sort_values("GAME_DATE_home").reset_index(drop=True)

    # Build rolling coach record vs opponent
    # coach_record[(coach_id, opp_team)] = [wins, total]
    coach_record = {}

    home_wl = []
    away_wl = []
    home_n = []
    away_n = []

    for _, game in gp.iterrows():
        hc = game["home_coach_id"]
        ac = game["away_coach_id"]
        home_team = game["home_abbrev"]
        away_team = game["away_abbrev"]
        home_won = game.get("WL_home") == "W"

        # Record BEFORE this game
        # Home coach vs away team
        key_h = (hc, away_team)
        if hc and key_h in coach_record:
            w, t = coach_record[key_h]
            home_wl.append(w / t if t > 0 else None)
            home_n.append(t)
        else:
            home_wl.append(None)
            home_n.append(0)

        # Away coach vs home team
        key_a = (ac, home_team)
        if ac and key_a in coach_record:
            w, t = coach_record[key_a]
            away_wl.append(w / t if t > 0 else None)
            away_n.append(t)
        else:
            away_wl.append(None)
            away_n.append(0)

        # Update records AFTER recording
        if hc:
            if key_h not in coach_record:
                coach_record[key_h] = [0, 0]
            coach_record[key_h][1] += 1
            if home_won:
                coach_record[key_h][0] += 1

        if ac:
            if key_a not in coach_record:
                coach_record[key_a] = [0, 0]
            coach_record[key_a][1] += 1
            if not home_won:
                coach_record[key_a][0] += 1

    gp["home_coach_vs_opp_wl"] = home_wl
    gp["away_coach_vs_opp_wl"] = away_wl
    gp["home_coach_vs_opp_n"] = home_n
    gp["away_coach_vs_opp_n"] = away_n

    return gp[["GAME_ID", "home_coach_vs_opp_wl", "away_coach_vs_opp_wl",
               "home_coach_vs_opp_n", "away_coach_vs_opp_n"]]


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    coaches_path = os.path.join(data_dir, "coaches.csv")

    if os.path.exists(coaches_path):
        print("Loading existing coach data...")
        coaches = pd.read_csv(coaches_path)
    else:
        print("Fetching coach data from NBA API...")
        coaches = fetch_all_coaches()
        coaches.to_csv(coaches_path, index=False)

    print(f"  {len(coaches)} coach-team-season records")
    print(f"  Unique head coaches: {coaches['COACH_ID'].nunique()}")

    print("\nLoading game pairs...")
    gp = pd.read_csv(os.path.join(data_dir, "game_pairs.csv"))
    print(f"  {len(gp)} games")

    print("\nComputing coach vs opponent features...")
    features = build_coach_vs_opponent(coaches, gp)

    output_path = os.path.join(data_dir, "coach_features.csv")
    features.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("COACH FEATURES SUMMARY")
    print(f"{'='*60}")
    print(f"Games with home coach history: {features['home_coach_vs_opp_wl'].notna().sum()}")
    print(f"Games with away coach history: {features['away_coach_vs_opp_wl'].notna().sum()}")
    print(f"Mean home coach vs opp WL: {features['home_coach_vs_opp_wl'].mean():.3f}")
    print(f"Mean away coach vs opp WL: {features['away_coach_vs_opp_wl'].mean():.3f}")
    print(f"Mean home coach matchup games: {features['home_coach_vs_opp_n'].mean():.1f}")
    print(f"\nSaved: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
