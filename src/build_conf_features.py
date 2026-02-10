"""
Build conference-level features:
  - Rolling East vs West inter-conference win rate (last N games)
  - Per-team inter-conference record
  
Computed from existing game data, no API calls needed.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from arena_coordinates import TEAM_ID_TO_ABBREV, TEAM_CONFERENCE


def build_interconf_features(game_pairs):
    """
    For inter-conference games, compute rolling East vs West win rate.
    Also compute each team's rolling record in inter-conference play.
    """
    gp = game_pairs.copy()
    gp["GAME_DATE_home"] = pd.to_datetime(gp["GAME_DATE_home"])
    gp["home_abbrev"] = gp["TEAM_ID_home"].map(TEAM_ID_TO_ABBREV)
    gp["away_abbrev"] = gp["TEAM_ID_away"].map(TEAM_ID_TO_ABBREV)
    gp["home_conf"] = gp["home_abbrev"].map(TEAM_CONFERENCE)
    gp["away_conf"] = gp["away_abbrev"].map(TEAM_CONFERENCE)
    gp["is_interconf"] = (gp["home_conf"] != gp["away_conf"]).astype(int)

    gp = gp.sort_values("GAME_DATE_home").reset_index(drop=True)

    # Rolling East win rate in inter-conference games (last 100 inter-conf games)
    east_wins_rolling = []
    interconf_results = []  # 1 = East won, 0 = West won

    # Per-team inter-conference rolling record (last 20 inter-conf games)
    team_interconf = {}  # team -> list of W/L in interconf

    home_team_interconf_wl = []
    away_team_interconf_wl = []

    for _, row in gp.iterrows():
        is_ic = row["is_interconf"]

        # Rolling East vs West rate (last 100 inter-conf games)
        if len(interconf_results) > 0:
            recent = interconf_results[-100:]
            east_wins_rolling.append(np.mean(recent))
        else:
            east_wins_rolling.append(np.nan)

        # Per-team inter-conf record
        ht = row["home_abbrev"]
        at = row["away_abbrev"]

        # Home team's inter-conf record (last 20)
        if ht in team_interconf and len(team_interconf[ht]) > 0:
            recent = team_interconf[ht][-20:]
            home_team_interconf_wl.append(np.mean(recent))
        else:
            home_team_interconf_wl.append(np.nan)

        # Away team's inter-conf record (last 20)
        if at in team_interconf and len(team_interconf[at]) > 0:
            recent = team_interconf[at][-20:]
            away_team_interconf_wl.append(np.mean(recent))
        else:
            away_team_interconf_wl.append(np.nan)

        # Update after recording
        if is_ic:
            home_won = row["WL_home"] == "W"
            # Determine if East won
            if row["home_conf"] == 0:  # Home is East
                east_won = 1 if home_won else 0
            else:
                east_won = 0 if home_won else 1
            interconf_results.append(east_won)

            # Update per-team records
            if ht not in team_interconf:
                team_interconf[ht] = []
            team_interconf[ht].append(1 if home_won else 0)

            if at not in team_interconf:
                team_interconf[at] = []
            team_interconf[at].append(0 if home_won else 1)

    gp["east_vs_west_wl_100"] = east_wins_rolling
    gp["is_interconf"] = gp["is_interconf"]
    gp["home_interconf_wl_20"] = home_team_interconf_wl
    gp["away_interconf_wl_20"] = away_team_interconf_wl

    return gp[["GAME_ID", "is_interconf", "east_vs_west_wl_100",
               "home_interconf_wl_20", "away_interconf_wl_20"]]


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    print("Loading game pairs...")
    gp = pd.read_csv(os.path.join(data_dir, "game_pairs.csv"))
    print(f"  {len(gp)} games")

    print("\nBuilding inter-conference features...")
    features = build_interconf_features(gp)

    output_path = os.path.join(data_dir, "conf_features.csv")
    features.to_csv(output_path, index=False)

    interconf = features[features["is_interconf"] == 1]
    print(f"\n{'='*60}")
    print("CONFERENCE FEATURES SUMMARY")
    print(f"{'='*60}")
    print(f"Total games: {len(features)}")
    print(f"Inter-conference games: {len(interconf)} ({len(interconf)/len(features)*100:.1f}%)")
    print(f"Mean East vs West WL (rolling 100): {features['east_vs_west_wl_100'].mean():.3f}")
    print(f"\nSaved: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
