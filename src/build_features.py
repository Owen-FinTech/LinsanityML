"""
Build the feature set defined in 'Defining Initial Features.txt'.
Takes paired game data and computes rolling stats, distances, etc.
"""

import os
import math
import pandas as pd
import numpy as np
from arena_coordinates import (
    ARENA_COORDS,
    ARENA_ALTITUDE_M,
    TEAM_ID_TO_ABBREV,
    TEAM_ALPHA_INDEX,
    TEAM_CONFERENCE,
)


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def get_arena_for_game(home_abbrev):
    """Get arena coordinates for a game (always at home team's arena)."""
    return ARENA_COORDS.get(home_abbrev, (None, None))


def compute_rolling_stats(team_games, n):
    """
    Compute rolling win-loss ratio and average points for last n games.
    Returns Series aligned to team_games index.
    """
    wins = (team_games["WL"] == "W").astype(float)
    wl_ratio = wins.rolling(n, min_periods=1).mean().shift(1)
    pts_avg = team_games["PTS"].rolling(n, min_periods=1).mean().shift(1)
    pts_allowed_avg = team_games["PTS_OPP"].rolling(n, min_periods=1).mean().shift(1)
    return wl_ratio, pts_avg, pts_allowed_avg


def compute_opponent_rolling_stats(team_games, n):
    """
    Compute rolling stats against specific opponents over last n meetings.
    """
    results = pd.DataFrame(index=team_games.index)
    results["wl"] = np.nan
    results["pts"] = np.nan
    results["pts_allowed"] = np.nan

    # Group by opponent and compute rolling within each matchup
    for opp, group in team_games.groupby("OPP_ABBREV"):
        wins = (group["WL"] == "W").astype(float)
        wl = wins.rolling(n, min_periods=1).mean().shift(1)
        pts = group["PTS"].rolling(n, min_periods=1).mean().shift(1)
        pts_a = group["PTS_OPP"].rolling(n, min_periods=1).mean().shift(1)
        results.loc[group.index, "wl"] = wl
        results.loc[group.index, "pts"] = pts
        results.loc[group.index, "pts_allowed"] = pts_a

    return results["wl"], results["pts"], results["pts_allowed"]


def build_team_histories(games_df):
    """
    Build a chronological game history for each team with rolling stats.
    """
    records = []

    # Create one record per team per game
    for _, row in games_df.iterrows():
        game_date = row["GAME_DATE_home"]
        game_id = row["GAME_ID"]
        home_abbrev = TEAM_ID_TO_ABBREV.get(row["TEAM_ID_home"], row["TEAM_ABBREVIATION_home"])
        away_abbrev = TEAM_ID_TO_ABBREV.get(row["TEAM_ID_away"], row["TEAM_ABBREVIATION_away"])

        # Home team record
        records.append({
            "GAME_ID": game_id,
            "GAME_DATE": game_date,
            "TEAM_ABBREV": home_abbrev,
            "OPP_ABBREV": away_abbrev,
            "IS_HOME": True,
            "WL": row["WL_home"],
            "PTS": row["PTS_home"],
            "PTS_OPP": row["PTS_away"],
            "SEASON": row["SEASON_home"],
            "SEASON_TYPE": row["SEASON_TYPE_home"],
        })

        # Away team record
        records.append({
            "GAME_ID": game_id,
            "GAME_DATE": game_date,
            "TEAM_ABBREV": away_abbrev,
            "OPP_ABBREV": home_abbrev,
            "IS_HOME": False,
            "WL": row["WL_away"],
            "PTS": row["PTS_away"],
            "PTS_OPP": row["PTS_home"],
            "SEASON": row["SEASON_home"],
            "SEASON_TYPE": row["SEASON_TYPE_home"],
        })

    team_df = pd.DataFrame(records)
    team_df["GAME_DATE"] = pd.to_datetime(team_df["GAME_DATE"])
    team_df = team_df.sort_values(["TEAM_ABBREV", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    # Compute game number within each season for each team
    team_df["GAME_NUM"] = team_df.groupby(["TEAM_ABBREV", "SEASON"]).cumcount() + 1

    # Compute rolling stats per team
    all_team_stats = []
    for team, tg in team_df.groupby("TEAM_ABBREV"):
        tg = tg.sort_values(["GAME_DATE", "GAME_ID"]).copy()

        # Overall rolling stats
        tg["wl_last_2"], tg["pts_last_2"], tg["allow_last_2"] = compute_rolling_stats(tg, 2)
        tg["wl_last_10"], tg["pts_last_10"], tg["allow_last_10"] = compute_rolling_stats(tg, 10)

        # Opponent-specific rolling stats
        tg["wl_opp_last_2"], tg["pts_opp_last_2"], tg["allow_opp_last_2"] = compute_opponent_rolling_stats(tg, 2)
        tg["wl_opp_last_10"], tg["pts_opp_last_10"], tg["allow_opp_last_10"] = compute_opponent_rolling_stats(tg, 10)

        # Distance travelled from previous game venue
        distances = [np.nan]  # First game has no previous
        for i in range(1, len(tg)):
            prev_row = tg.iloc[i - 1]
            curr_row = tg.iloc[i]

            # Previous game location
            if prev_row["IS_HOME"]:
                prev_arena = prev_row["TEAM_ABBREV"]
            else:
                prev_arena = prev_row["OPP_ABBREV"]

            # Current game location
            if curr_row["IS_HOME"]:
                curr_arena = curr_row["TEAM_ABBREV"]
            else:
                curr_arena = curr_row["OPP_ABBREV"]

            prev_coords = ARENA_COORDS.get(prev_arena)
            curr_coords = ARENA_COORDS.get(curr_arena)

            if prev_coords and curr_coords:
                dist = haversine_km(prev_coords[0], prev_coords[1], curr_coords[0], curr_coords[1])
            else:
                dist = np.nan

            distances.append(dist)

        tg["dist_trav"] = distances

        # Rest days (days since previous game)
        game_dates = tg["GAME_DATE"].values
        rest = [np.nan]
        for i in range(1, len(tg)):
            delta = (game_dates[i] - game_dates[i - 1]) / np.timedelta64(1, "D")
            rest.append(delta)
        tg["rest_days"] = rest

        # Win/loss streak: positive = win streak, negative = loss streak
        wl_binary = (tg["WL"] == "W").astype(int).values
        streaks = [0.0]
        for i in range(1, len(tg)):
            # Streak BEFORE this game (shift by 1)
            prev = wl_binary[i - 1]
            if i == 1:
                streaks.append(1.0 if prev == 1 else -1.0)
            else:
                prev_streak = streaks[-1]
                if prev == 1:
                    streaks.append(max(prev_streak, 0) + 1)
                else:
                    streaks.append(min(prev_streak, 0) - 1)
        # Shift: streak entering the game (not including its result)
        tg["streak"] = [np.nan] + streaks[1:]

        # Home/away win rate (last 10 home or away games respectively)
        home_mask = tg["IS_HOME"].values
        home_wins = []
        away_wins = []
        home_count = 0
        away_count = 0
        home_w = 0
        away_w = 0
        # Use simple expanding lists for last-10 home/away
        h_results = []
        a_results = []
        for i in range(len(tg)):
            # Record rate BEFORE this game
            if home_mask[i]:
                home_wins.append(len(h_results) and sum(h_results[-10:]) / len(h_results[-10:]) or np.nan)
                away_wins.append(len(a_results) and sum(a_results[-10:]) / len(a_results[-10:]) or np.nan)
                h_results.append(1 if wl_binary[i] else 0)
            else:
                home_wins.append(len(h_results) and sum(h_results[-10:]) / len(h_results[-10:]) or np.nan)
                away_wins.append(len(a_results) and sum(a_results[-10:]) / len(a_results[-10:]) or np.nan)
                a_results.append(1 if wl_binary[i] else 0)
        tg["home_wl_last_10"] = home_wins
        tg["away_wl_last_10"] = away_wins

        # Pace proxy: total points (own + opponent) rolling last 10
        tg["pace_last_10"] = (tg["PTS"] + tg["PTS_OPP"]).rolling(10, min_periods=1).mean().shift(1)

        all_team_stats.append(tg)

    return pd.concat(all_team_stats, ignore_index=True)


def build_feature_matrix(games_df, team_histories):
    """
    Build the final feature matrix matching the defined feature set.
    """
    games_df = games_df.copy()
    games_df["GAME_DATE_home"] = pd.to_datetime(games_df["GAME_DATE_home"])

    # Create lookup from team histories
    th = team_histories.set_index(["GAME_ID", "TEAM_ABBREV"])

    features = []

    for _, row in games_df.iterrows():
        game_id = row["GAME_ID"]
        game_date = pd.to_datetime(row["GAME_DATE_home"])
        home_abbrev = TEAM_ID_TO_ABBREV.get(row["TEAM_ID_home"], row["TEAM_ABBREVIATION_home"])
        away_abbrev = TEAM_ID_TO_ABBREV.get(row["TEAM_ID_away"], row["TEAM_ABBREVIATION_away"])

        # Lookup team histories for this game
        try:
            home_stats = th.loc[(game_id, home_abbrev)]
            away_stats = th.loc[(game_id, away_abbrev)]
        except KeyError:
            continue

        # Game-level features
        feat = {
            "game_id": game_id,
            "game_date": game_date,
            "season": row["SEASON_home"],
            "season_type": row["SEASON_TYPE_home"],
            "home_team": home_abbrev,
            "away_team": away_abbrev,

            # Target: actual point differential (home - away)
            # Will be used with spread to determine spread_covered
            "home_pts": row["PTS_home"],
            "away_pts": row["PTS_away"],
            "actual_margin": row["PTS_home"] - row["PTS_away"],

            # Date features
            "g_yr": game_date.year % 2000,
            "g_day_wk": game_date.isoweekday() % 7 + 1,  # 1=Sun, 7=Sat
            "g_mth_yr": game_date.month,
            "g_day_mth": game_date.day,
            # g_time: would need tip-off time data (not in basic API)
            # f_d_sprd_away: needs The Odds API

            # Home team features
            "home_t_idx": TEAM_ALPHA_INDEX.get(home_abbrev, 0),
            "home_conf": TEAM_CONFERENCE.get(home_abbrev, 0),
            "home_dist_trav": home_stats.get("dist_trav", np.nan),
            "home_g_num": home_stats.get("GAME_NUM", np.nan),
            "home_w_l_last_2": home_stats.get("wl_last_2", np.nan),
            "home_w_l_last_10": home_stats.get("wl_last_10", np.nan),
            "home_w_l_opp_last_2": home_stats.get("wl_opp_last_2", np.nan),
            "home_w_l_opp_last_10": home_stats.get("wl_opp_last_10", np.nan),
            "home_pts_last_2": home_stats.get("pts_last_2", np.nan),
            "home_pts_last_10": home_stats.get("pts_last_10", np.nan),
            "home_allow_last_2": home_stats.get("allow_last_2", np.nan),
            "home_allow_last_10": home_stats.get("allow_last_10", np.nan),
            "home_pts_opp_last_2": home_stats.get("pts_opp_last_2", np.nan),
            "home_pts_opp_last_10": home_stats.get("pts_opp_last_10", np.nan),
            "home_allow_opp_last_2": home_stats.get("allow_opp_last_2", np.nan),
            "home_allow_opp_last_10": home_stats.get("allow_opp_last_10", np.nan),

            # New: rest days, streaks, home/away splits, pace, altitude
            "home_rest_days": home_stats.get("rest_days", np.nan),
            "home_streak": home_stats.get("streak", np.nan),
            "home_home_wl_10": home_stats.get("home_wl_last_10", np.nan),
            "home_away_wl_10": home_stats.get("away_wl_last_10", np.nan),
            "home_pace_10": home_stats.get("pace_last_10", np.nan),
            "home_altitude_m": ARENA_ALTITUDE_M.get(home_abbrev, np.nan),

            # Away team features
            "away_t_idx": TEAM_ALPHA_INDEX.get(away_abbrev, 0),
            "away_conf": TEAM_CONFERENCE.get(away_abbrev, 0),
            "away_dist_trav": away_stats.get("dist_trav", np.nan),
            "away_g_num": away_stats.get("GAME_NUM", np.nan),
            "away_w_l_last_2": away_stats.get("wl_last_2", np.nan),
            "away_w_l_last_10": away_stats.get("wl_last_10", np.nan),
            "away_w_l_opp_last_2": away_stats.get("wl_opp_last_2", np.nan),
            "away_w_l_opp_last_10": away_stats.get("wl_opp_last_10", np.nan),
            "away_pts_last_2": away_stats.get("pts_last_2", np.nan),
            "away_pts_last_10": away_stats.get("pts_last_10", np.nan),
            "away_allow_last_2": away_stats.get("allow_last_2", np.nan),
            "away_allow_last_10": away_stats.get("allow_last_10", np.nan),
            "away_pts_opp_last_2": away_stats.get("pts_opp_last_2", np.nan),
            "away_pts_opp_last_10": away_stats.get("pts_opp_last_10", np.nan),
            "away_allow_opp_last_2": away_stats.get("allow_opp_last_2", np.nan),
            "away_allow_opp_last_10": away_stats.get("allow_opp_last_10", np.nan),

            # New: away rest days, streaks, splits, pace
            "away_rest_days": away_stats.get("rest_days", np.nan),
            "away_streak": away_stats.get("streak", np.nan),
            "away_home_wl_10": away_stats.get("home_wl_last_10", np.nan),
            "away_away_wl_10": away_stats.get("away_wl_last_10", np.nan),
            "away_pace_10": away_stats.get("pace_last_10", np.nan),

            # Pace differential (home - away): positive = home plays faster
            "pace_diff": (home_stats.get("pace_last_10", np.nan) or np.nan) - (away_stats.get("pace_last_10", np.nan) or np.nan) if pd.notna(home_stats.get("pace_last_10")) and pd.notna(away_stats.get("pace_last_10")) else np.nan,

            # Rest advantage (positive = home more rested)
            "rest_advantage": (home_stats.get("rest_days", np.nan) or np.nan) - (away_stats.get("rest_days", np.nan) or np.nan) if pd.notna(home_stats.get("rest_days")) and pd.notna(away_stats.get("rest_days")) else np.nan,
        }

        features.append(feat)

    feature_df = pd.DataFrame(features)
    return feature_df


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    print("Loading paired game data...")
    games = pd.read_csv(os.path.join(data_dir, "game_pairs.csv"))
    print(f"  {len(games)} games loaded")

    print("\nBuilding team histories with rolling stats...")
    team_histories = build_team_histories(games)
    print(f"  {len(team_histories)} team-game records")

    print("\nBuilding feature matrix...")
    features = build_feature_matrix(games, team_histories)
    print(f"  {len(features)} games with features")

    # Merge weather data if available
    weather_path = os.path.join(data_dir, "weather_cache.csv")
    if os.path.exists(weather_path):
        print("\nMerging weather data...")
        weather = pd.read_csv(weather_path)
        weather = weather.rename(columns={"team": "home_team", "date": "game_date", "temp_c": "home_temp_c"})
        weather["game_date"] = pd.to_datetime(weather["game_date"])
        features["game_date"] = pd.to_datetime(features["game_date"])
        before = len(features)
        features = features.merge(weather, on=["home_team", "game_date"], how="left")
        print(f"  Matched {features['home_temp_c'].notna().sum()}/{before} games with temperature")
    else:
        features["home_temp_c"] = np.nan
        print("\nNo weather data found — run fetch_weather.py first to add temperatures")

    # Merge injury/absence features if available
    injury_path = os.path.join(data_dir, "injury_features.csv")
    if os.path.exists(injury_path):
        print("\nMerging injury/rust features...")
        injury = pd.read_csv(injury_path)
        features = features.merge(injury, on="game_id", how="left")
        print(f"  Matched {features['home_absent_impact'].notna().sum()}/{len(features)} games")
    else:
        print("\nNo injury data found — run build_injury_features.py first")

    # Merge coach features if available
    coach_path = os.path.join(data_dir, "coach_features.csv")
    if os.path.exists(coach_path):
        print("\nMerging coach vs opponent features...")
        coach = pd.read_csv(coach_path)
        coach = coach.rename(columns={"GAME_ID": "game_id"})
        features = features.merge(coach, on="game_id", how="left")
        print(f"  Matched {features['home_coach_vs_opp_wl'].notna().sum()}/{len(features)} games with coach data")
    else:
        print("\nNo coach data found — run fetch_coaches.py first")

    # Save
    features_path = os.path.join(data_dir, "features.csv")
    features.to_csv(features_path, index=False)
    print(f"\nSaved features to {features_path}")

    # Summary stats
    print("\n" + "=" * 60)
    print("FEATURE MATRIX SUMMARY")
    print("=" * 60)
    print(f"Shape: {features.shape}")
    print(f"\nNull counts (top 10):")
    nulls = features.isnull().sum().sort_values(ascending=False)
    print(nulls[nulls > 0].head(10).to_string())
    print(f"\nSample row:")
    print(features.iloc[50].to_string())


if __name__ == "__main__":
    main()
