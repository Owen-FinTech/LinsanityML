"""
LinsanityML — Daily Prediction Pipeline

Fetches today's NBA games and FanDuel spreads, generates features,
and runs predictions using the current production model.

Usage:
  python daily_predict.py              # Today's games
  python daily_predict.py 2026-02-15   # Specific date

Output:
  - Console predictions with confidence levels
  - Saves to data/predictions/YYYY-MM-DD.csv
  - Logs to data/predictions/history.csv for tracking accuracy
"""

import os
import sys
import json
import pickle
import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests as req
from nba_api.stats.endpoints import ScoreboardV2, LeagueGameLog

sys.path.insert(0, os.path.dirname(__file__))
from arena_coordinates import (
    ARENA_COORDS, ARENA_ALTITUDE_M, TEAM_ID_TO_ABBREV,
    TEAM_NAME_TO_ABBREV, TEAM_ALPHA_INDEX, TEAM_CONFERENCE,
)

# Load .env file if present
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PRED_DIR = os.path.join(DATA_DIR, "predictions")


def get_model_version():
    """Read current model version from version file."""
    version_path = os.path.join(MODEL_DIR, "version.json")
    if os.path.exists(version_path):
        with open(version_path) as f:
            return json.load(f)
    return {"version": "v1.0", "name": "LinsanityML v1.0", "date": "2026-02-10"}


def load_model():
    """Load the current production model."""
    model_path = os.path.join(MODEL_DIR, "production_model.pkl")
    if not os.path.exists(model_path):
        # Fall back to best_model_v2
        model_path = os.path.join(MODEL_DIR, "best_model_v2.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "best_model.pkl")

    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_cols"], data.get("model_name", "Unknown")


def fetch_todays_games(date_str):
    """Fetch today's NBA schedule."""
    print(f"  Fetching games for {date_str}...")
    try:
        scoreboard = ScoreboardV2(game_date=date_str)
        games_df = scoreboard.get_data_frames()[0]
        if len(games_df) == 0:
            return []

        games = []
        for _, row in games_df.iterrows():
            home_id = row.get("HOME_TEAM_ID")
            away_id = row.get("VISITOR_TEAM_ID")
            home = TEAM_ID_TO_ABBREV.get(home_id)
            away = TEAM_ID_TO_ABBREV.get(away_id)
            if home and away:
                games.append({
                    "game_id": row.get("GAME_ID"),
                    "home_team": home,
                    "away_team": away,
                    "game_date": date_str,
                    "status": row.get("GAME_STATUS_TEXT", ""),
                })
        return games
    except Exception as e:
        print(f"  Error fetching schedule: {e}")
        return []


def fetch_spreads(date_str):
    """Fetch current FanDuel spreads for today's games."""
    print(f"  Fetching FanDuel spreads...")
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "spreads",
        "bookmakers": "fanduel",
        "oddsFormat": "american",
    }

    try:
        resp = req.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"  Odds API error: {resp.status_code}")
            return {}

        remaining = resp.headers.get("x-requests-remaining", "?")
        print(f"  API credits remaining: {remaining}")

        spreads = {}
        for game in resp.json():
            away_team = TEAM_NAME_TO_ABBREV.get(game.get("away_team"))
            home_team = TEAM_NAME_TO_ABBREV.get(game.get("home_team"))
            if not away_team or not home_team:
                continue

            for bk in game.get("bookmakers", []):
                if bk.get("key") != "fanduel":
                    continue
                for market in bk.get("markets", []):
                    if market.get("key") != "spreads":
                        continue
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == game.get("away_team"):
                            spreads[(home_team, away_team)] = outcome.get("point")

        return spreads
    except Exception as e:
        print(f"  Error fetching spreads: {e}")
        return {}


def get_recent_team_stats(team, n_games=10):
    """
    Get a team's recent game stats for feature generation.
    Uses the current season's game log.
    """
    try:
        log = LeagueGameLog(season="2024-25")
        all_games = log.get_data_frames()[0]

        team_games = all_games[all_games["TEAM_ABBREVIATION"] == team].copy()
        team_games["GAME_DATE"] = pd.to_datetime(team_games["GAME_DATE"])
        team_games = team_games.sort_values("GAME_DATE", ascending=False)

        return team_games.head(n_games)
    except Exception as e:
        print(f"  Error getting stats for {team}: {e}")
        return pd.DataFrame()


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def build_game_features(home, away, spread, game_date, all_games_df):
    """
    Build feature vector for a single upcoming game.
    Mirrors the features from build_features.py but computed live.
    """
    game_date = pd.to_datetime(game_date)

    # Filter to games before this date
    hist = all_games_df[all_games_df["GAME_DATE"] < game_date].copy()

    feat = {}

    # Date features
    feat["g_yr"] = game_date.year % 2000
    feat["g_day_wk"] = game_date.isoweekday() % 7 + 1
    feat["g_mth_yr"] = game_date.month
    feat["g_day_mth"] = game_date.day
    feat["f_d_sprd_away"] = spread

    for side, team in [("home", home), ("away", away)]:
        team_games = hist[hist["TEAM_ABBREVIATION"] == team].sort_values("GAME_DATE")

        feat[f"{side}_t_idx"] = TEAM_ALPHA_INDEX.get(team, 0)
        feat[f"{side}_conf"] = TEAM_CONFERENCE.get(team, 0)

        if len(team_games) == 0:
            # No history available, fill with NaN
            for suffix in ["w_l_last_2", "w_l_last_10", "pts_last_2", "pts_last_10",
                           "allow_last_2", "allow_last_10", "g_num", "dist_trav",
                           "streak", "away_wl_10", "home_wl_10", "pace_10",
                           "rest_days"]:
                feat[f"{side}_{suffix}"] = np.nan
            continue

        # Game number
        season_games = team_games[team_games["GAME_DATE"] >= f"{game_date.year if game_date.month >= 10 else game_date.year - 1}-10-01"]
        feat[f"{side}_g_num"] = len(season_games)

        # Win/loss ratios
        wins = (team_games["WL"] == "W").astype(float)
        last_2 = team_games.tail(2)
        last_10 = team_games.tail(10)

        feat[f"{side}_w_l_last_2"] = (last_2["WL"] == "W").mean() if len(last_2) > 0 else np.nan
        feat[f"{side}_w_l_last_10"] = (last_10["WL"] == "W").mean() if len(last_10) > 0 else np.nan

        # Points
        feat[f"{side}_pts_last_2"] = last_2["PTS"].mean() if len(last_2) > 0 else np.nan
        feat[f"{side}_pts_last_10"] = last_10["PTS"].mean() if len(last_10) > 0 else np.nan

        # Points allowed (need to compute from matchup data)
        # Using opponent points from PLUS_MINUS: PTS_allowed ≈ PTS - PLUS_MINUS
        if "PLUS_MINUS" in team_games.columns:
            team_games = team_games.copy()
            team_games["PTS_ALLOWED"] = team_games["PTS"] - team_games["PLUS_MINUS"]
            feat[f"{side}_allow_last_2"] = last_2["PTS"].values.mean() - last_2["PLUS_MINUS"].values.mean() if len(last_2) > 0 else np.nan
            feat[f"{side}_allow_last_10"] = last_10["PTS"].values.mean() - last_10["PLUS_MINUS"].values.mean() if len(last_10) > 0 else np.nan
        else:
            feat[f"{side}_allow_last_2"] = np.nan
            feat[f"{side}_allow_last_10"] = np.nan

        # Rest days
        if len(team_games) >= 1:
            last_game_date = pd.to_datetime(team_games.iloc[-1]["GAME_DATE"])
            feat[f"{side}_rest_days"] = (game_date - last_game_date).days
        else:
            feat[f"{side}_rest_days"] = np.nan

        # Streak
        recent = team_games.tail(20)["WL"].values
        streak = 0
        if len(recent) > 0:
            last_result = recent[-1]
            for r in reversed(recent):
                if r == last_result:
                    streak += 1
                else:
                    break
            if last_result == "L":
                streak = -streak
        feat[f"{side}_streak"] = streak

        # Home/away splits (last 10 of each)
        matchup_col = "MATCHUP"
        if matchup_col in team_games.columns:
            home_games = team_games[team_games[matchup_col].str.contains("vs.")]
            away_games = team_games[team_games[matchup_col].str.contains("@")]
            feat[f"{side}_home_wl_10"] = (home_games.tail(10)["WL"] == "W").mean() if len(home_games) > 0 else np.nan
            feat[f"{side}_away_wl_10"] = (away_games.tail(10)["WL"] == "W").mean() if len(away_games) > 0 else np.nan
        else:
            feat[f"{side}_home_wl_10"] = np.nan
            feat[f"{side}_away_wl_10"] = np.nan

        # Distance travelled (from last game venue to this game venue)
        game_arena = home  # Game is at home team's arena
        if len(team_games) > 0 and matchup_col in team_games.columns:
            last_matchup = team_games.iloc[-1][matchup_col]
            if "@" in last_matchup:
                # Was away, extract opponent
                prev_venue = last_matchup.split("@ ")[-1].strip()
                # Try to map to abbreviation
                prev_abbrev = None
                for name, abbr in TEAM_NAME_TO_ABBREV.items():
                    if abbr in prev_venue or prev_venue in name:
                        prev_abbrev = abbr
                        break
                if prev_abbrev is None:
                    prev_abbrev = team  # Fallback
            else:
                prev_abbrev = team  # Was home

            coords_prev = ARENA_COORDS.get(prev_abbrev)
            coords_curr = ARENA_COORDS.get(game_arena)
            if coords_prev and coords_curr:
                feat[f"{side}_dist_trav"] = haversine_km(
                    coords_prev[0], coords_prev[1],
                    coords_curr[0], coords_curr[1]
                )
            else:
                feat[f"{side}_dist_trav"] = np.nan
        else:
            feat[f"{side}_dist_trav"] = np.nan

        # Pace (total points proxy)
        if "PLUS_MINUS" in team_games.columns:
            last10 = team_games.tail(10).copy()
            last10["TOTAL"] = last10["PTS"] + (last10["PTS"] - last10["PLUS_MINUS"])
            feat[f"{side}_pace_10"] = last10["TOTAL"].mean()
        else:
            feat[f"{side}_pace_10"] = np.nan

    # Derived features
    feat["home_altitude_m"] = ARENA_ALTITUDE_M.get(home, np.nan)
    feat["pace_diff"] = (feat.get("home_pace_10", np.nan) or np.nan) - (feat.get("away_pace_10", np.nan) or np.nan) if pd.notna(feat.get("home_pace_10")) and pd.notna(feat.get("away_pace_10")) else np.nan
    feat["rest_advantage"] = (feat.get("home_rest_days", np.nan) or np.nan) - (feat.get("away_rest_days", np.nan) or np.nan) if pd.notna(feat.get("home_rest_days")) and pd.notna(feat.get("away_rest_days")) else np.nan

    # Opponent-specific stats (simplified — use NaN, model handles via imputation)
    for side in ["home", "away"]:
        for suffix in ["w_l_opp_last_2", "w_l_opp_last_10",
                       "pts_opp_last_2", "pts_opp_last_10",
                       "allow_opp_last_2", "allow_opp_last_10"]:
            if f"{side}_{suffix}" not in feat:
                feat[f"{side}_{suffix}"] = np.nan

    # Injury/rust features (simplified — would need live roster check)
    for side in ["home", "away"]:
        for suffix in ["absent_impact", "absent_count", "rust_impact", "rusty_count"]:
            feat[f"{side}_{suffix}"] = np.nan  # Will be imputed

    # Coach features (simplified)
    feat["home_coach_vs_opp_wl"] = np.nan
    feat["away_coach_vs_opp_wl"] = np.nan
    feat["home_coach_vs_opp_n"] = np.nan
    feat["away_coach_vs_opp_n"] = np.nan

    # Physical features (simplified)
    for side in ["home", "away"]:
        for suffix in ["avg_age", "avg_height", "avg_weight", "players_used"]:
            feat[f"{side}_{suffix}"] = np.nan

    # Conference features
    feat["is_interconf"] = int(TEAM_CONFERENCE.get(home, 0) != TEAM_CONFERENCE.get(away, 0))
    feat["east_vs_west_wl_100"] = np.nan
    feat["home_interconf_wl_20"] = np.nan
    feat["away_interconf_wl_20"] = np.nan

    return feat


def confidence_label(prob):
    """Convert probability to human-readable confidence."""
    diff = abs(prob - 0.5)
    if diff < 0.05:
        return "Coin Flip"
    elif diff < 0.10:
        return "Slight Edge"
    elif diff < 0.15:
        return "Moderate Edge"
    elif diff < 0.20:
        return "Confident"
    else:
        return "Lock"


def main():
    os.makedirs(PRED_DIR, exist_ok=True)

    # Date
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

    version = get_model_version()

    print("=" * 60)
    print(f"LinsanityML — Daily Predictions")
    print(f"  Date:  {date_str}")
    print(f"  Model: {version['name']}")
    print("=" * 60)

    # Load model
    model, feature_cols, model_name = load_model()
    print(f"  Model type: {model_name}")
    print(f"  Features: {len(feature_cols)}")

    # Fetch games
    games = fetch_todays_games(date_str)
    if not games:
        print("\n  No games found for this date.")
        return

    print(f"\n  Found {len(games)} games")

    # Fetch spreads
    spreads = fetch_spreads(date_str)
    print(f"  Spreads available: {len(spreads)}")

    # Fetch league game log for feature generation
    print("\n  Fetching season game log for feature computation...")
    try:
        log = LeagueGameLog(season="2024-25")
        all_games = log.get_data_frames()[0]
        all_games["GAME_DATE"] = pd.to_datetime(all_games["GAME_DATE"])
        print(f"  {len(all_games)} team-game records loaded")
    except Exception as e:
        print(f"  Error loading game log: {e}")
        all_games = pd.DataFrame()

    # Generate predictions
    predictions = []

    for game in games:
        home = game["home_team"]
        away = game["away_team"]
        spread = spreads.get((home, away))

        if spread is None:
            print(f"\n  {away} @ {home} — No spread available, skipping")
            continue

        # Build features
        feat = build_game_features(home, away, spread, date_str, all_games)

        # Create DataFrame with correct columns
        feat_df = pd.DataFrame([feat])

        # Ensure all model columns exist
        for col in feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = np.nan

        feat_df = feat_df[feature_cols]

        # Predict
        prob = model.predict_proba(feat_df)[:, 1][0]
        pred = int(prob >= 0.5)
        conf = confidence_label(prob)

        prediction = {
            "date": date_str,
            "away_team": away,
            "home_team": home,
            "spread": spread,
            "prob_covered": round(prob, 4),
            "prediction": "COVERS" if pred == 1 else "DOESN'T COVER",
            "confidence": conf,
            "model_version": version["version"],
        }
        predictions.append(prediction)

    # Display results
    print(f"\n{'='*60}")
    print(f"PREDICTIONS — {date_str}")
    print(f"{'='*60}")

    if not predictions:
        print("  No predictions generated (no spreads available)")
        return

    for p in predictions:
        emoji = "🟢" if p["prediction"] == "COVERS" else "🔴"
        print(f"\n  {p['away_team']} @ {p['home_team']}")
        print(f"    Spread (away): {p['spread']:+.1f}")
        print(f"    Prediction: {emoji} Away {p['prediction']} ({p['confidence']})")
        print(f"    Probability: {p['prob_covered']:.1%}")

    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_path = os.path.join(PRED_DIR, f"{date_str}.csv")
    pred_df.to_csv(pred_path, index=False)

    # Append to history
    history_path = os.path.join(PRED_DIR, "history.csv")
    if os.path.exists(history_path):
        history = pd.read_csv(history_path)
        history = pd.concat([history, pred_df], ignore_index=True)
    else:
        history = pred_df
    history.to_csv(history_path, index=False)

    print(f"\n  Saved: {pred_path}")
    print(f"  History: {history_path}")
    print(f"  Total historical predictions: {len(history)}")
    print(f"{'='*60}")

    return predictions


if __name__ == "__main__":
    main()
