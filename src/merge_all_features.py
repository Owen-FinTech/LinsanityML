"""
Final merge: combine all feature sources into one training-ready dataset.

Sources:
  1. features.csv — core features (rolling stats, rest, streaks, pace, altitude, weather)
  2. injury_features.csv — absent player impact, rust curves
  3. coach_features.csv — coach career record vs opponent
  4. conf_features.csv — inter-conference features
  5. game_details.csv — national TV, attendance, tip-off time (if available)
  6. rosters_enriched.csv + player_game_logs.csv — team avg age/height/weight, international count
  7. spreads.csv — FanDuel spread + target variable

Output: features_with_spreads.csv (training-ready)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from arena_coordinates import TEAM_ID_TO_ABBREV, TEAM_NAME_TO_ABBREV


def build_team_physical_features(data_dir):
    """
    Compute per-team-game physical features:
      - avg age, height (cm), weight (kg) of players who played
      - international player count among those who played
    
    Uses player_game_logs + rosters_enriched.
    """
    plogs_path = os.path.join(data_dir, "player_game_logs.csv")
    roster_path = os.path.join(data_dir, "rosters_enriched.csv")

    if not os.path.exists(roster_path):
        # Try basic rosters without country data
        roster_path = os.path.join(data_dir, "rosters.csv")
        if not os.path.exists(roster_path):
            print("  No roster data available, skipping physical features")
            return None

    plogs = pd.read_csv(plogs_path)
    rosters = pd.read_csv(roster_path)

    # Merge roster physical data onto player game logs
    # Match by player_id + season
    roster_cols = ["PLAYER_ID", "SEASON", "AGE", "HEIGHT_CM", "WEIGHT_KG"]
    if "IS_INTERNATIONAL" in rosters.columns:
        roster_cols.append("IS_INTERNATIONAL")

    plogs = plogs.merge(
        rosters[roster_cols],
        left_on=["PLAYER_ID", "SEASON_YEAR"],
        right_on=["PLAYER_ID", "SEASON"],
        how="left",
    )

    # Only players who actually played (MIN > 0)
    played = plogs[plogs["MIN"] > 0].copy()

    # Aggregate per team per game
    agg = played.groupby(["GAME_ID", "TEAM_ABBREVIATION"]).agg(
        avg_age=("AGE", "mean"),
        avg_height_cm=("HEIGHT_CM", "mean"),
        avg_weight_kg=("WEIGHT_KG", "mean"),
        intl_count=("IS_INTERNATIONAL", "sum") if "IS_INTERNATIONAL" in played.columns else ("AGE", "count"),
        players_used=("PLAYER_ID", "nunique"),
    ).reset_index()

    if "IS_INTERNATIONAL" not in played.columns:
        agg = agg.drop(columns=["intl_count"], errors="ignore")

    return agg


def merge_game_details(features, data_dir):
    """Merge national TV, attendance, game time features."""
    path = os.path.join(data_dir, "game_details.csv")
    if not os.path.exists(path):
        print("  No game details data, skipping")
        return features

    details = pd.read_csv(path)

    # National TV as binary flag
    details["is_national_tv"] = details["national_tv"].notna().astype(int)

    # Parse game time to minutes since midnight (for tip-off time feature)
    def parse_game_time(gt):
        if pd.isna(gt) or not isinstance(gt, str) or ":" not in gt:
            return np.nan
        try:
            parts = gt.split(":")
            hours = int(parts[0])
            mins = int(parts[1])
            return hours * 60 + mins
        except (ValueError, IndexError):
            return np.nan

    details["g_time_mins"] = details["game_time"].apply(parse_game_time)

    merge_cols = ["game_id", "is_national_tv", "attendance", "g_time_mins"]
    features = features.merge(details[merge_cols], on="game_id", how="left")

    matched = features["is_national_tv"].notna().sum()
    print(f"  Game details merged: {matched}/{len(features)} games")
    return features


def merge_physical_features(features, physical_data, game_pairs):
    """Merge team physical features (age, height, weight, intl count)."""
    if physical_data is None:
        return features

    gp = game_pairs[["GAME_ID"]].copy()
    gp["home_abbrev"] = game_pairs["TEAM_ID_home"].map(TEAM_ID_TO_ABBREV)
    gp["away_abbrev"] = game_pairs["TEAM_ID_away"].map(TEAM_ID_TO_ABBREV)

    # Home team physical stats
    home_phys = physical_data.rename(columns={
        "avg_age": "home_avg_age",
        "avg_height_cm": "home_avg_height",
        "avg_weight_kg": "home_avg_weight",
        "players_used": "home_players_used",
    })
    if "intl_count" in home_phys.columns:
        home_phys = home_phys.rename(columns={"intl_count": "home_intl_count"})

    features = features.merge(
        home_phys,
        left_on=["game_id", "home_team"],
        right_on=["GAME_ID", "TEAM_ABBREVIATION"],
        how="left",
    ).drop(columns=["GAME_ID", "TEAM_ABBREVIATION"], errors="ignore")

    # Away team physical stats
    away_phys = physical_data.rename(columns={
        "avg_age": "away_avg_age",
        "avg_height_cm": "away_avg_height",
        "avg_weight_kg": "away_avg_weight",
        "players_used": "away_players_used",
    })
    if "intl_count" in away_phys.columns:
        away_phys = away_phys.rename(columns={"intl_count": "away_intl_count"})

    features = features.merge(
        away_phys,
        left_on=["game_id", "away_team"],
        right_on=["GAME_ID", "TEAM_ABBREVIATION"],
        how="left",
    ).drop(columns=["GAME_ID", "TEAM_ABBREVIATION"], errors="ignore")

    matched = features["home_avg_age"].notna().sum()
    print(f"  Physical features merged: {matched}/{len(features)} games")
    return features


def merge_spreads(features, data_dir):
    """
    Merge FanDuel spreads and compute target variable.
    Also computes spread-derived features.
    """
    spreads_path = os.path.join(data_dir, "spreads.csv")
    if not os.path.exists(spreads_path):
        print("  ERROR: spreads.csv not found!")
        return features

    spreads = pd.read_csv(spreads_path)
    spreads["game_date"] = pd.to_datetime(spreads["game_date"])

    # Map team names to abbreviations
    spreads["away_abbrev"] = spreads["away_team"].map(TEAM_NAME_TO_ABBREV)
    spreads["home_abbrev"] = spreads["home_team"].map(TEAM_NAME_TO_ABBREV)

    features["game_date"] = pd.to_datetime(features["game_date"])

    # Merge on date + teams (since game_id formats differ)
    merged = features.merge(
        spreads[["game_date", "home_abbrev", "away_abbrev", "f_d_sprd_away", "commence_time"]],
        left_on=["game_date", "home_team", "away_team"],
        right_on=["game_date", "home_abbrev", "away_abbrev"],
        how="inner",
    ).drop(columns=["home_abbrev", "away_abbrev"], errors="ignore")

    # Compute target: did away team cover the spread?
    # away_margin = away_pts - home_pts
    # If away_margin + spread > 0: away covered
    # If away_margin + spread < 0: not covered
    # If away_margin + spread == 0: push (exclude)
    merged["away_margin"] = merged["away_pts"] - merged["home_pts"]
    merged["spread_result"] = merged["away_margin"] + merged["f_d_sprd_away"]

    # Exclude pushes
    before = len(merged)
    merged = merged[merged["spread_result"] != 0].copy()
    pushes = before - len(merged)
    merged["sprd_cvrd"] = (merged["spread_result"] > 0).astype(int)

    # Clean up
    merged = merged.drop(columns=["away_margin", "spread_result"], errors="ignore")

    print(f"  Spreads merged: {len(merged)} games (excluded {pushes} pushes)")
    print(f"  Target rate: {merged['sprd_cvrd'].mean():.3f}")

    return merged


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    print("=" * 60)
    print("LinsanityML — Final Feature Merge")
    print("=" * 60)

    # 1. Start with core features (already includes weather, injury, coach, conf)
    print("\n1. Loading core features...")
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    features["game_date"] = pd.to_datetime(features["game_date"])
    print(f"  {len(features)} games, {features.shape[1]} columns")

    # 2. Load game pairs for helper merges
    game_pairs = pd.read_csv(os.path.join(data_dir, "game_pairs.csv"))

    # 3. Game details (national TV, attendance, game time)
    print("\n2. Merging game details...")
    features = merge_game_details(features, data_dir)

    # 4. Physical features (age, height, weight)
    print("\n3. Building team physical features...")
    physical = build_team_physical_features(data_dir)
    features = merge_physical_features(features, physical, game_pairs)

    # 5. Merge spreads + compute target
    print("\n4. Merging spreads + computing target...")
    final = merge_spreads(features, data_dir)

    # Save
    output_path = os.path.join(data_dir, "features_with_spreads.csv")
    final.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("FINAL DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Shape: {final.shape}")
    print(f"Columns: {final.shape[1]}")
    print(f"\nNull counts (top 15):")
    nulls = final.isnull().sum().sort_values(ascending=False)
    print(nulls[nulls > 0].head(15).to_string())
    print(f"\nTarget: sprd_cvrd")
    print(f"  Covered:     {(final['sprd_cvrd']==1).sum()} ({final['sprd_cvrd'].mean():.3f})")
    print(f"  Not covered: {(final['sprd_cvrd']==0).sum()} ({1-final['sprd_cvrd'].mean():.3f})")
    print(f"\nSeasons:")
    print(final.groupby("season").size().to_string())
    print(f"\nSaved: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
