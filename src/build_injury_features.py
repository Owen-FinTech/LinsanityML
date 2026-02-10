"""
Build injury/absence impact and rust features from player box scores.

Key insight: PlayerGameLogs only includes players who PLAYED. To find absences,
we compare each game's participants against the team's active roster (defined as
players who played in games surrounding this one).

For each game, computes:
  1. absent_impact_home/away — sum of (baseline_SPER × baseline_MIN) for absent players
  2. rust_impact_home/away — expected deficit from recently-returned players
  3. absent_count_home/away — number of absent roster players
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from arena_coordinates import TEAM_ID_TO_ABBREV


def build_roster_windows(plogs):
    """
    Define the 'active roster' for each team-game as players who played in at least
    5 of the surrounding 20 games (10 before, 10 after). This captures players who
    are genuinely on the active roster vs those traded/in G-league/etc.
    """
    plogs = plogs.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

    # Get all unique team-games in order
    team_games = plogs.groupby("TEAM_ABBREVIATION")["GAME_ID"].apply(
        lambda x: x.unique()
    ).to_dict()

    # For each team, get chronological game list
    team_game_order = {}
    for team in plogs["TEAM_ABBREVIATION"].unique():
        tg = plogs[plogs["TEAM_ABBREVIATION"] == team].sort_values("GAME_DATE")
        ordered_games = tg.drop_duplicates("GAME_ID")["GAME_ID"].values
        team_game_order[team] = ordered_games

    return team_game_order


def compute_active_roster(plogs, team_game_order):
    """
    For each team-game, determine which players are on the active roster.
    Active roster = played in ≥5 of surrounding 20 games (10 before, 10 after).
    """
    print("  Computing active rosters per team-game...")

    # Build lookup: team -> game_idx -> set of players who played
    team_game_players = {}
    for team, games in team_game_order.items():
        team_data = plogs[plogs["TEAM_ABBREVIATION"] == team]
        game_player_map = {}
        for gid in games:
            players = team_data[team_data["GAME_ID"] == gid]["PLAYER_ID"].values
            game_player_map[gid] = set(players)
        team_game_players[team] = game_player_map

    # For each team-game, compute roster from surrounding window
    roster_records = []  # (GAME_ID, TEAM, PLAYER_ID, on_roster)
    total = sum(len(g) for g in team_game_order.values())
    processed = 0

    for team, games in team_game_order.items():
        game_list = list(games)
        n = len(game_list)

        for i, gid in enumerate(game_list):
            # Window: 10 games before and after (excluding this game)
            start = max(0, i - 10)
            end = min(n, i + 11)
            window_games = [game_list[j] for j in range(start, end) if j != i]

            # Count appearances in window
            player_counts = {}
            for wgid in window_games:
                for pid in team_game_players[team].get(wgid, set()):
                    player_counts[pid] = player_counts.get(pid, 0) + 1

            # Roster = players who appeared in 5+ of window games
            roster = {pid for pid, cnt in player_counts.items() if cnt >= 5}

            # Who played this game
            played = team_game_players[team].get(gid, set())

            # Absent = on roster but didn't play
            absent = roster - played

            for pid in roster:
                roster_records.append({
                    "GAME_ID": gid,
                    "TEAM": team,
                    "PLAYER_ID": pid,
                    "PLAYED": 1 if pid in played else 0,
                })

            processed += 1
            if processed % 2000 == 0:
                print(f"    {processed}/{total} team-games processed...")

    return pd.DataFrame(roster_records)


def build_player_baselines(plogs):
    """
    Compute rolling baseline SPER and MIN for each player (last 20 games played).
    Returns TWO dicts:
      - game_baselines: (PLAYER_ID, GAME_ID) -> (baseline_sper, baseline_min)
      - latest_baselines: PLAYER_ID -> (baseline_sper, baseline_min)  [most recent]
    """
    print("  Computing player baselines...")
    plogs = plogs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    game_baselines = {}
    latest_baselines = {}
    grouped = plogs.groupby("PLAYER_ID")
    n_players = len(grouped)

    for i, (pid, group) in enumerate(grouped):
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n_players} players...")

        spers = []
        mins = []
        current_baseline = (None, None)

        for _, row in group.iterrows():
            # Baseline BEFORE this game
            if len(spers) >= 3:
                b_sper = np.mean(spers[-20:])
                b_min = np.mean(mins[-20:])
            elif len(spers) > 0:
                b_sper = np.mean(spers)
                b_min = np.mean(mins)
            else:
                b_sper = None
                b_min = None

            if b_sper is not None:
                current_baseline = (b_sper, b_min)

            game_baselines[(pid, row["GAME_ID"])] = current_baseline

            if row["MIN"] > 0 and pd.notna(row["SPER"]):
                spers.append(row["SPER"])
                mins.append(row["MIN"])

        # Store the latest baseline for this player (for games they're absent from)
        latest_baselines[pid] = current_baseline

    return game_baselines, latest_baselines


def build_absence_tracking(roster_data, team_game_order):
    """
    Track consecutive missed games and games-back-from-absence per player,
    using the roster_data which knows about both played AND absent games.
    Returns dict: (PLAYER_ID, GAME_ID) -> (miss_streak_entering, games_back)
    """
    print("  Building absence tracking from roster data...")

    # Sort roster data by team, player, then game order
    # Returns dict: (PLAYER_ID, GAME_ID) -> (miss_streak_that_caused_return, games_back)
    # miss_streak_that_caused_return: the length of the absence they're returning from
    # games_back: how many games since they came back (0 = still absent or no recent absence)
    tracking = {}
    total_players = roster_data.groupby(["PLAYER_ID", "TEAM"]).ngroups
    processed = 0

    team_game_idx = {}
    for team, games in team_game_order.items():
        team_game_idx[team] = {gid: i for i, gid in enumerate(games)}

    for (pid, team), grp in roster_data.groupby(["PLAYER_ID", "TEAM"]):
        processed += 1
        if processed % 500 == 0:
            print(f"    {processed}/{total_players} player-teams tracked...")

        gidx = team_game_idx.get(team, {})
        grp = grp.copy()
        grp["_order"] = grp["GAME_ID"].map(gidx)
        grp = grp.sort_values("_order")

        consec_missed = 0
        games_back = 0
        return_miss_streak = 0  # the absence length they're recovering from

        for _, row in grp.iterrows():
            gid = row["GAME_ID"]
            played = row["PLAYED"] == 1

            # Record state ENTERING this game
            tracking[(pid, gid)] = (return_miss_streak, games_back)

            if played:
                if consec_missed >= 3:
                    # Just returned from significant absence
                    return_miss_streak = consec_missed
                    games_back = 1
                elif return_miss_streak > 0:
                    games_back += 1
                    # Check if recovery period is over
                    if consec_missed <= 5 and games_back > 5:
                        return_miss_streak = 0
                        games_back = 0
                    elif consec_missed <= 15 and games_back > 10:
                        return_miss_streak = 0
                        games_back = 0
                    elif games_back > 18:
                        return_miss_streak = 0
                        games_back = 0
                consec_missed = 0
            else:
                consec_missed += 1
                # If currently absent, reset return tracking
                return_miss_streak = 0
                games_back = 0

    print(f"  {len(tracking)} tracking entries")
    return tracking


def compute_rust_discount(return_miss_streak, games_back):
    """Rust discount. return_miss_streak = how long the absence was they're returning from."""
    if return_miss_streak < 3 or games_back <= 0:
        return 1.0
    if return_miss_streak <= 5:
        drop, recovery = 0.12, 5
    elif return_miss_streak <= 15:
        drop, recovery = 0.20, 10
    else:
        drop, recovery = 0.28, 18
    if games_back >= recovery:
        return 1.0
    return 1.0 - drop * (1.0 - games_back / recovery)


def build_game_features(roster_data, game_baselines, latest_baselines, tracking, game_pairs):
    """Aggregate player-level data into game-level features."""
    print("  Aggregating to game level...")

    gp = game_pairs.copy()
    gp["home_abbrev"] = gp["TEAM_ID_home"].map(TEAM_ID_TO_ABBREV)
    gp["away_abbrev"] = gp["TEAM_ID_away"].map(TEAM_ID_TO_ABBREV)

    records = []
    for idx, game in gp.iterrows():
        gid = game["GAME_ID"]
        feat = {"game_id": gid}

        for side, team in [("home", game["home_abbrev"]), ("away", game["away_abbrev"])]:
            team_roster = roster_data[
                (roster_data["GAME_ID"] == gid) & (roster_data["TEAM"] == team)
            ]

            absent_impact = 0.0
            absent_count = 0
            rust_impact = 0.0
            rusty_count = 0

            for _, pr in team_roster.iterrows():
                pid = pr["PLAYER_ID"]

                # Get baseline: try game-specific first, then latest known
                b = game_baselines.get((pid, gid))
                if b is None or b[0] is None:
                    b = latest_baselines.get(pid, (None, None))
                b_sper = b[0] if b and b[0] is not None else 0
                b_min = b[1] if b and b[1] is not None else 0

                if pr["PLAYED"] == 0:
                    # Absent player
                    absent_impact += b_sper * b_min
                    absent_count += 1
                else:
                    # Check rust for returning players
                    rms, gb = tracking.get((pid, gid), (0, 0))
                    disc = compute_rust_discount(rms, gb)
                    if disc < 1.0:
                        rust_impact += b_sper * b_min * (1.0 - disc)
                        rusty_count += 1

            feat[f"{side}_absent_impact"] = round(absent_impact, 2)
            feat[f"{side}_absent_count"] = absent_count
            feat[f"{side}_rust_impact"] = round(rust_impact, 2)
            feat[f"{side}_rusty_count"] = rusty_count

        records.append(feat)

        if (idx + 1) % 1000 == 0:
            print(f"    {idx+1}/{len(gp)} games aggregated...")

    return pd.DataFrame(records)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    print("Loading player game logs...")
    plogs = pd.read_csv(os.path.join(data_dir, "player_game_logs.csv"))
    plogs["GAME_DATE"] = pd.to_datetime(plogs["GAME_DATE"])
    print(f"  {len(plogs)} records, {plogs['PLAYER_ID'].nunique()} players")

    print("\nBuilding team game schedules...")
    team_game_order = build_roster_windows(plogs)
    total_team_games = sum(len(v) for v in team_game_order.values())
    print(f"  {total_team_games} team-games across {len(team_game_order)} teams")

    print("\nComputing active rosters...")
    roster_data = compute_active_roster(plogs, team_game_order)
    absent_count = (roster_data["PLAYED"] == 0).sum()
    print(f"  Total roster entries: {len(roster_data)}")
    print(f"  Absent player-games: {absent_count}")

    print("\nComputing player baselines...")
    game_baselines, latest_baselines = build_player_baselines(plogs)
    print(f"  {len(game_baselines)} game baseline entries, {len(latest_baselines)} player latest baselines")

    print("\nBuilding absence tracking...")
    tracking = build_absence_tracking(roster_data, team_game_order)

    print("\nLoading game pairs...")
    gp = pd.read_csv(os.path.join(data_dir, "game_pairs.csv"))
    print(f"  {len(gp)} games")

    print("\nBuilding game-level features...")
    features = build_game_features(roster_data, game_baselines, latest_baselines, tracking, gp)

    output_path = os.path.join(data_dir, "injury_features.csv")
    features.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("INJURY/RUST FEATURES SUMMARY")
    print(f"{'='*60}")
    print(f"Games: {len(features)}")
    for side in ["home", "away"]:
        print(f"\n{side.upper()}:")
        print(f"  Absent impact: mean={features[f'{side}_absent_impact'].mean():.2f}, "
              f"max={features[f'{side}_absent_impact'].max():.2f}")
        print(f"  Absent count:  mean={features[f'{side}_absent_count'].mean():.1f}, "
              f"max={features[f'{side}_absent_count'].max()}")
        print(f"  Rust impact:   mean={features[f'{side}_rust_impact'].mean():.2f}, "
              f"max={features[f'{side}_rust_impact'].max():.2f}")
        print(f"  Rusty players: mean={features[f'{side}_rusty_count'].mean():.1f}, "
              f"max={features[f'{side}_rusty_count'].max()}")
    print(f"\nSaved: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
