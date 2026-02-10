"""
Fetch game-level details: national TV broadcast, attendance, game time.
Uses BoxScoreSummaryV2 per game (slow but comprehensive).
Supports resume via checkpoint file.
"""

import os
import time
import pandas as pd
from nba_api.stats.endpoints import BoxScoreSummaryV2

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def fetch_game_detail(game_id):
    """Fetch broadcast, attendance, and game time for one game."""
    try:
        summary = BoxScoreSummaryV2(game_id=game_id)
        dfs = summary.get_data_frames()

        # DataFrame 0: broadcast info
        natl_tv = None
        if len(dfs[0]) > 0:
            natl_tv = dfs[0].iloc[0].get("NATL_TV_BROADCASTER_ABBREVIATION")
            if natl_tv and str(natl_tv).strip().lower() in ("none", "nan", ""):
                natl_tv = None

        # DataFrame 4: attendance and game time
        attendance = None
        game_time_str = None
        if len(dfs) > 4 and len(dfs[4]) > 0:
            att = dfs[4].iloc[0].get("ATTENDANCE")
            if att and str(att).strip() not in ("0", "", "None"):
                try:
                    attendance = int(att)
                except (ValueError, TypeError):
                    pass
            game_time_str = dfs[4].iloc[0].get("GAME_TIME")

        return {
            "national_tv": natl_tv,
            "attendance": attendance,
            "game_time": game_time_str,
        }
    except Exception as e:
        return {"national_tv": None, "attendance": None, "game_time": None, "error": str(e)}


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    print("Loading game IDs...")
    games = pd.read_csv(os.path.join(data_dir, "game_pairs.csv"))
    game_ids = games["GAME_ID"].unique()
    print(f"  {len(game_ids)} games to fetch")

    # Resume support
    output_path = os.path.join(data_dir, "game_details.csv")
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        done_ids = set(existing["game_id"].values)
        records = existing.to_dict("records")
        print(f"  Resuming: {len(done_ids)} already fetched")
    else:
        done_ids = set()
        records = []

    remaining = [gid for gid in game_ids if gid not in done_ids]
    print(f"  Remaining: {len(remaining)}")

    for i, gid in enumerate(remaining):
        detail = fetch_game_detail(gid)
        detail["game_id"] = gid
        records.append(detail)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(remaining)}] Latest: {gid} | TV: {detail.get('national_tv')} | Att: {detail.get('attendance')}")
            # Checkpoint
            pd.DataFrame(records).to_csv(output_path, index=False)

        time.sleep(0.7)

    # Final save
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    # Stats
    has_tv = df["national_tv"].notna().sum()
    has_att = df["attendance"].notna().sum()
    print(f"\n{'='*60}")
    print("GAME DETAILS SUMMARY")
    print(f"{'='*60}")
    print(f"Games: {len(df)}")
    print(f"National TV games: {has_tv} ({has_tv/len(df)*100:.1f}%)")
    if has_att > 0:
        print(f"Games with attendance: {has_att}")
        print(f"Mean attendance: {df['attendance'].mean():.0f}")
    print(f"Games with game time: {df['game_time'].notna().sum()}")
    print(f"\nSaved: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
