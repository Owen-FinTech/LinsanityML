"""
LinsanityML — X/Twitter Posting Module

Posts daily predictions to @RaspberryPiMsgs (LinsanityML).
Supports draft mode (preview only) and live posting.

Usage:
  python post_tweet.py                    # Draft today's tweet
  python post_tweet.py --post             # Post today's tweet
  python post_tweet.py --date 2026-02-15  # Specific date
  python post_tweet.py --post --date 2026-02-15

Fan project disclaimer included in every tweet.
Promotes No Dunks podcast while making non-affiliation clear.
"""

import os
import sys
import json
import csv
from datetime import datetime

import tweepy

# ─── Load .env ────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ─── Config ───────────────────────────────────────────────────────────
PRED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "predictions")
LEADERBOARD_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "leaderboard.json")
TWEET_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tweet_log.csv")

# Team emoji/nickname mapping for fun tweets
TEAM_EMOJI = {
    "ATL": "🦅", "BOS": "☘️", "BKN": "🔲", "CHA": "🐝", "CHI": "🐂",
    "CLE": "⚔️", "DAL": "🐴", "DEN": "⛏️", "DET": "🔧", "GSW": "🌉",
    "HOU": "🚀", "IND": "🏎️", "LAC": "⛵", "LAL": "👑", "MEM": "🐻",
    "MIA": "🔥", "MIL": "🦌", "MIN": "🐺", "NOP": "⚜️", "NYK": "🗽",
    "OKC": "⚡", "ORL": "✨", "PHI": "🔔", "PHX": "☀️", "POR": "🌹",
    "SAC": "👑", "SAS": "🖤", "TOR": "🦖", "UTA": "🎵", "WAS": "🧙",
}


def get_x_client():
    """Create authenticated X/Twitter client."""
    return tweepy.Client(
        consumer_key=os.environ["X_API_KEY"],
        consumer_secret=os.environ["X_API_KEY_SECRET"],
        access_token=os.environ["X_ACCESS_TOKEN"],
        access_token_secret=os.environ["X_ACCESS_TOKEN_SECRET"],
    )


def load_predictions(date_str):
    """Load predictions for a given date."""
    pred_path = os.path.join(PRED_DIR, f"{date_str}.csv")
    if not os.path.exists(pred_path):
        return None

    preds = []
    with open(pred_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds.append(row)
    return preds


def load_leaderboard():
    """Load the running leaderboard (LinsanityML vs No Dunks)."""
    if os.path.exists(LEADERBOARD_PATH):
        with open(LEADERBOARD_PATH) as f:
            return json.load(f)
    return {
        "linsanity": {"wins": 0, "losses": 0, "pushes": 0},
        "nodunks": {"wins": 0, "losses": 0, "pushes": 0},
        "season": "2024-25",
        "started": datetime.utcnow().strftime("%Y-%m-%d"),
    }


def save_leaderboard(board):
    """Save the leaderboard."""
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(board, f, indent=2)


def format_spread(spread_val):
    """Format spread for display: +3.5 or -3.5."""
    spread = float(spread_val)
    if spread > 0:
        return f"+{spread:.1f}"
    return f"{spread:.1f}"


def format_prediction_line(pred, compact=False):
    """Format a single game prediction for the tweet."""
    away = pred["away_team"]
    home = pred["home_team"]
    spread = format_spread(pred["spread"])
    prob = float(pred["prob_covered"])
    conf = pred["confidence"]

    covers = pred["prediction"] == "COVERS"
    pick_emoji = "✅" if covers else "❌"

    if compact:
        # One-line format for multi-game days
        pick_word = "Cover" if covers else "No"
        return f"{pick_emoji} {away} ({spread}) @ {home} — {pick_word} {prob:.0%}"
    else:
        away_emoji = TEAM_EMOJI.get(away, "")
        home_emoji = TEAM_EMOJI.get(home, "")
        return f"{away_emoji} {away} ({spread}) @ {home_emoji} {home}\n{pick_emoji} {'Covers' if covers else 'No cover'} • {conf} ({prob:.0%})"


def build_tweet(predictions, date_str, leaderboard=None, nodunks_pick=None):
    """
    Build the tweet text.

    For single-game days (No Dunks Pick 'Em style):
      - Feature the one game prominently
      - Include No Dunks pick if available

    For multi-game days:
      - List all predictions concisely
    """
    date_display = datetime.strptime(date_str, "%Y-%m-%d").strftime("%b %d")

    lines = []
    compact = len(predictions) > 2  # Use compact format for 3+ games

    # Get model version for branding
    from daily_predict import get_model_version
    version = get_model_version()
    ver = version.get("version", "v1.0")

    lines.append(f"🏀 LinsanityML {ver} {'Pick' if len(predictions) == 1 else 'Picks'} — {date_display} 🤖")
    lines.append("")

    for pred in predictions:
        lines.append(format_prediction_line(pred, compact=compact))
    lines.append("")

    # No Dunks pick (if we have it)
    if nodunks_pick:
        lines.append(f"🎙️ @NoDunksInc picks: {nodunks_pick}")
        lines.append("")

    # Leaderboard (if tracking has started)
    if leaderboard:
        lml = leaderboard["linsanity"]
        nd = leaderboard["nodunks"]
        lml_record = f"{lml['wins']}-{lml['losses']}"
        nd_record = f"{nd['wins']}-{nd['losses']}"
        if lml["wins"] + lml["losses"] > 0 or nd["wins"] + nd["losses"] > 0:
            lines.append(f"📊 Season: LinsanityML {lml_record} | No Dunks {nd_record}")
            lines.append("")

    # Disclaimer + promo
    lines.append("Fan project, not affiliated with @NoDunksInc — just love the show! 🎧")

    tweet = "\n".join(lines).strip()
    return tweet


def log_tweet(date_str, tweet_text, tweet_id=None, status="draft"):
    """Log tweet to CSV for history tracking."""
    os.makedirs(os.path.dirname(TWEET_LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(TWEET_LOG_PATH)

    with open(TWEET_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "timestamp", "status", "tweet_id", "tweet_text"])
        writer.writerow([
            date_str,
            datetime.utcnow().isoformat(),
            status,
            tweet_id or "",
            tweet_text,
        ])


def post_tweet(tweet_text, dry_run=True):
    """Post tweet to X. Returns tweet ID if posted, None if dry run."""
    if dry_run:
        print("\n📝 DRAFT MODE — Tweet preview:")
        print("─" * 40)
        print(tweet_text)
        print("─" * 40)
        print(f"Characters: {len(tweet_text)}/280")
        if len(tweet_text) > 280:
            print("⚠️  OVER 280 CHARS — needs trimming!")
        return None

    client = get_x_client()
    response = client.create_tweet(text=tweet_text)
    tweet_id = response.data["id"]
    print(f"\n✅ Tweet posted! ID: {tweet_id}")
    print(f"   https://x.com/RaspberryPiMsgs/status/{tweet_id}")
    return tweet_id


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LinsanityML Tweet Poster")
    parser.add_argument("--post", action="store_true", help="Actually post (default: draft only)")
    parser.add_argument("--date", type=str, default=None, help="Date (YYYY-MM-DD)")
    parser.add_argument("--nodunks", type=str, default=None, help="No Dunks pick text")
    args = parser.parse_args()

    date_str = args.date or datetime.utcnow().strftime("%Y-%m-%d")

    print("=" * 60)
    print(f"LinsanityML — Tweet Builder")
    print(f"  Date: {date_str}")
    print(f"  Mode: {'🔴 LIVE POST' if args.post else '📝 Draft'}")
    print("=" * 60)

    # Load predictions
    predictions = load_predictions(date_str)
    if not predictions:
        print(f"\n  No predictions found for {date_str}")
        print(f"  Run daily_predict.py first!")
        return

    print(f"  Found {len(predictions)} predictions")

    # Load leaderboard
    leaderboard = load_leaderboard()

    # Build tweet
    tweet = build_tweet(predictions, date_str, leaderboard, args.nodunks)

    # Post or preview
    dry_run = not args.post
    tweet_id = post_tweet(tweet, dry_run=dry_run)

    # Log it
    status = "posted" if tweet_id else "draft"
    log_tweet(date_str, tweet, tweet_id, status)

    if dry_run:
        print("\n  To post for real: python post_tweet.py --post")


if __name__ == "__main__":
    main()
