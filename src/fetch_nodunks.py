"""
LinsanityML — No Dunks Pick 'Em Extractor

Fetches the latest No Dunks YouTube livestream, finds the Pick 'Em
timestamp from the video description, takes a screenshot of the
Pick 'Em graphic, and extracts picks via image analysis.

No Dunks Pick 'Em format:
  - One NBA game per weekday show (FanDuel sponsored)
  - Three hosts: Skeets, Tas Melas, Trey Kerby
  - Graphic shows: game, spread, each host's pick (team logo), monthly records
  - Timestamp in video description: e.g. "1:05:10 Tue. Pick 'Em | DAL @ PHX (-7.5)"

Usage:
  python fetch_nodunks.py                    # Latest show (description parse only)
  python fetch_nodunks.py --video <id>       # Specific video
  python fetch_nodunks.py --screenshot       # Also take screenshot (needs Chromium)

Dependencies:
  pip install yt-dlp scrapetube
  For screenshots: chromium-browser
"""

import os
import sys
import json
import re
import subprocess
from datetime import datetime, timezone

import scrapetube

# ─── Config ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
NODUNKS_DIR = os.path.join(DATA_DIR, "nodunks")
CHANNEL_URL = "https://www.youtube.com/@NoDunksInc"

# Team abbreviation mapping for description parsing
TEAM_ABBREVS = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET",
    "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
    "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS",
    "TOR", "UTA", "WAS",
}


def get_latest_stream_id():
    """Get the most recent No Dunks livestream video ID and title."""
    print("  Scanning No Dunks channel for latest livestream...")
    videos = scrapetube.get_channel(
        channel_url=CHANNEL_URL,
        limit=5,
        content_type="streams",
    )
    for v in videos:
        title = v["title"]["runs"][0]["text"]
        vid_id = v["videoId"]
        return vid_id, title
    return None, None


def fetch_video_info(video_id):
    """Fetch video description and metadata via yt-dlp."""
    print(f"  Fetching video info for {video_id}...")
    try:
        import yt_dlp
        ydl_opts = {"skip_download": True, "quiet": True, "compat_opts": set()}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}",
                download=False,
            )
            return {
                "title": info.get("title", ""),
                "description": info.get("description", ""),
                "duration": info.get("duration", 0),
                "upload_date": info.get("upload_date", ""),
            }
    except Exception as e:
        print(f"  Error fetching video info: {e}")
        return None


def parse_pickem_from_description(description):
    """
    Parse the Pick 'Em line from the video description.

    Expected format: "1:05:10 Tue. Pick 'Em | DAL @ PHX (-7.5)"
    Returns dict with timestamp, teams, spread, or None if not found.
    """
    # Match patterns like:
    # "1:05:10 Tue. Pick 'Em | DAL @ PHX (-7.5)"
    # "45:30 Wed. Pick 'Em | BOS @ MIA (-3.5)"
    pattern = r"(\d+:\d+(?::\d+)?)\s+\w+\.?\s+Pick\s+['\u2019]Em\s*\|\s*(\w+)\s*@\s*(\w+)\s*\(([+-]?\d+\.?\d*)\)"

    match = re.search(pattern, description)
    if not match:
        # Try looser pattern
        pattern2 = r"(\d+:\d+(?::\d+)?)\s+.*?Pick.*?Em.*?\|\s*(\w+)\s*@\s*(\w+)\s*\(([+-]?\d+\.?\d*)\)"
        match = re.search(pattern2, description, re.IGNORECASE)

    if not match:
        print("  ⚠️  No Pick 'Em found in video description")
        return None

    timestamp_str = match.group(1)
    away_team = match.group(2).upper()
    home_team = match.group(3).upper()
    spread = float(match.group(4))

    # Parse timestamp to seconds
    parts = timestamp_str.split(":")
    if len(parts) == 3:
        seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        seconds = int(parts[0]) * 60 + int(parts[1])
    else:
        seconds = 0

    # Validate teams
    if away_team not in TEAM_ABBREVS or home_team not in TEAM_ABBREVS:
        print(f"  ⚠️  Unrecognized team: {away_team} or {home_team}")
        return None

    result = {
        "timestamp_str": timestamp_str,
        "timestamp_seconds": seconds,
        "away_team": away_team,
        "home_team": home_team,
        "spread": spread,
    }

    print(f"  ✅ Found Pick 'Em: {away_team} @ {home_team} ({spread:+.1f})")
    print(f"     Timestamp: {timestamp_str} ({seconds}s)")

    return result


def take_screenshots(video_id, timestamp_seconds, output_dir, date_str):
    """
    Extract video frames around the Pick 'Em timestamp using yt-dlp + ffmpeg.

    The graphic doesn't appear instantly — the producer (JD) brings it up
    after the hosts start discussing picks. We capture several frames across
    a ~90 second window to reliably catch the graphic.

    Approach: yt-dlp downloads a short segment, ffmpeg extracts frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  ⚠️  ffmpeg not found — install with: sudo apt install ffmpeg")
        return []

    # Offsets from the segment timestamp (in seconds)
    # Graphic typically appears 15-60s after the segment starts
    offsets = [15, 30, 45, 60, 90]

    url = f"https://www.youtube.com/watch?v={video_id}"
    screenshots = []

    # Get the high-quality stream URL once (reuse for all frames)
    print("  Getting video stream URL...")
    env = os.environ.copy()
    deno_path = os.path.expanduser("~/.deno/bin")
    if os.path.isdir(deno_path):
        env["PATH"] = deno_path + ":" + env.get("PATH", "")

    try:
        stream_urls = subprocess.check_output(
            [
                "yt-dlp",
                "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
                "--remote-components", "ejs:github",
                "--get-url", url,
            ],
            text=True,
            timeout=60,
            env=env,
        ).strip()
        # May return two URLs (video+audio), take first (video)
        stream_url = stream_urls.split("\n")[0]
        print(f"  Stream URL obtained ({len(stream_url)} chars)")
    except Exception as e:
        print(f"  ⚠️  Could not get stream URL: {e}")
        return []

    for offset in offsets:
        seek_time = timestamp_seconds + offset
        mins = seek_time // 60
        secs = seek_time % 60
        label = f"t+{offset}s ({mins}m{secs:02d}s)"

        filename = f"{date_str}_pickem_{offset:03d}s.jpg"
        output_path = os.path.join(output_dir, filename)

        # Format seek time as HH:MM:SS
        h = seek_time // 3600
        m = (seek_time % 3600) // 60
        s = seek_time % 60
        seek_str = f"{h:02d}:{m:02d}:{s:02d}"

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", seek_str,
                    "-i", stream_url,
                    "-frames:v", "1",
                    "-q:v", "1",  # Highest JPEG quality
                    output_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if os.path.exists(output_path) and os.path.getsize(output_path) > 5000:
                size = os.path.getsize(output_path)
                print(f"  ✅ {label}: {os.path.basename(output_path)} ({size:,} bytes)")
                screenshots.append(output_path)
            else:
                print(f"  ⚠️  {label}: frame extraction failed")

        except subprocess.TimeoutExpired:
            print(f"  ⚠️  {label}: timed out")
        except Exception as e:
            print(f"  ⚠️  {label}: {e}")

    print(f"\n  Captured {len(screenshots)}/{len(offsets)} frames")
    return screenshots


def save_pickem(video_id, title, date_str, pickem_info, picks=None, records=None):
    """Save extracted Pick 'Em data."""
    os.makedirs(NODUNKS_DIR, exist_ok=True)

    data = {
        "video_id": video_id,
        "video_title": title,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "date": date_str,
        "away_team": pickem_info.get("away_team"),
        "home_team": pickem_info.get("home_team"),
        "spread": pickem_info.get("spread"),
        "timestamp": pickem_info.get("timestamp_str"),
        "picks": picks or {},
        "records": records or {},
        "source": "description" if not picks else "screenshot",
        "extracted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    path = os.path.join(NODUNKS_DIR, f"{date_str}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")
    return data


def main():
    import argparse
    parser = argparse.ArgumentParser(description="No Dunks Pick 'Em Extractor")
    parser.add_argument("--video", type=str, default=None, help="YouTube video ID")
    parser.add_argument("--screenshot", action="store_true", help="Take screenshot of Pick 'Em graphic")
    parser.add_argument("--date", type=str, default=None, help="Date override (YYYY-MM-DD)")
    args = parser.parse_args()

    os.makedirs(NODUNKS_DIR, exist_ok=True)

    # Get video
    if args.video:
        video_id = args.video
        title = "Manual lookup"
    else:
        video_id, title = get_latest_stream_id()
        if not video_id:
            print("  No recent streams found!")
            return None

    print(f"\n{'='*60}")
    print(f"No Dunks Pick 'Em Extractor")
    print(f"  Video: {title}")
    print(f"  ID:    {video_id}")
    print(f"  URL:   https://www.youtube.com/watch?v={video_id}")
    print(f"{'='*60}")

    # Fetch video info (description)
    info = fetch_video_info(video_id)
    if not info:
        print("  Failed to fetch video info")
        return None

    if args.video == None:
        title = info["title"]

    print(f"  Title: {info['title']}")
    print(f"  Duration: {info['duration']//60}m {info['duration']%60}s")

    # Parse Pick 'Em from description
    pickem = parse_pickem_from_description(info["description"])
    if not pickem:
        print("\n  No Pick 'Em in this episode (might be a special/weekend show)")
        return None

    date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Take screenshots if requested
    screenshots = []
    if args.screenshot:
        screenshots = take_screenshots(
            video_id, pickem["timestamp_seconds"], NODUNKS_DIR, date_str
        )
        if screenshots:
            print(f"\n  📸 {len(screenshots)} screenshots ready for pick extraction")
            print(f"  Use image analysis to read picks from the clearest one")

    # Save what we have (picks will be filled in after screenshot analysis)
    data = save_pickem(video_id, title, date_str, pickem)
    if screenshots:
        data["screenshots"] = screenshots
        # Re-save with screenshot paths
        path = os.path.join(NODUNKS_DIR, f"{date_str}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  ✅ Pick 'Em info extracted from description:")
    print(f"     Game: {pickem['away_team']} @ {pickem['home_team']}")
    print(f"     Spread: {pickem['spread']:+.1f}")
    print(f"     Timestamp: {pickem['timestamp_str']}")
    if screenshots:
        print(f"     Screenshots: {len(screenshots)} captured")
        for s in screenshots:
            print(f"       → {os.path.basename(s)}")
    else:
        print(f"\n  ℹ️  Run with --screenshot to capture the Pick 'Em graphic")
        print(f"     Then use image analysis to extract host picks")
    print(f"{'='*60}")

    return data


if __name__ == "__main__":
    main()
