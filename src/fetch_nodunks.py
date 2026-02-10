"""
LinsanityML — No Dunks Pick 'Em Extractor

Fetches the latest No Dunks YouTube livestream, extracts the Pick 'Em
segment from the transcript, and identifies which game they picked
and who picked what.

No Dunks Pick 'Em format:
  - One NBA game per weekday show (FanDuel sponsored)
  - Three hosts pick sides: Skeets, Tas Melas, Trey Kerby
  - Usually near the end of the show (~last 15 minutes)
  - They reference "Pick 'Em", the spread, and each person's pick

Usage:
  python fetch_nodunks.py              # Latest show
  python fetch_nodunks.py <video_id>   # Specific video

Dependencies:
  pip install youtube-transcript-api scrapetube
"""

import os
import sys
import json
import re
from datetime import datetime

import scrapetube
from youtube_transcript_api import YouTubeTranscriptApi

# ─── Config ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
NODUNKS_DIR = os.path.join(DATA_DIR, "nodunks")
CHANNEL_URL = "https://www.youtube.com/@NoDunksInc"

# Team name variants that might appear in transcripts
TEAM_NAMES = {
    # Full names
    "hawks": "ATL", "celtics": "BOS", "nets": "BKN", "hornets": "CHA",
    "bulls": "CHI", "cavaliers": "CLE", "cavs": "CLE",
    "mavericks": "DAL", "mavs": "DAL", "nuggets": "DEN",
    "pistons": "DET", "warriors": "GSW", "dubs": "GSW",
    "rockets": "HOU", "pacers": "IND", "clippers": "LAC",
    "lakers": "LAL", "grizzlies": "MEM", "grizz": "MEM",
    "heat": "MIA", "bucks": "MIL", "timberwolves": "MIN",
    "wolves": "MIN", "pelicans": "NOP", "pels": "NOP",
    "knicks": "NYK", "thunder": "OKC", "magic": "ORL",
    "sixers": "PHI", "76ers": "PHI", "suns": "PHX",
    "phoenix": "PHX", "blazers": "POR", "trail blazers": "POR",
    "kings": "SAC", "spurs": "SAS", "raptors": "TOR",
    "jazz": "UTA", "wizards": "WAS",
    # City names
    "atlanta": "ATL", "boston": "BOS", "brooklyn": "BKN",
    "charlotte": "CHA", "chicago": "CHI", "cleveland": "CLE",
    "dallas": "DAL", "denver": "DEN", "detroit": "DET",
    "golden state": "GSW", "houston": "HOU", "indiana": "IND",
    "la clippers": "LAC", "la lakers": "LAL", "los angeles lakers": "LAL",
    "memphis": "MEM", "miami": "MIA", "milwaukee": "MIL",
    "minnesota": "MIN", "new orleans": "NOP", "new york": "NYK",
    "oklahoma city": "OKC", "orlando": "ORL", "philadelphia": "PHI",
    "philly": "PHI", "portland": "POR", "sacramento": "SAC",
    "san antonio": "SAS", "toronto": "TOR", "utah": "UTA",
    "washington": "WAS",
}

# Host names and common transcript variants
HOSTS = {
    "skeets": "Skeets",
    "skeats": "Skeets",
    "tas": "Tas",
    "taz": "Tas",
    "trey": "Trey",
    "tray": "Trey",
    "train": "Trey",  # Auto-caption often mishears "Trey" as "Train"
    "tren": "Trey",
}


def get_latest_stream():
    """Get the most recent No Dunks livestream video ID and title."""
    print("  Fetching latest No Dunks livestream...")
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


def fetch_transcript(video_id):
    """Fetch YouTube auto-generated transcript."""
    print(f"  Fetching transcript for {video_id}...")
    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(video_id)
    entries = list(transcript)
    print(f"  Got {len(entries)} transcript entries (~{entries[-1].start/60:.0f} min)")
    return entries


def find_pickem_segment(entries):
    """
    Find the Pick 'Em segment in the transcript.
    Usually in the last ~15 minutes, marked by strong keywords like
    "pick 'em/them tonight", "tonight's game", "favored by X points".
    """
    # Strong keywords that reliably signal the Pick 'Em segment
    # (weak terms like "spread" appear in trade/general discussion)
    strong_keywords = [
        "pick 'em", "pick em", "pickem",
        "pick them tonight", "pick him tonight",
        "tonight's game",
        "favored by",
        "favourite by",
    ]

    duration = entries[-1].start if entries else 0

    # Search the last 15 minutes (Pick 'Em is almost always right at the end)
    search_start = max(0, duration - 15 * 60)
    candidates = [e for e in entries if e.start >= search_start]

    # Find the first strong keyword match
    segment_start = None
    for i, entry in enumerate(candidates):
        text = entry.text.lower()
        if any(kw in text for kw in strong_keywords):
            segment_start = max(0, i - 5)
            break

    if segment_start is None:
        # Fallback: look for "tonight" + team names in proximity
        for i, entry in enumerate(candidates):
            text = entry.text.lower()
            if "tonight" in text and find_teams_in_text(entry.text):
                segment_start = max(0, i - 3)
                break

    if segment_start is None:
        print("  ⚠️  Could not find Pick 'Em segment")
        return None

    # Extract ~5 minutes from the start of the segment
    start_time = candidates[segment_start].start
    segment = [e for e in candidates if start_time <= e.start <= start_time + 300]

    return segment


def extract_pickem_text(segment):
    """Join segment entries into readable text."""
    if not segment:
        return ""
    return " ".join(e.text for e in segment)


def find_teams_in_text(text):
    """Find NBA team references in text."""
    text_lower = text.lower()
    found = set()
    for name, abbrev in TEAM_NAMES.items():
        # Word boundary check to avoid partial matches
        pattern = r'\b' + re.escape(name) + r'\b'
        if re.search(pattern, text_lower):
            found.add(abbrev)
    return list(found)


def parse_picks(segment_text, teams_playing=None):
    """
    Parse the Pick 'Em segment to extract:
    - Which game (teams)
    - Who picked what
    - The spread

    Returns dict with parsed info, or None if can't parse.
    """
    text_lower = segment_text.lower()
    result = {
        "raw_text": segment_text,
        "teams": find_teams_in_text(segment_text),
        "spread": None,
        "picks": {},
        "parsed": False,
    }

    # Try to find the spread (e.g., "favored by 7", "7 and a half", "minus 6.5")
    spread_patterns = [
        r'favou?red by (\d+)\s*(?:and a half)?',
        r'(\d+)\s*(?:and a half)\s*point',
        r'(?:minus|plus)\s*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*point\s*(?:spread|line)',
    ]

    for pattern in spread_patterns:
        match = re.search(pattern, text_lower)
        if match:
            spread_val = float(match.group(1))
            if "and a half" in text_lower[match.start():match.end() + 20]:
                spread_val += 0.5
            result["spread"] = spread_val
            break

    # Try to identify who picked what
    # This is the hardest part — auto-captions are messy
    for host_variant, host_name in HOSTS.items():
        if host_variant in text_lower:
            # Look for team mentions near host name
            for match in re.finditer(re.escape(host_variant), text_lower):
                # Get surrounding context (100 chars each way)
                start = max(0, match.start() - 100)
                end = min(len(text_lower), match.end() + 100)
                context = text_lower[start:end]

                teams_near = find_teams_in_text(context)
                if teams_near and host_name not in result["picks"]:
                    result["picks"][host_name] = teams_near[0]

    if result["teams"] and result["spread"]:
        result["parsed"] = True

    return result


def save_pickem(video_id, title, date_str, result):
    """Save extracted Pick 'Em data."""
    os.makedirs(NODUNKS_DIR, exist_ok=True)

    data = {
        "video_id": video_id,
        "video_title": title,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "date": date_str,
        "teams": result.get("teams", []),
        "spread": result.get("spread"),
        "picks": result.get("picks", {}),
        "parsed": result.get("parsed", False),
        "raw_segment": result.get("raw_text", "")[:2000],  # Truncate for storage
        "extracted_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    path = os.path.join(NODUNKS_DIR, f"{date_str}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")

    return data


def main():
    os.makedirs(NODUNKS_DIR, exist_ok=True)

    # Get video ID
    if len(sys.argv) > 1:
        video_id = sys.argv[1]
        title = "Manual lookup"
    else:
        video_id, title = get_latest_stream()
        if not video_id:
            print("  No recent streams found!")
            return

    print(f"\n{'='*60}")
    print(f"No Dunks Pick 'Em Extractor")
    print(f"  Video: {title}")
    print(f"  ID:    {video_id}")
    print(f"  URL:   https://www.youtube.com/watch?v={video_id}")
    print(f"{'='*60}")

    # Fetch transcript
    transcript = fetch_transcript(video_id)

    # Find Pick 'Em segment
    segment = find_pickem_segment(transcript)
    if not segment:
        print("\n  Could not locate Pick 'Em segment.")
        print("  This might be a special episode without Pick 'Em.")

        # Save raw info anyway
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        save_pickem(video_id, title, date_str, {
            "teams": [],
            "parsed": False,
            "raw_text": "",
        })
        return

    # Extract and parse
    segment_text = extract_pickem_text(segment)
    start_time = segment[0].start
    end_time = segment[-1].start

    print(f"\n  Pick 'Em segment found: {start_time/60:.1f}m - {end_time/60:.1f}m")
    print(f"  Segment text ({len(segment_text)} chars):")
    print(f"  {'─'*50}")
    # Print first 500 chars
    print(f"  {segment_text[:500]}...")
    print(f"  {'─'*50}")

    result = parse_picks(segment_text)

    print(f"\n  Teams mentioned: {result['teams']}")
    print(f"  Spread found: {result['spread']}")
    print(f"  Picks parsed: {result['picks']}")
    print(f"  Successfully parsed: {result['parsed']}")

    # Save
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    data = save_pickem(video_id, title, date_str, result)

    # Summary
    print(f"\n{'='*60}")
    if result["parsed"]:
        print(f"  ✅ Pick 'Em segment extracted!")
        if len(result["teams"]) >= 2:
            print(f"  Teams mentioned: {', '.join(result['teams'])}")
        if result["spread"]:
            print(f"  Spread (auto-detected): {result['spread']}")
        for host, team in result["picks"].items():
            print(f"  {host} picks (auto-detected): {team}")
        print(f"\n  ⚠️  Auto-parsing is approximate — review raw segment")
        print(f"     for accurate picks (auto-captions are messy)")
    else:
        print(f"  ⚠️  Partial extraction — needs manual review")
        print(f"  Raw segment saved for review")
    print(f"{'='*60}")

    return data


if __name__ == "__main__":
    main()
