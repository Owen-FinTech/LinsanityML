"""
NBA Arena coordinates (latitude, longitude) for distance calculations.
Updated for 2024-25 season.
"""

ARENA_COORDS = {
    # Team abbreviation: (latitude, longitude)
    "ATL": (33.7573, -84.3963),    # State Farm Arena, Atlanta
    "BOS": (42.3662, -71.0621),    # TD Garden, Boston
    "BKN": (40.6826, -73.9754),    # Barclays Center, Brooklyn
    "CHA": (35.2251, -80.8392),    # Spectrum Center, Charlotte
    "CHI": (41.8807, -87.6742),    # United Center, Chicago
    "CLE": (41.4964, -81.6882),    # Rocket Mortgage FieldHouse, Cleveland
    "DAL": (32.7905, -96.8103),    # American Airlines Center, Dallas
    "DEN": (39.7487, -105.0077),   # Ball Arena, Denver
    "DET": (42.3410, -83.0551),    # Little Caesars Arena, Detroit
    "GSW": (37.7680, -122.3877),   # Chase Center, San Francisco
    "HOU": (29.7508, -95.3621),    # Toyota Center, Houston
    "IND": (39.7640, -86.1555),    # Gainbridge Fieldhouse, Indianapolis
    "LAC": (33.4264, -118.2618),   # Intuit Dome, Inglewood
    "LAL": (34.0430, -118.2673),   # Crypto.com Arena, Los Angeles
    "MEM": (35.1382, -90.0506),    # FedExForum, Memphis
    "MIA": (25.7814, -80.1870),    # Kaseya Center, Miami
    "MIL": (43.0451, -87.9174),    # Fiserv Forum, Milwaukee
    "MIN": (44.9795, -93.2761),    # Target Center, Minneapolis
    "NOP": (29.9490, -90.0821),    # Smoothie King Center, New Orleans
    "NYK": (40.7505, -73.9934),    # Madison Square Garden, New York
    "OKC": (35.4634, -97.5151),    # Paycom Center, Oklahoma City
    "ORL": (28.5392, -81.3839),    # Amway Center, Orlando
    "PHI": (39.9012, -75.1720),    # Wells Fargo Center, Philadelphia
    "PHX": (33.4457, -112.0712),   # Footprint Center, Phoenix
    "POR": (45.5316, -122.6668),   # Moda Center, Portland
    "SAC": (38.5802, -121.4997),   # Golden 1 Center, Sacramento
    "SAS": (29.4270, -98.4375),    # Frost Bank Center, San Antonio
    "TOR": (43.6435, -79.3791),    # Scotiabank Arena, Toronto
    "UTA": (40.7683, -111.9011),   # Delta Center, Salt Lake City
    "WAS": (38.8981, -77.0209),    # Capital One Arena, Washington DC
}

# Team name to abbreviation mapping
TEAM_NAME_TO_ABBREV = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# Team ID (nba_api) to abbreviation mapping
TEAM_ID_TO_ABBREV = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612751: "BKN",
    1610612766: "CHA",
    1610612741: "CHI",
    1610612739: "CLE",
    1610612742: "DAL",
    1610612743: "DEN",
    1610612765: "DET",
    1610612744: "GSW",
    1610612745: "HOU",
    1610612754: "IND",
    1610612746: "LAC",
    1610612747: "LAL",
    1610612763: "MEM",
    1610612748: "MIA",
    1610612749: "MIL",
    1610612750: "MIN",
    1610612740: "NOP",
    1610612752: "NYK",
    1610612760: "OKC",
    1610612753: "ORL",
    1610612755: "PHI",
    1610612756: "PHX",
    1610612757: "POR",
    1610612758: "SAC",
    1610612759: "SAS",
    1610612761: "TOR",
    1610612762: "UTA",
    1610612764: "WAS",
}

# Alphabetical index (1-30) for team feature
TEAM_ALPHA_INDEX = {
    "ATL": 1,  "BOS": 2,  "BKN": 3,  "CHA": 4,  "CHI": 5,
    "CLE": 6,  "DAL": 7,  "DEN": 8,  "DET": 9,  "GSW": 10,
    "HOU": 11, "IND": 12, "LAC": 13, "LAL": 14, "MEM": 15,
    "MIA": 16, "MIL": 17, "MIN": 18, "NOP": 19, "NYK": 20,
    "OKC": 21, "ORL": 22, "PHI": 23, "PHX": 24, "POR": 25,
    "SAC": 26, "SAS": 27, "TOR": 28, "UTA": 29, "WAS": 30,
}

# Arena altitude in metres (above sea level)
ARENA_ALTITUDE_M = {
    "ATL": 320, "BOS": 6, "BKN": 10, "CHA": 229, "CHI": 181,
    "CLE": 199, "DAL": 131, "DEN": 1609, "DET": 183, "GSW": 3,
    "HOU": 15, "IND": 218, "LAC": 30, "LAL": 87, "MEM": 103,
    "MIA": 2, "MIL": 188, "MIN": 253, "NOP": 1, "NYK": 10,
    "OKC": 366, "ORL": 25, "PHI": 5, "PHX": 331, "POR": 15,
    "SAC": 9, "SAS": 198, "TOR": 76, "UTA": 1288, "WAS": 22,
}

# Conference mapping (0 = East, 1 = West)
TEAM_CONFERENCE = {
    "ATL": 0, "BOS": 0, "BKN": 0, "CHA": 0, "CHI": 0,
    "CLE": 0, "DET": 0, "IND": 0, "MIA": 0, "MIL": 0,
    "NYK": 0, "ORL": 0, "PHI": 0, "TOR": 0, "WAS": 0,
    "DAL": 1, "DEN": 1, "GSW": 1, "HOU": 1, "LAC": 1,
    "LAL": 1, "MEM": 1, "MIN": 1, "NOP": 1, "OKC": 1,
    "PHX": 1, "POR": 1, "SAC": 1, "SAS": 1, "UTA": 1,
}
