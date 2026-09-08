"""Fetch (or load from cache) raw batting stats and apply preprocessing.

Outputs
-------
data/raw/batting_stats_2021_2025.csv  (created only if absent or --refetch)

Notes
-----
FanGraphs may block requests from certain IPs (anti-scraping). If fetching
fails, place a pre-downloaded CSV at data/raw/batting_stats_2021_2025.csv
and re-run — the script will detect it and skip the network call.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from baseball2vec.data import RAW_CACHE, load_raw, preprocess

parser = argparse.ArgumentParser(description="Build Baseball2Vec dataset")
parser.add_argument(
    "--refetch", action="store_true", help="Force re-fetch from pybaseball"
)
args = parser.parse_args()

print("=" * 60)
print("Step 1: Build dataset")
print("=" * 60)

raw = load_raw(refetch=args.refetch)
df = preprocess(raw)

print(f"  Player-seasons: {len(df)}")
print(f"  Seasons: {sorted(df['Season'].unique())}")
print(f"  Raw cache: {RAW_CACHE}")
