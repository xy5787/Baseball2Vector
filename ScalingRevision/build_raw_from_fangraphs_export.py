"""Convert a manually-downloaded FanGraphs leaderboard export into the raw
cache format `baseball2vec.data.load_raw()` expects, then run it through the
existing `preprocess()` pipeline and compare against the archived
`data/v3_results_clean.csv` (row counts, players, columns) as a first
reproducibility check — *before* touching Z-Score/PCA/VAE.

Input
-----
data/raw/fangraphs-leaderboards.csv
  A "select everything" FanGraphs batting leaderboard CSV export. Known
  quirks handled here:
    - Duplicate Name/Team/Season columns (pandas suffixes them Name.1 etc.
      on read since FanGraphs' combined-report export repeats the identifier
      columns) -> dropped.
    - May include the in-progress current season -> filtered to 2021-2025.
    - Percentage columns as 0-1 fractions rather than 0-100 -> left as-is;
      z-scoring is scale-invariant to a uniform per-column rescale, so this
      does not change any downstream Z-Score/PCA/VAE/Pentagon-Area result
      (same argument as the MinMax->SD scaling invariance in scaling.py).

Output
------
data/raw/batting_stats_2021_2025.csv   -- shared cache path baseball2vec.data
                                           reads (only written after review;
                                           see __main__ guard below)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from baseball2vec.data import RAW_CACHE, preprocess
from baseball2vec.tools import FEATURE_GROUPS, TOOL_NAMES

EXPORT_PATH = REPO_ROOT / "data" / "raw" / "fangraphs-leaderboards (1).csv"
V3_CLEAN_PATH = REPO_ROOT / "data" / "v3_results_clean.csv"
SEASONS = [2021, 2022, 2023, 2024, 2025]
QUAL_PA = 100

# Known Name+Season collisions in this export (verified against MLBAMID/Team
# that they're two distinct real players) -- disambiguated the same way the
# archived v3_results_clean.csv already does, so UniqueName construction in
# preprocess() lines up. Extend this if a future re-fetch finds more.
NAME_COLLISION_FIXES = [
    # (Name, Season, Team) -> new Name
    ("Max Muncy", 2025, "ATH", "Max Muncy (ATH)"),
]

REQUIRED_COLS = (
    ["Name", "Team", "Season", "PA", "WAR", "wRC+", "OPS"]
    + [c for cols in FEATURE_GROUPS.values() for c in cols]
)


def load_and_clean_export(path: Path = EXPORT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop exact-duplicate identifier columns FanGraphs' combined export repeats.
    for base in ["Name", "Team", "Season"]:
        dup = f"{base}.1"
        if dup in df.columns:
            assert (df[base] == df[dup]).all(), f"{base} / {dup} mismatch"
            df = df.drop(columns=[dup])

    before = len(df)
    df = df[df["Season"].isin(SEASONS)].reset_index(drop=True)
    print(f"  Filtered {before} -> {len(df)} rows (seasons {SEASONS})")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Export is missing required columns: {missing}")

    for name, season, team, new_name in NAME_COLLISION_FIXES:
        mask = (df["Name"] == name) & (df["Season"] == season) & (df["Team"] == team)
        n = mask.sum()
        if n:
            df.loc[mask, "Name"] = new_name
            print(f"  Disambiguated: {name!r} ({season}, {team}) -> {new_name!r} ({n} row)")

    return df


def build_qualified_raw_cache(df: pd.DataFrame, qual_pa: int = QUAL_PA) -> pd.DataFrame:
    """Apply the qual=100 PA floor the original pipeline used at fetch time.

    (The original `load_raw(qual=100)` filtered server-side via the FanGraphs
    query; this manual export has no such filter, so it's applied here.)
    """
    out = df[df["PA"] >= qual_pa].reset_index(drop=True)
    dup = out.duplicated(subset=["Name", "Season"], keep=False)
    assert not dup.any(), f"Unresolved Name+Season collisions:\n{out[dup][['Name','Season','Team']]}"
    return out


def compare_to_v3(df: pd.DataFrame) -> None:
    v3 = pd.read_csv(V3_CLEAN_PATH)
    print(f"\n  This export (2021-2025):      {len(df)} rows, "
          f"{df['Name'].nunique()} unique names")
    print(f"  Archived v3_results_clean.csv: {len(v3)} rows, "
          f"{v3['UniqueName'].str.replace(r' \(\d+\)$', '', regex=True).nunique()} unique names")

    export_pairs = set(zip(df["Name"], df["Season"]))
    v3_names = v3["UniqueName"].str.extract(r"^(.*) \((\d+)\)$")
    v3_pairs = set(zip(v3_names[0], v3_names[1].astype(int)))

    only_in_export = export_pairs - v3_pairs
    only_in_v3 = v3_pairs - export_pairs
    print(f"  Player-seasons only in new export: {len(only_in_export)}")
    print(f"  Player-seasons only in archived v3: {len(only_in_v3)}")
    if len(v3_pairs):
        overlap = len(export_pairs & v3_pairs) / len(v3_pairs)
        print(f"  Overlap vs archived v3 population: {overlap:.1%}")


if __name__ == "__main__":
    print("=" * 60)
    print("Building raw cache from manual FanGraphs export")
    print("=" * 60)

    if not EXPORT_PATH.exists():
        print(f"Not found: {EXPORT_PATH} -- nothing to do yet.")
        sys.exit(0)

    df = load_and_clean_export()
    compare_to_v3(df)

    print(f"\n  PA range (unfiltered): {df['PA'].min()}-{df['PA'].max()}")

    qualified = build_qualified_raw_cache(df)
    print(f"  Qualified (PA>={QUAL_PA}): {len(qualified)} rows")

    processed = preprocess(qualified.copy())
    print(f"\n  preprocess() output: {len(processed)} rows")
    print(f"  UniqueName sample: {processed['UniqueName'].head(3).tolist()}")

    RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    qualified.to_csv(RAW_CACHE, index=False)
    print(f"\n  Wrote: {RAW_CACHE} ({len(qualified)} rows)")
