"""Scalar-matched profile diversity, re-run on the rescaled (SD=10) 20-80 grades.

Self-contained port of
final_academic_revision/run_20260808_055335/experiments/scripts/run_matching_analysis.py,
using this folder's rescaled_results.csv (ZScore_*_old / ZScore_*_new) joined to the
Chadwick-linked stable player_id from data_intermediate/player_seasons_with_ids.csv.
Nothing in the original final_academic_revision run is modified.

Runs the identical deterministic matching + focal-player-cluster bootstrap procedure
for BOTH the archived MinMax(20,80) scale ("old") and the SD=10 rescaled grades ("new"),
so the two can be compared directly and against the paper's published numbers.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
IDS_PATH = REPO_ROOT / "final_academic_revision" / "run_20260808_055335" / "data_intermediate" / "player_seasons_with_ids.csv"
RESCALED_PATH = HERE / "outputs" / "rescaled_results.csv"
OUT_DIR = HERE / "outputs"

SEED = 42
N_BOOT = 2000
WRC_TOLERANCE = 5.0
WAR_TOLERANCE = 0.2
DIMS = ["Contact", "Power", "Discipline", "Defense", "Speed"]


def pa_stratum(pa: float) -> str:
    if pa < 250:
        return "100-249"
    if pa < 500:
        return "250-499"
    return "500+"


def load_players(variant: str) -> tuple[pd.DataFrame, list[str]]:
    ids = pd.read_csv(IDS_PATH)
    ids = ids[ids["id_status"].eq("matched")].copy()
    keep_ids = ["Name", "Season", "PA", "WAR", "wRC+", "OPS", "player_id", "source_row"]
    ids = ids[keep_ids]

    rescaled = pd.read_csv(RESCALED_PATH)
    tool_cols = [f"ZScore_{d}_{variant}" for d in DIMS]
    rescaled = rescaled[["Name", "Season", *tool_cols]]

    merged = ids.merge(rescaled, on=["Name", "Season"], how="inner")
    assert len(merged) == len(ids), "join dropped rows: rescaled_results.csv missing some stable-ID player-seasons"
    assert not merged.duplicated(["player_id", "Season"]).any()
    return merged.sort_values(["player_id", "Season", "source_row"], kind="mergesort").reset_index(drop=True), tool_cols


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for player_id, player in df.groupby("player_id", sort=True):
        player = player.sort_values(["Season", "source_row"], kind="mergesort")
        records = player.to_dict("records")
        for left, right in zip(records, records[1:]):
            if int(right["Season"]) != int(left["Season"]) + 1:
                continue
            rows.append({"player_id": player_id, "Season_t": left["Season"], "Season_t1": right["Season"]})
    out = pd.DataFrame(rows)
    assert not out.duplicated(["player_id", "Season_t", "Season_t1"]).any()
    return out


def profile_distance(left: pd.Series, right: pd.Series, tool_cols: list[str], stds: np.ndarray) -> float:
    delta = (left[tool_cols].to_numpy(float) - right[tool_cols].to_numpy(float)) / stds
    return float(np.linalg.norm(delta) / math.sqrt(len(tool_cols)))


def pair_row(pair_type: str, focal: pd.Series, comparison: pd.Series, tool_cols: list[str], stds: np.ndarray, gap: float = np.nan) -> dict:
    return {
        "pair_type": pair_type,
        "focal_player_id": focal["player_id"], "comparison_player_id": comparison["player_id"],
        "focal_name": focal["Name"], "comparison_name": comparison["Name"],
        "focal_season": int(focal["Season"]), "comparison_season": int(comparison["Season"]),
        "focal_pa": float(focal["PA"]), "comparison_pa": float(comparison["PA"]),
        "pa_stratum": pa_stratum(float(focal["PA"])),
        "focal_wrc": float(focal["wRC+"]), "comparison_wrc": float(comparison["wRC+"]),
        "focal_war": float(focal["WAR"]), "comparison_war": float(comparison["WAR"]),
        "standardized_scalar_gap": float(gap),
        "profile_distance": profile_distance(focal, comparison, tool_cols, stds),
    }


def deterministic_matches(players: pd.DataFrame, tool_cols: list[str], stds: np.ndarray) -> pd.DataFrame:
    pairs: list[dict] = []
    for season, cohort in players.groupby("Season", sort=True):
        cohort = cohort.sort_values(["player_id", "source_row"], kind="mergesort")
        wrc_sd = float(cohort["wRC+"].std(ddof=1))
        war_sd = float(cohort["WAR"].std(ddof=1))
        for _, focal in cohort.iterrows():
            candidates = cohort[~cohort["player_id"].eq(focal["player_id"])].copy()
            candidates["_dwrc"] = (candidates["wRC+"] - focal["wRC+"]).abs()
            candidates["_dwar"] = (candidates["WAR"] - focal["WAR"]).abs()
            candidates = candidates[
                (candidates["_dwrc"] <= WRC_TOLERANCE) & (candidates["_dwar"] <= WAR_TOLERANCE)
            ].copy()
            if candidates.empty:
                continue
            candidates["_gap"] = np.sqrt((candidates["_dwrc"] / wrc_sd) ** 2 + (candidates["_dwar"] / war_sd) ** 2)
            candidates["_gap_sort"] = candidates["_gap"].round(12)
            selected = candidates.sort_values(["_gap_sort", "player_id", "source_row"], kind="mergesort").iloc[0]
            pairs.append(pair_row("scalar_matched", focal, selected, tool_cols, stds, selected["_gap"]))
    return pd.DataFrame(pairs)


def random_pairs_for_matched_focals(players: pd.DataFrame, matched: pd.DataFrame, tool_cols: list[str], stds: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    lookup = players.set_index(["player_id", "Season"], drop=False)
    rows: list[dict] = []
    for m in matched.sort_values(["focal_season", "focal_player_id"], kind="mergesort").itertuples(index=False):
        focal = lookup.loc[(m.focal_player_id, m.focal_season)]
        cohort = players[
            players["Season"].eq(m.focal_season)
            & ~players["player_id"].eq(m.focal_player_id)
            & players["PA"].map(pa_stratum).eq(m.pa_stratum)
        ].sort_values(["player_id", "source_row"], kind="mergesort")
        comparison = cohort.iloc[int(rng.integers(0, len(cohort)))]
        rows.append(pair_row("random_same_season_pa_stratum", focal, comparison, tool_cols, stds))
    return pd.DataFrame(rows)


def same_player_pairs(players: pd.DataFrame, tool_cols: list[str], stds: np.ndarray) -> pd.DataFrame:
    transitions = build_transitions(players)
    lookup = players.set_index(["player_id", "Season"], drop=False)
    rows = []
    for t in transitions.itertuples(index=False):
        focal = lookup.loc[(t.player_id, t.Season_t)]
        comparison = lookup.loc[(t.player_id, t.Season_t1)]
        rows.append(pair_row("same_player_adjacent", focal, comparison, tool_cols, stds))
    return pd.DataFrame(rows)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cutoff = q * weights.sum()
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def percentile_ci(values) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    return tuple(np.quantile(clean, [0.025, 0.975]))


def clustered_summary(pairs: pd.DataFrame, all_player_ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = ["same_player_adjacent", "scalar_matched", "random_same_season_pa_stratum"]
    rng = np.random.default_rng(SEED)
    bootstrap = {g: [] for g in groups}
    for _ in range(N_BOOT):
        sampled = rng.choice(all_player_ids, len(all_player_ids), replace=True)
        ids, counts = np.unique(sampled, return_counts=True)
        weight_map = dict(zip(ids, counts))
        for g in groups:
            sub = pairs[pairs["pair_type"].eq(g)]
            selected = sub[sub["focal_player_id"].isin(weight_map)]
            weights = selected["focal_player_id"].map(weight_map).to_numpy(float)
            bootstrap[g].append(weighted_quantile(selected["profile_distance"].to_numpy(float), weights, 0.5))
    summary = []
    for g in groups:
        sub = pairs[pairs["pair_type"].eq(g)]
        low, high = percentile_ci(bootstrap[g])
        summary.append({
            "pair_type": g, "n_pairs": len(sub),
            "n_unique_focal_players": sub["focal_player_id"].nunique(),
            "n_unique_comparison_players": sub["comparison_player_id"].nunique(),
            "median": sub["profile_distance"].median(),
            "q1": sub["profile_distance"].quantile(0.25), "q3": sub["profile_distance"].quantile(0.75),
            "ci_low": low, "ci_high": high,
        })
    differences = []
    for left, right in [("scalar_matched", "same_player_adjacent"), ("random_same_season_pa_stratum", "scalar_matched")]:
        values = np.asarray(bootstrap[left]) - np.asarray(bootstrap[right])
        low, high = percentile_ci(values)
        point = pairs.loc[pairs["pair_type"].eq(left), "profile_distance"].median() - pairs.loc[pairs["pair_type"].eq(right), "profile_distance"].median()
        differences.append({"left": left, "right": right, "median_difference": point, "ci_low": low, "ci_high": high})
    return pd.DataFrame(summary), pd.DataFrame(differences)


def run_variant(variant: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players, tool_cols = load_players(variant)
    stds = players[tool_cols].std(ddof=1).to_numpy(float)
    matched = deterministic_matches(players, tool_cols, stds)
    random_pairs = random_pairs_for_matched_focals(players, matched, tool_cols, stds)
    adjacent = same_player_pairs(players, tool_cols, stds)
    pairs = pd.concat([adjacent, matched, random_pairs], ignore_index=True)
    summary, differences = clustered_summary(pairs, players["player_id"].unique())
    summary.insert(0, "scale_variant", variant)
    differences.insert(0, "scale_variant", variant)
    pairs.insert(0, "scale_variant", variant)
    return pairs, summary, differences


def main() -> None:
    all_pairs, all_summary, all_diff = [], [], []
    for variant in ["old", "new"]:
        pairs, summary, differences = run_variant(variant)
        all_pairs.append(pairs)
        all_summary.append(summary)
        all_diff.append(differences)
        print(f"\n=== variant: {variant} ===")
        print(summary.to_string(index=False))
        print(differences.to_string(index=False))

    pd.concat(all_pairs, ignore_index=True).to_csv(OUT_DIR / "matched_profile_distances_rescaled.csv", index=False)
    pd.concat(all_summary, ignore_index=True).to_csv(OUT_DIR / "matched_profile_distance_summary_rescaled.csv", index=False)
    pd.concat(all_diff, ignore_index=True).to_csv(OUT_DIR / "matched_profile_distance_differences_rescaled.csv", index=False)


if __name__ == "__main__":
    main()
