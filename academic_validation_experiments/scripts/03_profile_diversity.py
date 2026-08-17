"""Scalar-matched profile diversity with deterministic, value-only matching."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validation_utils import (  # noqa: E402
    EXP_ROOT,
    TOOL_COLS,
    build_transitions,
    ensure_dirs,
    load_config,
    pa_stratum,
    percentile_ci,
    vector_distance,
    weighted_quantile,
)


def make_pair(
    focal: pd.Series,
    candidate: pd.Series,
    pair_type: str,
    stds: np.ndarray,
    scalar_gap: float,
) -> dict:
    row = {
        "pair_type": pair_type,
        "focal_player_id": focal["player_id"],
        "comparison_player_id": candidate["player_id"],
        "focal_name": focal["Name"],
        "comparison_name": candidate["Name"],
        "focal_season": int(focal["Season"]),
        "comparison_season": int(candidate["Season"]),
        "focal_pa": float(focal["PA"]),
        "comparison_pa": float(candidate["PA"]),
        "focal_wrc": float(focal["wRC+"]),
        "comparison_wrc": float(candidate["wRC+"]),
        "focal_war": float(focal["WAR"]),
        "comparison_war": float(candidate["WAR"]),
        "abs_delta_wrc": float(abs(focal["wRC+"] - candidate["wRC+"])),
        "abs_delta_war": float(abs(focal["WAR"] - candidate["WAR"])),
        "scalar_gap": float(scalar_gap),
        "profile_distance": vector_distance(
            focal[TOOL_COLS].to_numpy(float), candidate[TOOL_COLS].to_numpy(float), stds
        ),
    }
    for col in TOOL_COLS:
        row[f"focal_{col}"] = float(focal[col])
        row[f"comparison_{col}"] = float(candidate[col])
    return row


def matched_pairs(
    df: pd.DataFrame,
    stds: np.ndarray,
    mode: str,
    wrc_tol: float,
    war_tol: float,
) -> list[dict]:
    rows: list[dict] = []
    pair_type = {
        "intersection": "scalar_matched_intersection",
        "wrc_only": "matched_wrc_only",
        "war_only": "matched_war_only",
    }[mode]
    for _, season_df in df.groupby("Season", sort=True):
        season_df = season_df.sort_values(["player_id", "source_row"], kind="mergesort")
        wrc_sd = float(season_df["wRC+"].std(ddof=1))
        war_sd = float(season_df["WAR"].std(ddof=1))
        for _, focal in season_df.iterrows():
            candidates = season_df[~season_df["player_id"].eq(focal["player_id"])].copy()
            dwrc = (candidates["wRC+"] - focal["wRC+"]).abs()
            dwar = (candidates["WAR"] - focal["WAR"]).abs()
            if mode == "intersection":
                candidates = candidates[(dwrc <= wrc_tol) & (dwar <= war_tol)].copy()
            elif mode == "wrc_only":
                candidates = candidates[dwrc <= wrc_tol].copy()
            else:
                candidates = candidates[dwar <= war_tol].copy()
            if candidates.empty:
                continue
            candidates["_dwrc"] = (candidates["wRC+"] - focal["wRC+"]).abs()
            candidates["_dwar"] = (candidates["WAR"] - focal["WAR"]).abs()
            if mode == "intersection":
                candidates["_scalar_gap"] = np.sqrt(
                    (candidates["_dwrc"] / wrc_sd) ** 2 + (candidates["_dwar"] / war_sd) ** 2
                )
            elif mode == "wrc_only":
                candidates["_scalar_gap"] = candidates["_dwrc"] / wrc_sd
            else:
                candidates["_scalar_gap"] = candidates["_dwar"] / war_sd
            candidate = candidates.sort_values(
                ["_scalar_gap", "_dwrc", "_dwar", "player_id", "source_row"],
                kind="mergesort",
            ).iloc[0]
            rows.append(make_pair(focal, candidate, pair_type, stds, candidate["_scalar_gap"]))
    return rows


def random_pairs(df: pd.DataFrame, stds: np.ndarray, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    working = df.copy()
    working["pa_stratum"] = working["PA"].map(pa_stratum)
    for _, season_df in working.groupby("Season", sort=True):
        season_df = season_df.sort_values(["player_id", "source_row"], kind="mergesort")
        for _, focal in season_df.iterrows():
            candidates = season_df[
                ~season_df["player_id"].eq(focal["player_id"])
                & season_df["pa_stratum"].eq(focal["pa_stratum"])
            ]
            if candidates.empty:
                continue
            candidate = candidates.iloc[int(rng.integers(0, len(candidates)))]
            rows.append(make_pair(focal, candidate, "random_same_season_pa_stratum", stds, np.nan))
    return rows


def adjacent_pairs(player_df: pd.DataFrame, stds: np.ndarray) -> list[dict]:
    transitions = build_transitions(player_df)
    rows: list[dict] = []
    for _, transition in transitions.iterrows():
        focal = pd.Series(
            {
                "player_id": transition["player_id"],
                "Name": transition["Name"],
                "Season": transition["Season_t"],
                "PA": transition["PA_t"],
                "wRC+": transition["wRC+_t"],
                "WAR": transition["WAR_t"],
                **{col: transition[f"{col}_t"] for col in TOOL_COLS},
            }
        )
        candidate = pd.Series(
            {
                "player_id": transition["player_id"],
                "Name": transition["Name"],
                "Season": transition["Season_t1"],
                "PA": transition["PA_t1"],
                "wRC+": transition["wRC+_t1"],
                "WAR": transition["WAR_t1"],
                **{col: transition[f"{col}_t1"] for col in TOOL_COLS},
            }
        )
        rows.append(make_pair(focal, candidate, "same_player_adjacent", stds, np.nan))
    return rows


def bootstrap_summaries(pairs: pd.DataFrame, player_ids: np.ndarray, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = ["same_player_adjacent", "scalar_matched_intersection", "random_same_season_pa_stratum"]
    rng = np.random.default_rng(seed)
    medians = {group: [] for group in primary}
    for _ in range(n_boot):
        sampled = rng.choice(player_ids, size=len(player_ids), replace=True)
        ids, counts = np.unique(sampled, return_counts=True)
        weights_by_id = dict(zip(ids, counts))
        for group in primary:
            sub = pairs[pairs["pair_type"].eq(group)]
            selected = sub[sub["focal_player_id"].isin(weights_by_id)].copy()
            weights = selected["focal_player_id"].map(weights_by_id).to_numpy(float)
            medians[group].append(weighted_quantile(selected["profile_distance"].to_numpy(), weights, 0.5))
    summary_rows = []
    for group in primary:
        sub = pairs[pairs["pair_type"].eq(group)]
        lo, hi = percentile_ci(medians[group])
        summary_rows.append(
            {
                "pair_type": group,
                "n_pairs": len(sub),
                "n_focal_players": sub["focal_player_id"].nunique(),
                "median": sub["profile_distance"].median(),
                "q1": sub["profile_distance"].quantile(0.25),
                "q3": sub["profile_distance"].quantile(0.75),
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    diff_rows = []
    for left, right in [
        ("scalar_matched_intersection", "same_player_adjacent"),
        ("random_same_season_pa_stratum", "scalar_matched_intersection"),
        ("random_same_season_pa_stratum", "same_player_adjacent"),
    ]:
        diff = np.asarray(medians[left]) - np.asarray(medians[right])
        lo, hi = percentile_ci(diff)
        point = (
            pairs.loc[pairs["pair_type"].eq(left), "profile_distance"].median()
            - pairs.loc[pairs["pair_type"].eq(right), "profile_distance"].median()
        )
        diff_rows.append({"left": left, "right": right, "median_difference": point, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(summary_rows), pd.DataFrame(diff_rows)


def make_plot(pairs: pd.DataFrame) -> None:
    groups = ["same_player_adjacent", "scalar_matched_intersection", "random_same_season_pa_stratum"]
    labels = ["Same player\nadjacent seasons", "Scalar-matched\ndifferent players", "Random same-season\nPA-stratified"]
    data = [pairs.loc[pairs["pair_type"].eq(group), "profile_distance"].to_numpy() for group in groups]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    colors = ["#4C78A8", "#F58518", "#9D755D"]
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.65)
    ax.boxplot(data, widths=0.16, showfliers=False, patch_artist=True, boxprops={"facecolor": "white"}, medianprops={"color": "black", "linewidth": 1.5})
    ax.set_xticks([1, 2, 3], labels)
    ax.set_ylabel("Standardized five-dimensional distance")
    ax.set_title("Profile distance by pair construction")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXP_ROOT / "figures" / "matched_profile_distances.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    player_df = pd.read_csv(EXP_ROOT / "data_intermediate" / "player_seasons_with_ids.csv")
    matched = player_df[player_df["id_status"].eq("matched")].copy()
    assert not matched.duplicated(["player_id", "Season"]).any()
    stds = matched[TOOL_COLS].std(ddof=1).to_numpy(float)
    rows = []
    rows.extend(adjacent_pairs(player_df, stds))
    rows.extend(matched_pairs(matched, stds, "intersection", cfg["matched_wrc_tolerance"], cfg["matched_war_tolerance"]))
    rows.extend(matched_pairs(matched, stds, "wrc_only", cfg["matched_wrc_tolerance"], cfg["matched_war_tolerance"]))
    rows.extend(matched_pairs(matched, stds, "war_only", cfg["matched_wrc_tolerance"], cfg["matched_war_tolerance"]))
    rows.extend(random_pairs(matched, stds, cfg["seed"]))
    pairs = pd.DataFrame(rows)
    assert (pairs.loc[pairs["pair_type"].eq("scalar_matched_intersection"), "abs_delta_wrc"] <= cfg["matched_wrc_tolerance"] + 1e-12).all()
    assert (pairs.loc[pairs["pair_type"].eq("scalar_matched_intersection"), "abs_delta_war"] <= cfg["matched_war_tolerance"] + 1e-12).all()
    assert (pairs.loc[~pairs["pair_type"].eq("same_player_adjacent"), "focal_player_id"] != pairs.loc[~pairs["pair_type"].eq("same_player_adjacent"), "comparison_player_id"]).all()
    pairs.to_csv(EXP_ROOT / "results" / "matched_profile_distances.csv", index=False)

    summary, differences = bootstrap_summaries(
        pairs, matched["player_id"].unique(), cfg["bootstrap_replicates"], cfg["seed"]
    )
    summary.to_csv(EXP_ROOT / "results" / "matched_profile_distance_summary.csv", index=False)
    differences.to_csv(EXP_ROOT / "results" / "matched_profile_distance_differences.csv", index=False)

    sensitivity = []
    for group in ["matched_wrc_only", "matched_war_only"]:
        sub = pairs[pairs["pair_type"].eq(group)]
        sensitivity.append(
            {
                "pair_type": group,
                "n_pairs": len(sub),
                "median": sub["profile_distance"].median(),
                "q1": sub["profile_distance"].quantile(0.25),
                "q3": sub["profile_distance"].quantile(0.75),
            }
        )
    pd.DataFrame(sensitivity).to_csv(EXP_ROOT / "results" / "matched_profile_sensitivity.csv", index=False)

    illustrative_pool = pairs[
        pairs["pair_type"].eq("scalar_matched_intersection")
        & (pairs["focal_pa"] >= 400)
        & (pairs["comparison_pa"] >= 400)
    ].copy()
    illustrative_pool["unordered_key"] = illustrative_pool.apply(
        lambda r: "|".join(sorted([f"{r.focal_player_id}:{r.focal_season}", f"{r.comparison_player_id}:{r.comparison_season}"])), axis=1
    )
    illustrative_pool = illustrative_pool.sort_values(["scalar_gap", "unordered_key"]).drop_duplicates("unordered_key")
    target_distance = illustrative_pool["profile_distance"].quantile(cfg["illustrative_distance_quantile"])
    illustrative_pool["distance_to_prespecified_quantile"] = (illustrative_pool["profile_distance"] - target_distance).abs()
    illustrative = illustrative_pool.sort_values(
        ["distance_to_prespecified_quantile", "scalar_gap", "unordered_key"], kind="mergesort"
    ).head(1)
    illustrative.to_csv(EXP_ROOT / "results" / "illustrative_matched_pair.csv", index=False)
    make_plot(pairs)
    print(summary.to_string(index=False))
    print("\nPairwise differences:\n", differences.to_string(index=False))
    print("\nIllustrative pair:\n", illustrative[["focal_name", "comparison_name", "focal_season", "profile_distance", "abs_delta_wrc", "abs_delta_war"]].to_string(index=False))


if __name__ == "__main__":
    main()
