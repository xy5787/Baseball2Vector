"""Matching thresholds, selection, and random-pair constraints."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

RUN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN_ROOT / "experiments" / "scripts"))

from run_matching_analysis import WAR_TOLERANCE, WRC_TOLERANCE, pa_stratum  # noqa: E402


def test_joint_threshold_and_identity_rules() -> None:
    pairs = pd.read_csv(RUN_ROOT / "results" / "matched_profile_distances.csv")
    matched = pairs[pairs["pair_type"].eq("scalar_matched")]
    assert len(matched) == 2080
    assert (matched["abs_delta_wrc"] <= WRC_TOLERANCE + 1e-12).all()
    assert (matched["abs_delta_war"] <= WAR_TOLERANCE + 1e-12).all()
    assert (matched["focal_season"] == matched["comparison_season"]).all()
    assert (matched["focal_player_id"] != matched["comparison_player_id"]).all()


def test_selected_candidate_minimizes_scalar_gap() -> None:
    players = pd.read_csv(RUN_ROOT / "data_intermediate" / "player_seasons_with_ids.csv")
    players = players[players["id_status"].eq("matched")]
    pairs = pd.read_csv(RUN_ROOT / "results" / "matched_profile_distances.csv")
    matched = pairs[pairs["pair_type"].eq("scalar_matched")]
    lookup = players.set_index(["player_id", "Season"])
    for row in matched.itertuples(index=False):
        focal = lookup.loc[(row.focal_player_id, row.focal_season)]
        cohort = players[players["Season"].eq(row.focal_season)].copy()
        cohort = cohort[~cohort["player_id"].eq(row.focal_player_id)]
        cohort["dwrc"] = (cohort["wRC+"] - focal["wRC+"]).abs()
        cohort["dwar"] = (cohort["WAR"] - focal["WAR"]).abs()
        cohort = cohort[(cohort["dwrc"] <= WRC_TOLERANCE) & (cohort["dwar"] <= WAR_TOLERANCE)]
        wrc_sd = players[players["Season"].eq(row.focal_season)]["wRC+"].std(ddof=1)
        war_sd = players[players["Season"].eq(row.focal_season)]["WAR"].std(ddof=1)
        cohort["gap"] = np.sqrt((cohort["dwrc"] / wrc_sd) ** 2 + (cohort["dwar"] / war_sd) ** 2)
        cohort["gap_sort"] = cohort["gap"].round(12)
        selected = cohort.sort_values(["gap_sort", "player_id", "source_row"], kind="mergesort").iloc[0]
        assert np.isclose(row.standardized_scalar_gap, selected["gap"])
        assert row.comparison_player_id == selected["player_id"]
        assert row.comparison_source_row == selected["source_row"]


def test_random_pairs_share_season_and_pa_stratum() -> None:
    pairs = pd.read_csv(RUN_ROOT / "results" / "matched_profile_distances.csv")
    random = pairs[pairs["pair_type"].eq("random_same_season_pa_stratum")]
    matched = pairs[pairs["pair_type"].eq("scalar_matched")]
    assert len(random) == len(matched) == 2080
    assert (random["focal_season"] == random["comparison_season"]).all()
    assert (random["focal_player_id"] != random["comparison_player_id"]).all()
    assert all(pa_stratum(pa) == stratum for pa, stratum in zip(random["comparison_pa"], random["pa_stratum"]))
