from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from common import RUN_ROOT, TOOL_COLS, load_player_seasons
from run_matching_analysis import WRC_TOLERANCE, WAR_TOLERANCE, pa_stratum


def test_joint_threshold_identity_and_season_rules():
    pairs = pd.read_csv(RUN_ROOT / "results" / "matched_profile_distances.csv")
    matched = pairs[pairs["pair_type"].eq("scalar_matched")]
    assert len(matched) == 2080
    assert (matched["abs_delta_wrc"] <= WRC_TOLERANCE + 1e-12).all()
    assert (matched["abs_delta_war"] <= WAR_TOLERANCE + 1e-12).all()
    assert (matched["focal_season"] == matched["comparison_season"]).all()
    assert (matched["focal_player_id"] != matched["comparison_player_id"]).all()


def test_selected_candidate_has_minimum_standardized_scalar_gap():
    players = load_player_seasons()
    pairs = pd.read_csv(RUN_ROOT / "results" / "matched_profile_distances.csv")
    matched = pairs[pairs["pair_type"].eq("scalar_matched")]
    for season, selected_rows in matched.groupby("focal_season"):
        cohort = players[players["Season"].eq(season)]
        wrc_sd, war_sd = cohort["wRC+"].std(ddof=1), cohort["WAR"].std(ddof=1)
        lookup = cohort.set_index("player_id")
        for selected in selected_rows.itertuples(index=False):
            focal = lookup.loc[selected.focal_player_id]
            candidates = cohort[~cohort["player_id"].eq(selected.focal_player_id)].copy()
            candidates["dwrc"] = (candidates["wRC+"] - focal["wRC+"]).abs()
            candidates["dwar"] = (candidates["WAR"] - focal["WAR"]).abs()
            candidates = candidates[(candidates["dwrc"] <= WRC_TOLERANCE) & (candidates["dwar"] <= WAR_TOLERANCE)]
            candidates["gap"] = np.sqrt((candidates["dwrc"] / wrc_sd) ** 2 + (candidates["dwar"] / war_sd) ** 2)
            expected = candidates.sort_values(["gap", "player_id", "source_row"], kind="mergesort").iloc[0]
            assert selected.comparison_player_id == expected["player_id"]


def test_random_group_uses_same_focal_basis_and_pa_strata():
    pairs = pd.read_csv(RUN_ROOT / "results" / "matched_profile_distances.csv")
    matched = pairs[pairs["pair_type"].eq("scalar_matched")].sort_values(["focal_season", "focal_player_id"])
    random = pairs[pairs["pair_type"].eq("random_same_season_pa_stratum")].sort_values(["focal_season", "focal_player_id"])
    assert len(random) == len(matched) == 2080
    assert list(zip(random["focal_player_id"], random["focal_season"])) == list(zip(matched["focal_player_id"], matched["focal_season"]))
    assert (random["focal_season"] == random["comparison_season"]).all()
    assert (random["focal_player_id"] != random["comparison_player_id"]).all()
    assert (random["focal_pa"].map(pa_stratum) == random["comparison_pa"].map(pa_stratum)).all()
