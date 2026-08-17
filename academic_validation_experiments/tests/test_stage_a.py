from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parent
sys.path.insert(0, str(EXP_ROOT / "scripts"))

from validation_utils import TOOL_COLS, build_transitions, sha256_file  # noqa: E402


def test_authoritative_inputs_unchanged_and_headlines_reproduced() -> None:
    provenance = json.loads((EXP_ROOT / "results" / "input_provenance.json").read_text())
    manuscript = REPO_ROOT / "paper" / "baseball2vector_en.tex"
    source = REPO_ROOT / "data" / "processed" / "v3_results.csv"
    assert sha256_file(manuscript) == provenance["manuscript_sha256_stage_a_start"]
    assert sha256_file(source) == provenance["source_sha256"]
    reproduction = json.loads((EXP_ROOT / "results" / "headline_reproduction.json").read_text())
    assert reproduction["rows"] == 2309
    assert np.isclose(reproduction["zscore_area_war_r"], 0.8060958222864365)
    assert np.isclose(reproduction["name_keyed_delta_wrc_lr_cv_r2"], 0.7068488591794327)


def test_stable_id_integrity_and_collision_exclusion() -> None:
    df = pd.read_csv(EXP_ROOT / "data_intermediate" / "player_seasons_with_ids.csv")
    matched = df[df["id_status"].eq("matched")]
    assert matched["player_id"].notna().all()
    assert not matched.duplicated(["player_id", "Season"]).any()
    muncy_2025 = df[df["Name"].eq("Max Muncy") & df["Season"].eq(2025)]
    assert len(muncy_2025) == 2
    assert muncy_2025["id_status"].eq("ambiguous").all()
    assert muncy_2025["player_id"].isna().all()


def test_transition_alignment_and_uniqueness() -> None:
    df = pd.read_csv(EXP_ROOT / "data_intermediate" / "player_seasons_with_ids.csv")
    transitions = build_transitions(df)
    assert len(transitions) == 1414
    assert (transitions["Season_t1"] == transitions["Season_t"] + 1).all()
    assert not transitions.duplicated(["player_id", "Season_t", "Season_t1"]).any()


def test_prospective_test_is_common_and_future_free() -> None:
    pred = pd.read_csv(EXP_ROOT / "results" / "prospective_predictions.csv")
    assert pred["Season_t"].eq(2024).all()
    assert pred["Season_t1"].eq(2025).all()
    for target, target_df in pred.groupby("target"):
        expected = None
        for _, model_df in target_df.groupby("model"):
            ids = set(model_df["player_id"])
            expected = ids if expected is None else expected
            assert ids == expected, target
            assert len(model_df) == 358
    results = pd.read_csv(EXP_ROOT / "results" / "prospective_prediction.csv")
    ok = results[results["status"].eq("ok")]
    feature_text = " ".join(ok["features"].dropna().astype(str))
    assert "_t1" not in feature_text
    assert set(ok["n_train"]) == {1056}
    assert set(ok["n_test"]) == {358}


def test_required_result_schemas_and_unavailable_statuses() -> None:
    required = {
        "prospective_prediction.csv": {"status", "target", "model", "metric", "estimate", "ci_low", "ci_high"},
        "prospective_prediction_bootstrap.csv": {"target", "replicate", "model", "metric", "value"},
        "representation_stability.csv": {"subset", "metric", "dimension", "estimate", "ci_low", "ci_high"},
        "shrinkage_sensitivity.csv": {"setting", "status", "thresholds_json"},
        "leave_one_out_ablation.csv": {"dimension", "removed_stat", "status"},
        "defense_variants.csv": {"variant", "status", "playing_time_dependent"},
        "matched_profile_distances.csv": {"pair_type", "focal_player_id", "comparison_player_id", "profile_distance"},
        "sample_flow.csv": {"stage", "n_rows", "n_players"},
    }
    for filename, columns in required.items():
        path = EXP_ROOT / "results" / filename
        assert path.exists(), filename
        assert columns.issubset(pd.read_csv(path).columns), filename
    shrink = pd.read_csv(EXP_ROOT / "results" / "shrinkage_sensitivity.csv")
    assert shrink.loc[shrink["setting"].isin(["no_shrinkage", "stronger_2x"]), "status"].eq("not_run_missing_raw").all()
    loo = pd.read_csv(EXP_ROOT / "results" / "leave_one_out_ablation.csv")
    assert len(loo) == 22
    assert loo["status"].eq("not_run_missing_raw").all()


def test_matching_rules_and_summaries() -> None:
    pairs = pd.read_csv(EXP_ROOT / "results" / "matched_profile_distances.csv")
    matched = pairs[pairs["pair_type"].eq("scalar_matched_intersection")]
    assert (matched["abs_delta_wrc"] <= 5 + 1e-12).all()
    assert (matched["abs_delta_war"] <= 0.2 + 1e-12).all()
    assert (matched["focal_player_id"] != matched["comparison_player_id"]).all()
    summary = pd.read_csv(EXP_ROOT / "results" / "matched_profile_distance_summary.csv")
    for row in summary.itertuples(index=False):
        actual = pairs.loc[pairs["pair_type"].eq(row.pair_type), "profile_distance"].median()
        assert np.isclose(actual, row.median)
    medians = summary.set_index("pair_type")["median"]
    assert medians["same_player_adjacent"] < medians["scalar_matched_intersection"]
    assert medians["scalar_matched_intersection"] < medians["random_same_season_pa_stratum"]


def test_illustrative_pair_rule() -> None:
    pairs = pd.read_csv(EXP_ROOT / "results" / "matched_profile_distances.csv")
    pool = pairs[
        pairs["pair_type"].eq("scalar_matched_intersection")
        & (pairs["focal_pa"] >= 400)
        & (pairs["comparison_pa"] >= 400)
    ].copy()
    pool["unordered_key"] = pool.apply(
        lambda r: "|".join(sorted([f"{r.focal_player_id}:{r.focal_season}", f"{r.comparison_player_id}:{r.comparison_season}"])), axis=1
    )
    pool = pool.sort_values(["scalar_gap", "unordered_key"]).drop_duplicates("unordered_key")
    target = pool["profile_distance"].quantile(0.90)
    pool["distance_to_prespecified_quantile"] = (pool["profile_distance"] - target).abs()
    expected = pool.sort_values(["distance_to_prespecified_quantile", "scalar_gap", "unordered_key"], kind="mergesort").iloc[0]
    actual = pd.read_csv(EXP_ROOT / "results" / "illustrative_matched_pair.csv").iloc[0]
    assert actual["unordered_key"] == expected["unordered_key"]

