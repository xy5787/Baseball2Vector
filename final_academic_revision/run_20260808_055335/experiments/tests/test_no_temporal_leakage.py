"""Temporal split and common held-out-set assertions."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

RUN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN_ROOT / "experiments" / "scripts"))

from common import SAFE_TOOL_COLS, TOOL_COLS  # noqa: E402


def test_temporal_split_and_predictor_provenance() -> None:
    transitions = pd.read_csv(RUN_ROOT / "data_intermediate" / "stable_id_transitions.csv")
    train = transitions[transitions["Season_t1"].isin([2022, 2023, 2024])]
    test = transitions[(transitions["Season_t"].eq(2024)) & (transitions["Season_t1"].eq(2025))]
    assert len(train) == 1056
    assert len(test) == 358
    assert train["Season_t1"].max() < test["Season_t1"].min()
    assert (test["Season_t1"] == test["Season_t"] + 1).all()
    predictors = ["wRC+_t", "PA_t", "age_t_t"]
    predictors += [f"{c}_t" for c in TOOL_COLS + SAFE_TOOL_COLS]
    assert all(c in transitions for c in predictors)
    assert not any(c.endswith("_t1") for c in predictors)
    assert (test["age_t_t"] == test["Season_t"] - test["birth_year_t"]).all()


def test_identical_test_players_and_training_only_pipeline() -> None:
    predictions = pd.read_csv(RUN_ROOT / "results" / "calibrated_predictions.csv")
    primary = predictions[predictions["feature_variant"].eq("within_input_season_standardized")]
    expected = None
    for _, group in primary.groupby("model"):
        ids = set(group["player_id"])
        expected = ids if expected is None else expected
        assert ids == expected
        assert len(group) == 358
        assert group["Season_t"].eq(2024).all()
        assert group["Season_t1"].eq(2025).all()
    source = (RUN_ROOT / "experiments" / "scripts" / "common.py").read_text()
    assert "SimpleImputer" in source and "StandardScaler" in source
    assert "Pipeline" in source and "GroupKFold" in source


def test_forecast_safe_dimensions_use_only_contemporaneous_cohorts() -> None:
    players = pd.read_csv(RUN_ROOT / "data_intermediate" / "player_seasons_with_ids.csv")
    players = players[players["id_status"].eq("matched")].copy()
    for source, safe in zip(TOOL_COLS, SAFE_TOOL_COLS):
        players[safe] = players.groupby("Season")[source].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )
        grouped = players.groupby("Season")[safe]
        assert np.allclose(grouped.mean().to_numpy(), 0.0, atol=1e-12)
        assert np.allclose(grouped.std(ddof=0).to_numpy(), 1.0, atol=1e-12)


def test_archived_and_safe_variants_are_explicitly_separated() -> None:
    predictions = pd.read_csv(RUN_ROOT / "results" / "calibrated_predictions.csv")
    assert set(predictions["feature_variant"]) == {
        "archived_pooled_20_80",
        "within_input_season_standardized",
    }
    counts = predictions.groupby(["feature_variant", "model"])["player_id"].nunique()
    assert counts.eq(358).all()
    primary_models = predictions[
        predictions["feature_variant"].eq("within_input_season_standardized")
    ]["model"].unique()
    assert len(primary_models) == 7
