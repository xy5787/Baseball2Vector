from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from common import RUN_ROOT, SAFE_TOOL_COLS, build_transitions, load_player_seasons, ridge_pipeline


def test_temporal_split_and_predictor_seasons():
    transitions = build_transitions(load_player_seasons())
    train = transitions[transitions["Season_t1"].isin([2022, 2023, 2024])]
    test = transitions[(transitions["Season_t"].eq(2024)) & transitions["Season_t1"].eq(2025)]
    assert len(train) == 1056
    assert len(test) == 358
    assert train["Season_t1"].max() == 2024
    assert test["Season_t"].nunique() == test["Season_t1"].nunique() == 1
    assert (test["Season_t1"] == test["Season_t"] + 1).all()
    features = [f"{col}_t" for col in SAFE_TOOL_COLS] + ["wRC+_t", "age_t_t", "PA_t"]
    assert all(not name.endswith("_t1") for name in features)
    assert all("2025" not in name for name in features)
    assert (test["age_t_t"] == test["Season_t"] - test["birth_year_t"]).all()


def test_grouped_inner_cv_and_train_only_scaler():
    transitions = build_transitions(load_player_seasons())
    train = transitions[transitions["Season_t1"].isin([2022, 2023, 2024])]
    test = transitions[(transitions["Season_t"].eq(2024)) & transitions["Season_t1"].eq(2025)]
    splitter = GroupKFold(5)
    for fit, valid in splitter.split(train, train["wRC+_t1"], train["player_id"]):
        assert set(train.iloc[fit]["player_id"]).isdisjoint(set(train.iloc[valid]["player_id"]))
    features = [f"{col}_t" for col in SAFE_TOOL_COLS]
    pipeline = ridge_pipeline(features, 10.0).fit(train[features], train["wRC+_t1"])
    scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
    imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    imputed_train = imputer.transform(train[features])
    assert np.allclose(scaler.mean_, imputed_train.mean(axis=0))
    combined = pd.concat([train[features], test[features]], ignore_index=True)
    assert not np.allclose(scaler.mean_, imputer.transform(combined).mean(axis=0))


def test_common_held_out_players_for_every_model():
    predictions = pd.read_csv(RUN_ROOT / "results" / "calibrated_predictions.csv")
    primary = predictions[predictions["feature_variant"].eq("within_input_season_standardized")]
    expected = set(primary["player_id"].unique())
    assert len(expected) == 358
    for _, group in primary.groupby("model"):
        assert set(group["player_id"]) == expected
        assert len(group) == 358

