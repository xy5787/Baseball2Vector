"""Stable-player identifier integrity tests."""

from pathlib import Path

import pandas as pd

RUN_ROOT = Path(__file__).resolve().parents[2]


def test_stable_id_rows_are_unique_by_season() -> None:
    players = pd.read_csv(RUN_ROOT / "data_intermediate" / "player_seasons_with_ids.csv")
    matched = players[players["id_status"].eq("matched")]
    assert len(matched) == 2286
    assert matched["player_id"].nunique() == 818
    assert matched["player_id"].notna().all()
    assert not matched.duplicated(["player_id", "Season"]).any()


def test_ambiguous_name_collision_is_not_guessed() -> None:
    players = pd.read_csv(RUN_ROOT / "data_intermediate" / "player_seasons_with_ids.csv")
    muncy = players[players["Name"].eq("Max Muncy") & players["Season"].eq(2025)]
    assert len(muncy) == 2
    assert muncy["id_status"].eq("ambiguous").all()
    assert muncy["player_id"].isna().all()


def test_transition_keys_are_unique() -> None:
    transitions = pd.read_csv(RUN_ROOT / "data_intermediate" / "stable_id_transitions.csv")
    assert len(transitions) == 1414
    assert transitions["player_id"].nunique() == 567
    assert not transitions.duplicated(["player_id", "Season_t", "Season_t1"]).any()
