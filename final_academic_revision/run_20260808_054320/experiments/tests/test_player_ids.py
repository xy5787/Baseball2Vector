from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from common import build_transitions, load_player_seasons


def test_stable_ids_are_complete_and_unique_per_season():
    players = load_player_seasons()
    assert len(players) == 2286
    assert players["player_id"].notna().all()
    assert not players.duplicated(["player_id", "Season"]).any()
    assert players["player_id"].str.startswith(("fg:", "mlbam:")).all()


def test_transitions_preserve_identity_and_are_unique():
    transitions = build_transitions(load_player_seasons())
    assert len(transitions) == 1414
    assert not transitions.duplicated(["player_id", "Season_t", "Season_t1"]).any()
    assert (transitions["Season_t1"] == transitions["Season_t"] + 1).all()
    assert (transitions["birth_year_t"] == transitions["birth_year_t1"]).all()

