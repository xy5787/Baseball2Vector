"""Recompute the manuscript's transition-association table with stable player IDs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from baseball2vec.roi import build_delta_dataset, run_regression  # noqa: E402


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "phase1_grouped_cv", REPO_ROOT / "scripts" / "revision" / "phase1_grouped_cv.py"
    )
    phase1 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(phase1)

    players = pd.read_csv(EXP_ROOT / "data_intermediate" / "player_seasons_with_ids.csv")
    players = players[players["id_status"].eq("matched")].copy()
    players["DisplayName"] = players["Name"]
    players["Name"] = players["player_id"]
    transitions = build_delta_dataset(players)
    assert len(transitions) == 1414

    pooled = phase1.pooled_kfold_r2(transitions)
    grouped = phase1.grouped_kfold_r2(transitions, players, refit_scaling=False)
    grouped_refit = phase1.grouped_kfold_r2(transitions, players, refit_scaling=True)
    forward = phase1.forward_time_holdout(transitions, players)
    fitted = {result.target: result for result in run_regression(transitions, random_state=42)}

    rows = []
    for target in ["D_wRC+", "D_WAR", "D_OPS"]:
        nice = target.replace("D_", "Δ")
        fit = fitted[nice]
        rows.append(
            {
                "target": nice,
                "n_transitions": len(transitions),
                "n_players": transitions["Name"].nunique(),
                "pooled_lr_r2": pooled[target]["lr_r2"],
                "pooled_rf_r2": pooled[target]["rf10_r2"],
                "grouped_lr_r2": grouped[target]["lr_r2"],
                "grouped_rf_r2": grouped[target]["rf10_r2"],
                "grouped_refit_lr_r2": grouped_refit[target]["lr_r2"],
                "grouped_refit_rf_r2": grouped_refit[target]["rf10_r2"],
                "forward_lr_r2": forward[target]["lr_r2"],
                "forward_rf_r2": forward[target]["rf10_r2"],
                "forward_n_train": forward[target]["n_train"],
                "forward_n_test": forward[target]["n_test"],
                **{f"coef_{name}": value for name, value in fit.lr_coefs.items()},
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(EXP_ROOT / "results" / "stable_id_transition_reanalysis.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
