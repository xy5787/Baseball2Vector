"""Extend the paper's Concurrent Dimension-Change Association analysis
(originally delta-v vs delta-wRC+ only) to also regress on delta-WAR and delta-OPS.

Same data, same transitions, same archived (pooled MinMax 20-80) ZScore_* scale,
same player-cluster bootstrap procedure as
final_academic_revision/run_20260808_055335/experiments/scripts/generate_artifacts.py::delta_coefficients().
Read-only against the original run; writes only to this folder's outputs/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
TRANSITIONS_PATH = REPO_ROOT / "final_academic_revision" / "run_20260808_055335" / "data_intermediate" / "stable_id_transitions.csv"
OUT_DIR = HERE / "outputs"

SEED = 42
N_BOOT = 2000
DIM_NAMES = ["Contact", "Power", "Plate Discipline", "Defense", "Speed"]
TOOL_COLS = ["ZScore_Contact", "ZScore_Power", "ZScore_Discipline", "ZScore_Defense", "ZScore_Speed"]
TARGETS = ["wRC+", "WAR", "OPS"]


def percentile_ci(values) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    return tuple(np.quantile(clean, [0.025, 0.975]))


def main() -> None:
    transitions = pd.read_csv(TRANSITIONS_PATH)

    features = []
    for name, col in zip(DIM_NAMES, TOOL_COLS):
        delta_col = f"delta_{name.replace(' ', '_')}"
        transitions[delta_col] = transitions[f"{col}_t1"] - transitions[f"{col}_t"]
        features.append(delta_col)

    for target in TARGETS:
        transitions[f"delta_{target}"] = transitions[f"{target}_t1"] - transitions[f"{target}_t"]

    ids = transitions["player_id"].to_numpy(str)
    unique = np.unique(ids)

    all_rows = []
    for target in TARGETS:
        y_col = f"delta_{target}"
        model = LinearRegression().fit(transitions[features], transitions[y_col])
        rng = np.random.default_rng(SEED)
        samples = np.empty((N_BOOT, len(features)))
        for replicate in range(N_BOOT):
            sampled = rng.choice(unique, len(unique), replace=True)
            idx = np.concatenate([np.flatnonzero(ids == pid) for pid in sampled])
            fitted = LinearRegression().fit(transitions.iloc[idx][features], transitions.iloc[idx][y_col])
            samples[replicate] = fitted.coef_
        r2 = model.score(transitions[features], transitions[y_col])
        for i, dimension in enumerate(DIM_NAMES):
            low, high = percentile_ci(samples[:, i])
            all_rows.append({
                "target": target, "dimension": dimension,
                "coefficient": model.coef_[i], "ci_low": low, "ci_high": high,
                "excludes_zero": not (low <= 0 <= high),
                "model_r2": r2, "n_transitions": len(transitions), "n_players": len(unique),
            })

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_DIR / "delta_association_coefficients_multi_target.csv", index=False)

    for target in TARGETS:
        sub = out[out["target"].eq(target)]
        print(f"\n=== delta-{target} ~ delta(5 dims)  (R^2={sub['model_r2'].iloc[0]:.3f}, n={sub['n_transitions'].iloc[0]}, players={sub['n_players'].iloc[0]}) ===")
        print(sub[["dimension", "coefficient", "ci_low", "ci_high", "excludes_zero"]].to_string(index=False))


if __name__ == "__main__":
    main()
