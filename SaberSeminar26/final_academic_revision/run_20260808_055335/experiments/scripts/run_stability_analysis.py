"""Adjacent-season dimension stability overall and by minimum-PA stratum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    N_BOOT, RUN_ROOT, SEED, TOOL_COLS, TOOL_NAMES, build_transitions,
    expanded_cluster_indices, load_player_seasons, metric, percentile_ci,
)


def pa_stratum(min_pa: float) -> str:
    if min_pa < 250:
        return "100-249"
    if min_pa < 500:
        return "250-499"
    return "At least 500"


def clustered_correlations(subset: pd.DataFrame, left: str, right: str, seed: int) -> dict:
    ids = subset["player_id"].to_numpy()
    unique = np.unique(ids)
    rng = np.random.default_rng(seed)
    pearson_values, spearman_values = [], []
    x, y = subset[left].to_numpy(float), subset[right].to_numpy(float)
    for _ in range(N_BOOT):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = expanded_cluster_indices(ids, sampled)
        pearson_values.append(metric(x[idx], y[idx], "pearson"))
        spearman_values.append(metric(x[idx], y[idx], "spearman"))
    p_low, p_high = percentile_ci(pearson_values)
    s_low, s_high = percentile_ci(spearman_values)
    return {
        "pearson": metric(x, y, "pearson"), "pearson_ci_low": p_low, "pearson_ci_high": p_high,
        "spearman": metric(x, y, "spearman"), "spearman_ci_low": s_low, "spearman_ci_high": s_high,
    }


def main() -> None:
    transitions = build_transitions(load_player_seasons())
    transitions["minimum_pa"] = transitions[["PA_t", "PA_t1"]].min(axis=1)
    transitions["pa_stratum"] = transitions["minimum_pa"].map(pa_stratum)
    subsets = {
        "Overall": transitions,
        "100-249": transitions[transitions["pa_stratum"].eq("100-249")],
        "250-499": transitions[transitions["pa_stratum"].eq("250-499")],
        "At least 500": transitions[transitions["pa_stratum"].eq("At least 500")],
    }
    rows: list[dict] = []
    for subset_index, (subset_name, subset) in enumerate(subsets.items()):
        for tool_index, (dimension, col) in enumerate(zip(TOOL_NAMES, TOOL_COLS)):
            values = clustered_correlations(
                subset, f"{col}_t", f"{col}_t1", SEED + 100 * subset_index + tool_index
            )
            rows.append({
                "dimension": dimension, "subset": subset_name,
                "n_pairs": len(subset), "n_players": subset["player_id"].nunique(),
                **values, "bootstrap_unit": "stable player ID", "n_boot": N_BOOT,
                "pa_definition": "minimum of PA_t and PA_t+1",
            })
    stability = pd.DataFrame(rows)
    stability.to_csv(RUN_ROOT / "results" / "stability_by_pa.csv", index=False)
    print(stability.to_string(index=False))


if __name__ == "__main__":
    main()
