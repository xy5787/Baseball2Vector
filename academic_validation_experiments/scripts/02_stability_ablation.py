"""Temporal stability plus explicit feasibility records for raw-stat ablations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validation_utils import (  # noqa: E402
    EXP_ROOT,
    TOOL_COLS,
    TOOL_NAMES,
    build_transitions,
    cluster_bootstrap_indices,
    ensure_dirs,
    load_config,
    pa_stratum,
    percentile_ci,
)


def safe_spearman(frame: pd.DataFrame, left: str, right: str) -> float:
    if len(frame) < 3 or frame[left].nunique() < 2 or frame[right].nunique() < 2:
        return float("nan")
    return float(spearmanr(frame[left], frame[right]).statistic)


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    player_df = pd.read_csv(EXP_ROOT / "data_intermediate" / "player_seasons_with_ids.csv")
    matched = player_df[player_df["id_status"].eq("matched")].copy()
    transitions = build_transitions(player_df)
    transitions["min_pa"] = transitions[["PA_t", "PA_t1"]].min(axis=1)
    transitions["pa_stratum"] = transitions["min_pa"].map(pa_stratum)

    vector_t = transitions[[f"{col}_t" for col in TOOL_COLS]].to_numpy(float)
    vector_t1 = transitions[[f"{col}_t1" for col in TOOL_COLS]].to_numpy(float)
    pooled_stds = matched[TOOL_COLS].std(ddof=1).to_numpy(float)
    transitions["cosine_similarity"] = np.sum(vector_t * vector_t1, axis=1) / (
        np.linalg.norm(vector_t, axis=1) * np.linalg.norm(vector_t1, axis=1)
    )
    transitions["standardized_euclidean"] = np.linalg.norm(
        (vector_t1 - vector_t) / pooled_stds, axis=1
    ) / np.sqrt(len(TOOL_COLS))
    transitions.to_csv(EXP_ROOT / "data_intermediate" / "stable_id_transitions.csv", index=False)

    subsets = {"overall": transitions}
    subsets.update(
        {stratum: transitions[transitions["pa_stratum"].eq(stratum)] for stratum in ["100-249", "250-499", "500+"]}
    )
    rows: list[dict] = []
    for subset_name, subset in subsets.items():
        bootstrap_indices = cluster_bootstrap_indices(
            subset["player_id"].to_numpy(), cfg["bootstrap_replicates"], cfg["seed"]
        )
        for tool, col in zip(TOOL_NAMES, TOOL_COLS):
            left, right = f"{col}_t", f"{col}_t1"
            estimate = safe_spearman(subset, left, right)
            left_values = subset[left].to_numpy(float)
            right_values = subset[right].to_numpy(float)
            boot = np.asarray(
                [spearmanr(left_values[idx], right_values[idx]).statistic for idx in bootstrap_indices],
                dtype=float,
            )
            lo, hi = percentile_ci(boot)
            rows.append(
                {
                    "subset": subset_name,
                    "metric": "spearman",
                    "dimension": tool,
                    "estimate": estimate,
                    "ci_low": lo,
                    "ci_high": hi,
                    "q1": np.nan,
                    "q3": np.nan,
                    "n_pairs": len(subset),
                    "n_players": subset["player_id"].nunique(),
                }
            )
        for metric in ["cosine_similarity", "standardized_euclidean"]:
            estimate = float(subset[metric].median())
            metric_values = subset[metric].to_numpy(float)
            boot = np.asarray([np.median(metric_values[idx]) for idx in bootstrap_indices])
            lo, hi = percentile_ci(boot)
            rows.append(
                {
                    "subset": subset_name,
                    "metric": metric,
                    "dimension": "all",
                    "estimate": estimate,
                    "ci_low": lo,
                    "ci_high": hi,
                    "q1": float(subset[metric].quantile(0.25)),
                    "q3": float(subset[metric].quantile(0.75)),
                    "n_pairs": len(subset),
                    "n_players": subset["player_id"].nunique(),
                }
            )
    stability = pd.DataFrame(rows)
    stability.to_csv(EXP_ROOT / "results" / "representation_stability.csv", index=False)

    thresholds = cfg["shrinkage_thresholds_current"]
    stronger = {key: (2 * value if value else 0) for key, value in thresholds.items()}
    low_pa = stability[(stability["subset"].eq("100-249")) & (stability["metric"].eq("spearman"))]
    high_pa_sd = matched[matched["PA"] >= 500][TOOL_COLS].std(ddof=1).mean()
    shrink_rows = [
        {
            "setting": "no_shrinkage",
            "status": "not_run_missing_raw",
            "thresholds_json": json.dumps({key: 0 for key in thresholds}),
            "low_pa_mean_dimension_spearman": np.nan,
            "high_pa_mean_dimension_sd": np.nan,
            "reason": "Raw constituent statistics are absent; pre-shrinkage values cannot be reconstructed.",
        },
        {
            "setting": "current_manuscript",
            "status": "available_saved_representation_only",
            "thresholds_json": json.dumps(thresholds),
            "low_pa_mean_dimension_spearman": float(low_pa["estimate"].mean()),
            "high_pa_mean_dimension_sd": float(high_pa_sd),
            "reason": "Current saved scores are observable, but cannot be recomputed from raw statistics.",
        },
        {
            "setting": "stronger_2x",
            "status": "not_run_missing_raw",
            "thresholds_json": json.dumps(stronger),
            "low_pa_mean_dimension_spearman": np.nan,
            "high_pa_mean_dimension_sd": np.nan,
            "reason": "Raw constituent statistics are absent; doubling pseudo-PA cannot be applied literally.",
        },
    ]
    pd.DataFrame(shrink_rows).to_csv(EXP_ROOT / "results" / "shrinkage_sensitivity.csv", index=False)

    loo_rows = []
    for dimension, stats in cfg["feature_groups"].items():
        for stat in stats:
            loo_rows.append(
                {
                    "dimension": dimension,
                    "removed_stat": stat,
                    "status": "not_run_missing_raw",
                    "spearman_full_vs_ablated": np.nan,
                    "top_decile_jaccard": np.nan,
                    "mean_abs_percentile_rank_change": np.nan,
                    "p95_abs_percentile_rank_change": np.nan,
                    "max_abs_percentile_rank_change": np.nan,
                    "reason": "The processed input stores only final dimension scores, not constituent statistics.",
                }
            )
    pd.DataFrame(loo_rows).to_csv(EXP_ROOT / "results" / "leave_one_out_ablation.csv", index=False)

    defense_stability = stability[
        stability["metric"].eq("spearman") & stability["dimension"].eq("Defense") & stability["subset"].eq("overall")
    ].iloc[0]
    defense_rows = [
        {
            "variant": "current_mean_of_Def_and_Fld",
            "status": "available_saved_representation_only",
            "includes_position_adjustment": True,
            "fielding_signal_counted_twice": True,
            "playing_time_dependent": True,
            "overall_year_to_year_spearman": defense_stability["estimate"],
            "ci_low": defense_stability["ci_low"],
            "ci_high": defense_stability["ci_high"],
            "reason": "Def = Fld + positional adjustment; averaging standardized Def and Fld repeats the Fld signal.",
        },
        {
            "variant": "Def_only",
            "status": "not_run_missing_raw",
            "includes_position_adjustment": True,
            "fielding_signal_counted_twice": False,
            "playing_time_dependent": True,
            "overall_year_to_year_spearman": np.nan,
            "reason": "Raw Def is absent from the committed processed input.",
        },
        {
            "variant": "Fld_only",
            "status": "not_run_missing_raw",
            "includes_position_adjustment": False,
            "fielding_signal_counted_twice": False,
            "playing_time_dependent": True,
            "overall_year_to_year_spearman": np.nan,
            "reason": "Raw Fld is absent from the committed processed input.",
        },
        {
            "variant": "fielding_only_position_excluded",
            "status": "not_run_missing_raw",
            "includes_position_adjustment": False,
            "fielding_signal_counted_twice": False,
            "playing_time_dependent": True,
            "overall_year_to_year_spearman": np.nan,
            "reason": "Equivalent to Fld-only in the documented source, but Fld is unavailable.",
        },
        {
            "variant": "innings_normalized_fielding",
            "status": "not_run_missing_raw",
            "includes_position_adjustment": False,
            "fielding_signal_counted_twice": False,
            "playing_time_dependent": False,
            "overall_year_to_year_spearman": np.nan,
            "reason": "Defensive innings and raw fielding runs are both absent.",
        },
    ]
    pd.DataFrame(defense_rows).to_csv(EXP_ROOT / "results" / "defense_variants.csv", index=False)
    print(stability.to_string(index=False))


if __name__ == "__main__":
    main()
