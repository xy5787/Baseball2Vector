"""Trace pooled 20--80 scaling and quantify a forecast-safe feature alternative."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .repro_common import DATA_PATH, REPO_ROOT, RUN_ROOT, SAFE_TOOL_COLS, TOOL_COLS, dump_json, sha256_file


def model_metrics(frame: pd.DataFrame) -> dict[str, float]:
    from .repro_common import metric

    return {
        name: metric(frame["y_true"].to_numpy(), frame["prediction"].to_numpy(), name)
        for name in ["mae", "r2", "spearman"]
    }


def main() -> None:
    players = pd.read_csv(DATA_PATH)
    matched = players[players["id_status"].eq("matched")].copy()
    source_code = REPO_ROOT / "src" / "baseball2vec" / "tools.py"
    build_code = REPO_ROOT / "src" / "baseball2vec" / "baselines.py"
    source_text = source_code.read_text(encoding="utf-8")
    raw_path = REPO_ROOT / "data" / "raw" / "batting_stats_2021_2025.csv"
    predictions = pd.read_csv(RUN_ROOT / "results" / "calibrated_predictions.csv")

    comparison_rows: list[dict] = []
    vector_models = ["B2V Ridge", "B2V + age + PA Ridge", "wRC+ + B2V + age + PA Ridge"]
    for variant in ["archived_pooled_20_80", "within_input_season_standardized"]:
        for model in vector_models:
            subset = predictions[(predictions["feature_variant"].eq(variant)) & predictions["model"].eq(model)]
            comparison_rows.append({"feature_variant": variant, "model": model, **model_metrics(subset)})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(RUN_ROOT / "results" / "scaling_feature_comparison.csv", index=False)

    diffs = pd.read_csv(RUN_ROOT / "results" / "scaling_prediction_differences.csv")
    max_diff_by_model = diffs.groupby("model")["safe_minus_archived"].apply(
        lambda x: float(np.max(np.abs(x)))
    ).to_dict()
    within_stats = {}
    for col in TOOL_COLS:
        grouped = matched.groupby("Season")[col]
        within_stats[col] = {
            "pooled_min": float(matched[col].min()),
            "pooled_max": float(matched[col].max()),
            "season_mean_range": [float(grouped.mean().min()), float(grouped.mean().max())],
            "season_sd_range_ddof0": [float(grouped.std(ddof=0).min()), float(grouped.std(ddof=0).max())],
        }

    audit = {
        "input": {
            "path": str(DATA_PATH), "sha256": sha256_file(DATA_PATH),
            "seasons": sorted(int(v) for v in matched["Season"].unique()),
            "n_matched_rows": int(len(matched)),
        },
        "implementation_trace": {
            "season_level_source_standardization": "src/baseball2vec/tools.py:56-72; raw features are z-scored within Season before dimension aggregation",
            "pooled_20_80_location": "src/baseball2vec/tools.py:91-106, called by src/baseball2vec/baselines.py:79-94 and scripts/02_baseline_comparison.py:59",
            "formula": "x_20_80 = 20 + 60 * (x - pooled_min_2021_2025) / (pooled_max_2021_2025 - pooled_min_2021_2025)",
            "pooled_years": [2021, 2022, 2023, 2024, 2025],
            "same_affine_map_all_seasons": True, "clipped": False,
            "evidence_for_no_clipping": "MinMaxScaler(feature_range=(20,80)) uses default clip=False; no clip argument or explicit clipping appears in tools.py",
            "source_code_checks": {
                "minmax_present": "MinMaxScaler(feature_range=(20, 80))" in source_text,
                "explicit_clip_present": "clip=" in source_text,
                "baselines_path": str(build_code),
            },
            "forecast_pipeline_second_scaler": "experiments/scripts/common.py ridge_pipeline: imputer then training-only StandardScaler inside each CV fold and final fit",
        },
        "availability": {
            "raw_constituent_cache_present": raw_path.exists(),
            "pre_minmax_dimension_scores_present": False,
            "reason": "Archived processed data contain only final 20-80 scores; raw cache and pre-min-max dimension scores are absent.",
        },
        "future_information": {
            "does_archived_2024_scaling_use_2025_extremes": True,
            "nature": "Only a shared per-dimension affine offset and scale; no clipping was applied.",
            "forecast_safe_variant": "standardize each dimension within its input-season cohort using population mean and SD (ddof=0), then fit preprocessing only on training transitions",
            "safe_columns": SAFE_TOOL_COLS,
            "exact_equivalence_expected": False,
            "why_not_exact": "The safe transform changes each season separately; the archived model's second StandardScaler pools training transitions.",
            "maximum_absolute_prediction_difference_by_model": max_diff_by_model,
            "primary_prediction_variant": "within_input_season_standardized",
            "wording_decision": "Use forecast-safe prospective wording; do not call archived globally scaled-vector results leakage-free.",
        },
        "archived_score_distribution": within_stats,
        "comparison_csv": str(RUN_ROOT / "results" / "scaling_feature_comparison.csv"),
    }
    dump_json(RUN_ROOT / "results" / "scaling_audit.json", audit)
    print(json.dumps(audit["future_information"], indent=2))
    print("\n", comparison.to_string(index=False))


if __name__ == "__main__":
    main()
